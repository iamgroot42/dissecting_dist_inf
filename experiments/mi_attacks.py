"""
    A lot of the code (especially interpolation) is borrowed from
    https://github.com/privacytrustlab/ml_privacy_meter/
"""

from re import L
import numpy as np
import torch as ch
from scipy.stats import norm
from tqdm import tqdm
from dataclasses import replace
from pathlib import Path
import torch.nn as nn
from typing import List
from simple_parsing import ArgumentParser

from distribution_inference.datasets.utils import get_dataset_wrapper
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import AttackConfig


def linear_itp_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs
) -> float:
    distribution = np.append(distribution, signal_min)
    distribution = np.append(distribution, signal_max)
    threshold = np.quantile(distribution, q=alpha, interpolation='linear',**kwargs)

    return threshold


def logit_rescale_threshold_func(
        distribution: List[float],
        alpha: List[float],
        **kwargs,
) -> float:
    distribution = np.log(
        np.divide(np.exp(- distribution), (1 - np.exp(- distribution))))
    len_dist = len(distribution)
    loc, scale = norm.fit(distribution)

    threshold = norm.ppf(1 - np.array(alpha), loc=loc, scale=scale)
    threshold = np.log(np.exp(threshold) + 1) - threshold
    return threshold


def min_linear_logit_threshold_func(
        distribution: List[float],
        alpha: List[float],
        signal_min=0,
        signal_max=1000,
        **kwargs,
) -> float:
    distribution_linear = np.append(distribution, signal_min)
    distribution_linear = np.append(distribution_linear, signal_max)
    threshold_linear = np.quantile(distribution_linear, q=alpha, interpolation='linear',**kwargs,)

    distribution = np.log(np.divide(np.exp(- distribution), (1 - np.exp(- distribution))))
    len_dist = len(distribution)
    loc, scale = norm.fit(distribution,**kwargs,)
    threshold_logit = norm.ppf(1 - alpha, loc=loc, scale=scale)
    threshold_logit = np.log(np.exp(threshold_logit) + 1) - threshold_logit

    threshold = min(threshold_logit, threshold_linear)

    return threshold

def predict_member(values, threshold: float):
    """
        Predict membership based on threshold
    """
    return values <= threshold


def get_thresholds(loss_values, alpha):
    # Look at distribution of loss values, and a threshold for given FPR
    # For each datapoint
    thresholds = []
    for i in range(loss_values.shape[0]):
        thresholds.append([min_linear_logit_threshold_func(loss_values[i, j], alpha) for j in range(loss_values.shape[1])])
    return np.array(thresholds)


def prior_knowledge(loader, n_per_dist):
    """
        Return n points each from D-, D+ for which membership is known
    """
    X, Y, P = [], [], []
    for (x, y, p) in loader:
        X.append(x)
        Y.append(y)
        P.append(p)
    P = ch.cat(P, axis=0).numpy()
    X = ch.cat(X, axis=0)
    Y = ch.cat(Y, axis=0)
    p_zero = np.where(P == 0)[0]
    p_one = np.where(P == 1)[0]
    p_zero = np.random.choice(p_zero, n_per_dist, replace=False)
    p_one = np.random.choice(p_one, n_per_dist, replace=False)
    indices_picked = np.concatenate([p_zero, p_one])
    return (X[p_zero], Y[p_zero]), (X[p_one], Y[p_one]), indices_picked


@ch.no_grad()
def get_loss_values(models, criterion, prior_data_one_x, prior_data_one_y, prior_data_zero_x, prior_data_zero_y):
    # Get loss values for prior knowledge
    losses_zero, losses_one = [], []
    for model_adv in tqdm(models, desc="Collecting loss values from shadow models"):
        model_adv.eval()
        lz_inner, lo_inner = [], []
        for pzx, pzy, pox, poy in zip(prior_data_zero_x, prior_data_zero_y, prior_data_one_x, prior_data_one_y):
            pz_out = model_adv(pzx.cuda()).detach()
            po_out = model_adv(pox.cuda()).detach()
            if pz_out.shape[1] == 1: # Squeeze if binary task (binary loss used with it)
                pz_out = pz_out.squeeze(1)
                po_out = po_out.squeeze(1)
            loss_zero = criterion(pz_out, pzy.cuda())
            loss_one = criterion(po_out, poy.cuda())
            lz_inner.append(loss_zero.cpu().numpy())
            lo_inner.append(loss_one.cpu().numpy())
        losses_zero.append(lz_inner)
        losses_one.append(lo_inner)
    losses_zero = np.array(losses_zero)
    losses_one = np.array(losses_one)
    return losses_zero, losses_one


@ch.no_grad()
def get_loss_values_victim(models, criterion, prior_data_one_x, prior_data_one_y, prior_data_zero_x, prior_data_zero_y):
    # Get loss values for prior knowledge
    losses_zero, losses_one = [], []
    for i, model_vic in tqdm(enumerate(models), desc="Collecting loss values from victim models"):
        model_vic.eval()
        pz_out = model_vic(prior_data_zero_x[i].cuda()).detach()
        # Weird bug
        if ch.sum(ch.isnan(pz_out)):
            pz_out = model_vic(prior_data_zero_x[i].cuda()).detach()
        po_out = model_vic(prior_data_one_x[i].cuda()).detach()
        if pz_out.shape[1] == 1: # Squeeze if binary task (binary loss used with it)
            pz_out = pz_out.squeeze(1)
            po_out = po_out.squeeze(1)
        loss_zero = criterion(pz_out, prior_data_zero_y[i].cuda())
        loss_one = criterion(po_out, prior_data_one_y[i].cuda())
        losses_zero.append(loss_zero.cpu().numpy())
        losses_one.append(loss_one.cpu().numpy())
    losses_zero = np.array(losses_zero)
    losses_one = np.array(losses_one)
    return losses_zero, losses_one


def mi_attacks_on_ratio(attack_config, ratio: float, n_per_dist: int = 20, alpha: float = 0.01):
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(attack_config.train_config.data_config.name)

    data_config_to_use = replace(attack_config.train_config.data_config, value=ratio)
    train_config_to_use = replace(attack_config.train_config, data_config=data_config_to_use)

    # Create new DS object for both adv and victim
    data_config_adv, data_config_vic = get_dfs_for_victim_and_adv(
        data_config_to_use)

    # Load up victim models
    # Create new DS object for both adv and victim
    ds_vic = ds_wrapper_class(
        data_config_vic, skip_data=False,
        label_noise=train_config_to_use.label_noise,
        epoch=train_config_to_use.save_every_epoch)
    ds_adv = ds_wrapper_class(data_config_adv)
    train_adv_config = get_train_config_for_adv(train_config_to_use, attack_config)
    train_adv_config = replace(train_adv_config, offset=0)

    # Load victim models
    models_vic, (ids_before, ids_after) = ds_vic.get_models(
        train_config_to_use,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        model_arch=attack_config.victim_model_arch,)
    
    # Load adv models
    models_adv = ds_adv.get_models(
        train_adv_config,
        n_models=attack_config.black_box.num_adv_models,
        on_cpu=attack_config.on_cpu,
        model_arch=attack_config.adv_model_arch,
        target_epoch = attack_config.adv_target_epoch)
    
    # Get loss values for data corresponding to ids_before and ids_after
    # Use adv models for these loss values
    if attack_config.train_config.multi_class:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    
    # Get "prior knowledge" per victim model
    prior_data_zero_x, prior_data_zero_y = [], []
    prior_data_one_x, prior_data_one_y = [], []
    indices_information = []
    for idb, ida in tqdm(zip(ids_before, ids_after), desc="Collecting data", total=len(ids_before)):
        loader_train, _ = ds_vic.get_loaders(batch_size=attack_config.train_config.batch_size,
                                             indexed_data=idb,
                                             shuffle=False)
        zero, one, ids_used = prior_knowledge(loader_train, n_per_dist)
        prior_data_zero_x.append(zero[0])
        prior_data_zero_y.append(zero[1])
        prior_data_one_x.append(one[0])
        prior_data_one_y.append(one[1])
        # Take not of indices information for train data
        ids_to_watch_for = idb[0][ids_used]
        ids_after_mask = 1 * np.isin(ids_to_watch_for, ida[0])
        indices_information.append(ids_after_mask)

    # Get loss values for prior knowledge (adv)
    losses_adv_zero, losses_adv_one = get_loss_values(models_adv, criterion,
                                                      prior_data_one_x, prior_data_one_y,
                                                      prior_data_zero_x, prior_data_zero_y)

    # Train MI attack thresholds using MI knowledge of 'before' members
    loss_values_together = np.concatenate((losses_adv_zero, losses_adv_one), axis=-1)
    loss_values_together = loss_values_together.transpose(1, 2, 0)

    thresholds = get_thresholds(loss_values_together, alpha)
    thresholds_zero = thresholds[:, :losses_adv_zero.shape[2]]
    thresholds_one = thresholds[:, losses_adv_one.shape[2]:]

    # Get loss values from victim model (to predict on)
    losses_vic_zero, losses_vic_one = get_loss_values_victim(models_vic, criterion,
                                                             prior_data_one_x, prior_data_one_y,
                                                             prior_data_zero_x, prior_data_zero_y)

    # Predict membership for all members
    members_zero = 1 * np.sum(losses_vic_zero <= thresholds_zero, axis=-1)
    members_one = 1 * np.sum(losses_vic_one <= thresholds_one, axis=-1)
    
    ratios = np.zeros_like(1. * members_zero)
    zero_top = (members_zero > members_one)
    one_top = (members_zero <= members_one)
    ratios[zero_top] = 1. * members_zero[zero_top] / (members_zero[zero_top] + members_one[zero_top])
    ratios[one_top] = 1. * members_one[one_top] / (members_zero[one_top] + members_one[one_top])
    # Cases where both are zero, just defer to predicting 0.5 (guess)
    ratios[np.isnan(ratios)] = 0.5

    # Make binary-based prediction using the ratio here
    num_predicted = np.sum(np.abs(ratios - 0.5) > 0.02)

    mse = np.sum((ratio - ratios) ** 2)
    return mse, num_predicted


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    args = parser.parse_args()
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Run MI attack
    mses = []
    neffs = []
    preds = []
    for value in attack_config.values:
        mse, num_predicted = mi_attacks_on_ratio(attack_config, ratio=value, n_per_dist=100, alpha=0.12)
        mses.append(mse)
        preds.append(num_predicted)
        neff = (value * (1 - value)) / mse
        neffs.append(neff)
        print(value, ":", mse)

    print("Final MSE value: ", np.mean(mses))
    print("Final n_eff: ", np.mean(neff))
    print("Num predicted as (not 0.5):", preds)
