"""
    A lot of the code (especially interpolation) is borrowed from
    https://github.com/privacytrustlab/ml_privacy_meter/
"""

import numpy as np
import torch as ch
from scipy.stats import norm, lognorm
from typing import List


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


def get_loss_values(models, data_x, data_y):
    loss_values = []
    for model in models:
        loss = ch.nn.BCEWithLogitsLoss(reduction='none')(model(data_x.cuda()), data_y.cuda())
        loss_values.append(loss.cpu().numpy())
    return np.array(loss_values)


def predict_member(values, alpha):
    return values <= alpha


def get_thresholds(loss_values, alpha):
    # Look at distribution of loss values, and a threshold for given FPR
    # For each datapoint
    thresholds = [min_linear_logit_threshold_func(loss_values[:, i], alpha) for i in range(loss_values.shape[1])]
    return np.array(thresholds)


def mi_attack(target_models, adv_models, data_x, data_y, data_attr, alpha):
    """
        Main idea is to look at MI risk for members from D- and D+, 
        look at their relative success, and compare that to the setting
        where no re-sampling is done.
    """
    # Get loss values for adv models
    loss_values = get_loss_values(adv_models, data_x, data_y)
    # Get threshold for each datapoint
    thresholds = get_thresholds(loss_values, alpha)
    # Get loss values on target models
    loss_values_target = get_loss_values(target_models, data_x, data_y)
    # Predict membership
    predictions = predict_member(loss_values_target, thresholds)
    return predictions


def d_attack():
    # if l(x_z, y_z) < c_alpha(model, x_z, y_z), reject H_0
    return