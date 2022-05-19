import torch as ch
import warnings
from typing import List
from torch.utils.data import DataLoader
import numpy as np

from distribution_inference.attacks.whitebox.affinity.affinity import AffinityMetaClassifier
from distribution_inference.attacks.whitebox.utils import BasicDataset
from distribution_inference.datasets.base import CustomDatasetWrapper
from distribution_inference.datasets.utils import collect_data, worker_init_fn
from distribution_inference.training.core import train
from distribution_inference.config import WhiteBoxAttackConfig
from distribution_inference.utils import warning_string
from distribution_inference.models.core import BaseModel
from distribution_inference.attacks.blackbox.utils import get_preds
from distribution_inference.attacks.blackbox.core import order_points


def get_seed_data_loader(ds_list: List[CustomDatasetWrapper],
                         attack_config: WhiteBoxAttackConfig,
                         num_samples_use: int = None,
                         adv_models: List[BaseModel] = None,
                         also_get_raw_data: bool = False):
    """
        Collect data from given datasets and wrap into a dataloader.
    """
    warnings.warn(warning_string("\nCollecting seed data\n"))
    all_data = []
    # For each given loader
    for ds in ds_list:
        # Use val-loader and collect all data
        _, test_loader = ds.get_loaders(
            attack_config.batch_size,
            eval_shuffle=True)

        # Collect this data
        data, _ = collect_data(test_loader)

        # Collect preds on adv data, if given
        # Use these for picking top points
        if num_samples_use is not None:

            if num_samples_use > len(data):
                warnings.warn(warning_string(
                    f"\nRequested using {num_samples_use} samples, but only {len(data)} samples available for {ds}.\n"))

            if adv_models is not None:
                # Use per-point threshold test criteria to pick points, instead of random sample
                if len(adv_models) > 2:
                    raise NotImplementedError("Per-point based selection not supported for regression")
                preds_1, _ = get_preds(
                    test_loader, adv_models[0],
                    verbose=True, multi_class=attack_config.multi_class)
                preds_2, _ = get_preds(
                    test_loader, adv_models[1],
                    verbose=True, multi_class=attack_config.multi_class)
                ordering = order_points(preds_1, preds_2)
                data = data[ordering][:num_samples_use]
            else:
                # Randomly pick num_samples_use samples
                data = data[np.random.choice(
                    data.shape[0], num_samples_use, replace=False)]

        all_data.append(data)

    all_data = ch.cat(all_data, dim=0)

    # Create a dataset out of this
    basic_ds, loader = make_ds_and_loader(all_data, attack_config)
    print(warning_string(f"Seed data has {len(basic_ds)} samples."))
    # Get loader using given dataset
    if also_get_raw_data:
        return basic_ds, loader, all_data
    return basic_ds, loader


def make_ds_and_loader(data, attack_config, specific_ids=None):
    if specific_ids is not None:
        data = data[specific_ids]
    ds = BasicDataset(data)
    loader = get_loader_for_seed_data(ds, attack_config)
    return ds, loader


def get_loader_for_seed_data(seed_data_ds,
                             attack_config: WhiteBoxAttackConfig):
    """
        Wrap seed data in loader and return
    """
    loader = DataLoader(
        seed_data_ds,
        batch_size=attack_config.batch_size,
        # batch_size=32,
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2)
    return loader


def wrap_into_x_y(features_list: List,
                  labels_list: List[float] = [0., 1.]):
    """
        Wrap given data into X & Y
    """
    Y = []
    for features, label in zip(features_list, labels_list):
        Y.append([label] * len(features))

    X = ch.cat(features_list, dim=0).float()
    Y = ch.from_numpy(np.concatenate(Y, axis=0))

    return X, Y


def identify_relevant_points(features, total_points: int, num_points: int):
    """
        Use heuristic to order points by their 'usefulness'
        for the meta-classifier, and return the top num_points
        points.
    """
    # Features has shape (n_models, n_layers, n_points - 1, *)
    # Where each of * is decreasing length, down till 1
    num_models = len(features)
    num_each = num_models // 2
    num_layers = len(features[0])
    point_scores = ch.zeros(num_layers, total_points)
    mean_vals, std_vals = [], []
    max_mean = 0
    # For each layer
    for i in range(num_layers):
        # Matrix to track cosine scores
        # Assume we have num_points magically
        cos_scores = ch.ones(num_models, total_points, total_points)
        # Populate matrix for each point
        xx, yy = ch.triu_indices(total_points - 1, total_points - 1)
        yy = yy + 1
        for j in range(num_models):
            cos_scores[j][xx, yy] = features[j][i]
            temp_mat = cos_scores[j]
            # Main-diagonal will always be zero
            # and is excluded in both temp_mat and temp_mat.T
            cos_scores[j] = temp_mat + temp_mat.T - ch.diag(ch.diag(temp_mat))
        # cos_scores is now a symmetrix matrix per model
        # split into both models
        cos_scores_1 = cos_scores[:num_each]
        cos_scores_2 = cos_scores[num_each:]
        std_vals.append(
            cos_scores_1.std(2).std(0) +
            cos_scores_2.std(2).std(0))
        mean_vals.append(ch.abs(cos_scores_1.mean(
            2).mean(0) - cos_scores_2.mean(2).mean(0)))
        max_mean = max(max_mean, mean_vals[-1].max())
    # Normalize and sum scores
    for i in range(num_layers):
        point_scores[i] = std_vals[i] + (max_mean - mean_vals[i])
    # Retain 'n_points' per layer
    point_scores = ch.argsort(point_scores, 1)[:, :num_points]
    # Make a count array to keep track of top pairs per layer
    count_array = np.zeros((total_points,))
    for i in range(num_layers):
        count_array[point_scores[i]] += 1
    # Ouf of these, pick the top self.frac_retain_pairs
    retained_points = np.sort(np.argsort(count_array)[::-1][:num_points])
    return retained_points


def coordinate_descent(models_train,
                       models_val,
                       num_features, num_layers,
                       get_features,
                       meta_train_args,
                       gen_optimal_fn, seed_data,
                       n_times: int = 10,
                       restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        num_layers: Number of layers of models used for activations
        get_features: Function that takes (models, data) as input and returns features
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = AffinityMetaClassifier(num_features, num_layers)
    metamodel = metamodel.cuda()

    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        train_loader = get_features(
            models_train[0], models_train[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])
        val_loader = get_features(
            models_val[0], models_val[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = AffinityMetaClassifier(num_features, num_layers)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        _, val_acc = train(metamodel, (train_loader, val_loader),
                           epoch_num=meta_train_args['epochs'],
                           expect_extra=False,
                           verbose=False)
        all_accs.append(val_acc)

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data)

    # Return all accuracies
    return all_accs
