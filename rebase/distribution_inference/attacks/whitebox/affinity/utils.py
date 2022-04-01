from distribution_inference.config.core import TrainConfig
import torch as ch
import warnings
from typing import List
import numpy as np

from distribution_inference.attacks.whitebox.affinity.affinity import AffinityMetaClassifier
from distribution_inference.datasets.base import CustomDatasetWrapper
from distribution_inference.datasets.utils import collect_data
from distribution_inference.training.core import train
from distribution_inference.config import TrainConfig
from distribution_inference.utils import warning_string


def get_seed_data(ds_list: List[CustomDatasetWrapper],
                  train_config: TrainConfig,
                  num_samples_use: int = None):
    all_data = []
    # For each given loader
    for ds in ds_list:
        # Use val-loader and collect all data
        _, test_loader = ds.get_loaders(
            train_config.batch_size,
            eval_shuffle=False)
        # Collect this data
        data, _ = collect_data(test_loader)
        # Randomly pick num_samples_use samples
        if num_samples_use is not None:
            if num_samples_use > len(data):
                warnings.warn(warning_string(
                    f"\nRequested using {num_samples_use} samples, but only {len(data)} samples available for {ds}.\n"))
            data = data[np.random.choice(
                data.shape[0], num_samples_use, replace=False)]

        all_data.append(data)
    return all_data


def wrap_into_x_y(features_list: List,
                  labels_list: List[float] = [0., 1.]):
    Y = []
    for features, label in zip(features_list, labels_list):
        Y.append([label] * len(features))

    X = ch.cat(features_list, dim=0).float()
    Y = ch.from_numpy(np.concatenate(Y, axis=0))

    return X, Y


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
