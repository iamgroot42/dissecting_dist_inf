import torch as ch
import numpy as np
import warnings
from tqdm import tqdm
from typing import List

from distribution_inference.attacks.whitebox.permutation.permutation import PINAttack
from distribution_inference.config import WhiteBoxAttackConfig
from distribution_inference.models.core import BaseModel
from distribution_inference.utils import warning_string


ATTACK_MAPPING = {
    "permutation_invariant": PINAttack,
}


def get_attack(attack_name: str):
    wrapper = ATTACK_MAPPING.get(attack_name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_name} not implemented")
    return wrapper


def prepare_batched_data(X,
                         reduce: bool = False,
                         verbose: bool = True):
    inputs = [[] for _ in range(len(X[0]))]
    iterator = X
    if verbose:
        iterator = tqdm(iterator, desc="Batching data")
    for x in iterator:
        for i, l in enumerate(x):
            inputs[i].append(l)

    inputs = np.array([ch.stack(x, 0) for x in inputs], dtype='object')
    if reduce:
        inputs = [x.view(-1, x.shape[-1]) for x in inputs]
    return inputs


def wrap_into_x_y(features_list: List,
                  labels_list: List[float] = [0., 1.]):
    """
        Wrap given features of models from N distributions
        into X and Y, to be used for model training. Use given list of
        labels for each distribution.
    """

    X, Y = [], []
    for features, label in zip(features_list, labels_list):
        X.append(features)
        Y.append([label] * len(features))

    X = np.concatenate(X, axis=0)
    X = prepare_batched_data(X)
    Y = ch.from_numpy(np.concatenate(Y, axis=0))
    Y = Y.float()

    return X, Y


def get_train_val_from_pool(features_list: List,
                            wb_config: WhiteBoxAttackConfig,
                            labels_list: List[float] = [0., 1.]):
    """
        Sample train and val data from pool of given data.
    """
    train_sample = wb_config.train_sample
    val_sample = wb_config.val_sample
    features_train, features_val = [], []
    for features, label in zip(features_list, labels_list):
        # Random shuffle
        shuffle_indices = np.random.permutation(len(features))

        # Pick train data
        indices_for_train = shuffle_indices[:train_sample]
        features_train.append(features[indices_for_train])

        if len(indices_for_train) != train_sample:
            warnings.warn(warning_string(
                f"\nNumber of models requested ({len(indices_for_train)}) for train shuffle is less than requested ({train_sample})"))

        if val_sample > 0:
            indices_for_val = shuffle_indices[train_sample:train_sample+val_sample]
            features_val.append(features[indices_for_val])

            if len(indices_for_val) != val_sample:
                warnings.warn(warning_string(
                    f"\nNumber of models requested ({len(indices_for_val)}) for val shuffle is less than requested ({val_sample})"))

    # Get train data
    train_data = wrap_into_x_y(features_train, labels_list)
    # Get val data
    val_data = None
    if val_sample > 0:
        val_data = wrap_into_x_y(features_val, labels_list)

    return train_data, val_data


def _get_weight_layers(model: BaseModel,
                       start_n: int = 0,
                       first_n: int = None,
                       custom_layers: List[int] = None,
                       include_all: bool = False,
                       is_conv: bool = False,
                       transpose_features: bool = True,
                       prune_mask=[]):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Treat 'None' as int
    first_n = np.inf if first_n is None else first_n

    # Sort and store desired layers, if specified
    custom_layers_sorted = sorted(
        custom_layers) if custom_layers is not None else None

    track = 0
    for name, param in model.named_parameters():
        if "weight" in name:

            param_data = param.data.detach().cpu()

            # Apply pruning masks if provided
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose_features:
                param_data = param_data.T

            weights.append(param_data)
            if is_conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

        # Assume each layer has weight & bias
        i += 1

        if custom_layers_sorted is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers_sorted[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

            # Break if all layers were processed
            if len(custom_layers_sorted) == j // 2:
                break

    if custom_layers_sorted is not None and len(custom_layers_sorted) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if is_conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    cctd = []
    for w, b in zip(weights, biases):
        if is_conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = ch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    if is_conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd


# Function to extract model parameters
def get_weight_layers(model: BaseModel,
                      attack_config: WhiteBoxAttackConfig,
                      prune_mask=[]):

    if model.is_conv:
        # Model has convolutional layers
        # Process FC and Conv layers separately
        dims_conv, fvec_conv = _get_weight_layers(
            model.features,
            first_n=attack_config.first_n_conv,
            start_n=attack_config.start_n_conv,
            is_conv=True,
            custom_layers=attack_config.custom_layers_conv,
            transpose_features=model.transpose_features,
            prune_mask=prune_mask,
            include_all=True)
        dims_fc, fvec_fc = _get_weight_layers(
            model.classifier,
            first_n=attack_config.first_n_fc,
            start_n=attack_config.start_n_fc,
            custom_layers=attack_config.custom_layers_fc,
            transpose_features=model.transpose_features,
            prune_mask=prune_mask,)
        feature_vector = fvec_conv + fvec_fc
        dimensions = (dims_conv, dims_fc)
    else:
        dims_fc, fvec_fc = _get_weight_layers(
            model,
            first_n=attack_config.first_n_fc,
            start_n=attack_config.start_n_fc,
            custom_layers=attack_config.custom_layers_fc,
            transpose_features=model.transpose_features,
            prune_mask=prune_mask,)
        feature_vector = fvec_fc
        dimensions = dims_fc

    return dimensions, feature_vector
