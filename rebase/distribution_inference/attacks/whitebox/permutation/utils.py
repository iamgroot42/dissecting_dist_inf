"""
    Utility functions for permutation-invariant meta-classifiers
"""
import torch as ch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy

import distribution_inference.utils as utils


def prepare_batched_data(X, reduce=False, verbose=True):
    """
        Given parameters of models, batch them layer-wise
        to use efficiently with the meta-classifier.
    """
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


def get_ratio_info_for_reg_meta(metamodel, X, Y, num_per_dist, batch_size, combined: bool = True):
    """
        Get MSE and actual predictions for each
        ratio given in Y, using a trained metamodel.
        Returns MSE per ratio, actual predictions per ratio, and
        predictions for each ratio a v/s be using regression
        meta-classifier for binary classification.
    """
    # Evaluate
    metamodel = metamodel.cuda()
    loss_fn = ch.nn.MSELoss(reduction='none')
    _, losses, preds = test_meta(
        metamodel, loss_fn, X, Y.cuda(),
        batch_size, None,
        binary=True, regression=True, gpu=True,
        combined=combined, element_wise=True,
        get_preds=True)
    y_np = Y.numpy()
    losses = losses.numpy()
    # Get all unique ratios (sorted) in GT, and their average losses from model
    ratios = np.unique(y_np)
    losses_dict = {}
    ratio_wise_preds = {}
    for ratio in ratios:
        losses_dict[ratio] = np.mean(losses[y_np == ratio])
        ratio_wise_preds[ratio] = preds[y_np == ratio]
    # Conctruct a matrix where every (i, j) entry is the accuracy
    # for ratio[i] v/s ratio [j], where whichever ratio is closer to the
    # ratios is considered the "correct" one
    # Assume equal number of models per ratio, stored in order of
    # ratios
    acc_mat = np.zeros((len(ratios), len(ratios)))
    for i in range(acc_mat.shape[0]):
        for j in range(i + 1, acc_mat.shape[0]):
            # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
            gt_z = (y_np[num_per_dist * i:num_per_dist * (i + 1)]
                    == float(ratios[j]))
            gt_o = (y_np[num_per_dist * j:num_per_dist * (j + 1)]
                    == float(ratios[j]))
            # Get relevant preds
            pred_z = preds[num_per_dist * i:num_per_dist * (i + 1)]
            pred_o = preds[num_per_dist * j:num_per_dist * (j + 1)]
            pred_z = (pred_z >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            pred_o = (pred_o >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            # Compute accuracies and store
            acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate(
                (pred_z, pred_o), 0)
            acc_mat[i, j] = np.mean(acc)

    return losses_dict, acc_mat, ratio_wise_preds


def get_preds(model, X, batch_size, on_gpu=True):
    """
        Get predictions for meta-classifier.
        Parameters:
            model: Model to get predictions for.
            X: Data to get predictions for.
            batch_size: Batch size to use.
            on_gpu: Whether to use GPU.
    """
    # Get predictions for model
    preds = []
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        if on_gpu:
            x_batch = [x.cuda() for x in x_batch]
        batch_preds = model(x_batch)
        preds.append(batch_preds.detach())
    return ch.cat(preds, 0)


# Function to extract model parameters
def get_weight_layers(m, normalize=False, transpose=True,
                      first_n=np.inf, start_n=0,
                      custom_layers=None,
                      conv=False, include_all=False,
                      prune_mask=[]):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Sort and store desired layers, if specified
    custom_layers = sorted(custom_layers) if custom_layers is not None else None

    track = 0
    for name, param in m.named_parameters():
        if "weight" in name:

            param_data = param.data.detach().cpu()

            # Apply pruning masks if provided
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose:
                param_data = param_data.T

            weights.append(param_data)
            if conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

        # Assume each layer has weight & bias
        i += 1

        if custom_layers is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

            # Break if all layers were processed
            if len(custom_layers) == j // 2:
                break

    if custom_layers is not None and len(custom_layers) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    if normalize:
        min_w = min([ch.min(x).item() for x in weights])
        max_w = max([ch.max(x).item() for x in weights])
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
        weights = [w / max_w for w in weights]

    cctd = []
    for w, b in zip(weights, biases):
        if conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = ch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    if conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd
