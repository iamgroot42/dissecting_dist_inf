"""
    Utility functions for permutation-invariant meta-classifiers
"""
import torch as ch
from tqdm import tqdm
import numpy as np


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
