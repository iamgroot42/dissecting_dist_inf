from model_utils import load_model, get_models_path
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import perpoint_threshold_test, threshold_and_loss_test, flash_utils, ensure_dir_exists
from typing import List
import torch.nn as nn
import torch as ch


def get_models(folder_path, n_models=1000):

    paths = np.random.permutation(os.listdir(folder_path))
    i = 0
    models = []
    for mpath in tqdm(paths):
        if i >= n_models:
            break
        if os.path.isdir(os.path.join(folder_path, mpath)):
            continue
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
        i += 1
    return models


def calculate_accuracies(data, labels, use_logit: bool = True):
    """
        Function to compute model-wise average-accuracy on
            given data.
    """
    # Get predictions from each model (each model outputs logits)
    if use_logit:
        preds = (data >= 0).astype('int')
    else:
        preds = (data >= 0.5).astype('int')

    # Repeat ground-truth (across models)
    expanded_gt = np.repeat(np.expand_dims(
        labels, axis=1), preds.shape[1], axis=1)

    return np.average(1. * (preds == expanded_gt), axis=0)


def get_preds(loader, models: List[nn.Module]):
    """
        Get predictions for given models on given data
    """
    predictions = []
    inputs = []
    ground_truth = []
    # Accumulate all data for given loader
    for data in loader:
        data_points, labels, _ = data
        inputs.append(data_points.cuda())
        ground_truth.append(labels.cpu().numpy())
        #print(labels.size())

    # Get predictions for each model
    for model in tqdm(models):
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        with ch.no_grad():
            predictions_on_model = []
            for data in inputs:
                predictions_on_model.append(model(data).detach()[:, 0])
        predictions_on_model = ch.cat(predictions_on_model)
        predictions.append(predictions_on_model)

    predictions = ch.stack(predictions, 0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    #ground_truth = np.array(ground_truth)
    #print(ground_truth.size)
    return predictions.cpu().numpy(), ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    parser.add_argument('--drop', action="store_true")
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--ratios', help="ratio of data points to try",
                        nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    parser.add_argument('--dp', type=float, default=None)
    parser.add_argument('--b', action="store_true")
    parser.add_argument('--gpu',
                        default='0', help="device number")
    args = parser.parse_args()
    flash_utils(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ratios = args.ratios
    adp = None
    if args.b:
        adp = "DP_%.2f" % args.dp
    dp = None
    if args.dp:
        dp = "DP_%.2f" % args.dp
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1, is_dp=dp))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2, is_dp=dp))
    thre, perp, bas = [], [], []
    for _ in range(args.tries):
        # Load adv models
        total_models = 100
        models_1 = get_models(get_models_path(
            args.filter, "adv", args.ratio_1, is_dp=adp), total_models // 2)
        models_2 = get_models(get_models_path(
            args.filter, "adv", args.ratio_2, is_dp=adp), total_models // 2)
        ds_1 = CensusWrapper(
            filter_prop=args.filter,
            ratio=float(args.ratio_1), split="adv",
            drop_senstive_cols=args.drop,
            scale=args.scale)
        ds_2 = CensusWrapper(
            filter_prop=args.filter,
            ratio=float(args.ratio_2), split="adv",
            drop_senstive_cols=args.drop,
            scale=args.scale)

        # Fetch test data from both ratios
        _, l1 = ds_1.get_loaders(2000, squeeze=True)
        _, l2 = ds_2.get_loaders(2000, squeeze=True)
        pred1 = get_preds(l1, models_1)
        y_te_1 = pred1[1]
        y_te_1 = y_te_1.flatten()
        pred2 = get_preds(l2, models_1)
        y_te_2 = pred2[1].flatten()
        p1 = [pred1[0], get_preds(l1, models_2)[0]]
        p2 = [pred2[0], get_preds(l2, models_2)[0]]
        pv1 = [get_preds(l1, models_victim_1)[0],
               get_preds(l1, models_victim_2)[0]]
        pv2 = [get_preds(l2, models_victim_1)[0],
               get_preds(l2, models_victim_2)[0]]
        (vacc, ba), _ = threshold_and_loss_test(calculate_accuracies, (p1, p2),
                                                (pv1, pv2),
                                                (y_te_1, y_te_2),
                                                ratios, granularity=0.05)
        thre.append(vacc)
        bas.append(ba)
        (vpacc, _), _, _ = perpoint_threshold_test((p1, p2),
                                                   (pv1, pv2),
                                                   # (y_te_1, y_te_2),
                                                   ratios, granularity=0.05)
        perp.append(vpacc)

    l = "./log"
    if args.b:
        l = os.path.join(l, "both_dp")
    if args.dp:
        l = os.path.join(l, dp)
    if args.scale != 1:
        l = os.path.join(l, 'sample_size_scale:{}'.format(args.scale))
    if args.drop:
        l = os.path.join(l, 'drop')
    content = 'Perpoint thresholds accuracy: {}'.format(perp)
    print(content)

    log_path = os.path.join(
        l, "perf_perpoint_{}:{}".format(args.filter, args.ratio_1))
    ensure_dir_exists(log_path)
    with open(os.path.join(log_path, args.ratio_2), "w") as wr:
        wr.write(content)
    log_path = os.path.join(
        l, "selective_loss_{}:{}".format(args.filter, args.ratio_1))

    cl = 'basline accuracy: {}'.format(bas)
    print(cl)

    ensure_dir_exists(log_path)
    with open(os.path.join(log_path, args.ratio_2), "w") as wr:
        wr.write(cl)

    content = 'thresholds accuracy: {}'.format(thre)
    print(content)

    log_path = os.path.join(
        l, "perf_quart_{}:{}".format(args.filter, args.ratio_1))
    ensure_dir_exists(log_path)
    with open(os.path.join(log_path, args.ratio_2), "w") as wr:
        wr.write(content)
