"""
    Analyze overlap in model predictions using various meta-classification methods.
    Useful tp compare different methods and predict utility in trying to
    combine them.
"""
import utils
import numpy as np
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
from model_utils import get_models_path, get_model_representations, get_models, get_models_path
import argparse
from tqdm import tqdm
import torch as ch


def get_preds(x, ms):
    ps = []
    for m in tqdm(ms):
        p = m.predict_proba(x)[:, 1]
        ps.append(p)
    return np.squeeze(np.array(ps))


def get_best_blackbox_results(adv_1, adv_2, vic_1, vic_2, args):
    """
        Launch best black-box attack for this dataset and
        return accuracy, along with model-wise predictions.
    """
    # Prepare data wrappers
    ds_1 = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.d_0), split="adv")
    ds_2 = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.d_1), split="adv")
    # Fetch test data from both ratios
    _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
    _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
    y_te_1 = y_te_1.ravel()
    y_te_2 = y_te_2.ravel()

    # Get predictions on adversary's models
    preds_1 = [get_preds(x_te_1, adv_1), get_preds(x_te_1, adv_2)]
    preds_2 = [get_preds(x_te_2, adv_1), get_preds(x_te_2, adv_2)]
    # Get predictions on victim's models
    preds_vic_1 = [get_preds(x_te_1, vic_1), get_preds(x_te_1, vic_1)]
    preds_vic_2 = [get_preds(x_te_2, vic_1), get_preds(x_te_2, vic_2)]

    # Get predictions using perpoint-threshold test
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    (vic_acc, vic_preds), _ = utils.perpoint_threshold_test(
        (preds_1, preds_2),
        (preds_vic_1, preds_vic_2),
        (y_te_1, y_te_2),
        ratios, granularity=0.005)

    return vic_preds, vic_acc


def get_best_whitebox_results(dims, vic_1, vic_2, args):
    """
        Launch best white-box attack for this dataset and
        return accuracy, along with model-wise predictions.
    """
    # Load meta-classifier
    metamodel = utils.PermInvModel(dims, dropout=0.5)
    # Load up state dict
    metamodel.load_state_dict(ch.load(args.meta))
    metamodel = metamodel.cuda()
    metamodel.eval()
    # Batch up data
    X = np.concatenate((vic_1, vic_2))
    X = utils.prepare_batched_data(X)
    # Get predictions on victim's models
    preds = utils.get_meta_preds(metamodel, X, args.batch_size)
    preds = 1. * (preds >= 0)
    return preds.cpu().numpy()[:, 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', help='path to meta-classifier')
    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='batch-size for meta-classifier inference')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--d_0', default="0.5",
                        help='ratios to use for D_0')
    parser.add_argument('--d_1', default=None, help='target ratio')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Load up test data with their features
    pos_w_test, pos_labels_test, dims, pos_models_vic = get_model_representations(
        get_models_path("victim", args.filter, args.d_0), 1, fetch_models=True,
        shuffle=False)
    neg_w_test, neg_labels_test, dims, neg_models_vic = get_model_representations(
        get_models_path("victim", args.filter, args.d_1), 0, fetch_models=True,
        shuffle=False)

    # Load up "train" data for Threshold Test
    total_models = 100
    pos_models_adv = get_models(get_models_path("adv", args.filter,
                                                value=args.d_0),
                                n_models=total_models // 2, shuffle=True)
    neg_models_adv = get_models(get_models_path("adv", args.filter,
                                                value=args.d_1),
                                n_models=total_models // 2, shuffle=True)

    # Get black-box test predictions
    print("Getting predictions for black-box test")
    black_preds, _ = get_best_blackbox_results(pos_models_adv, neg_models_adv,
                                               pos_models_vic, neg_models_vic,
                                               args)

    # Get meta-classifier predictions
    print("Getting predictions for white-box test")
    white_preds = get_best_whitebox_results(
        dims, pos_w_test, neg_w_test, args)

    ground_truth = np.concatenate((pos_labels_test, neg_labels_test))

    white_right = (white_preds == ground_truth)
    black_right = (black_preds == ground_truth)

    white_wrong_black_right = np.sum(np.logical_not(white_right) & black_right)
    white_right_black_wrong = np.sum(white_right & np.logical_not(black_right))
    white_right_black_right = np.sum(
        (white_preds == ground_truth) & (black_preds == ground_truth))
    max_together = np.sum(white_right) + np.sum(black_right) - white_right_black_right

    print("White-box accuracy", np.mean(ground_truth == white_preds))
    print("Black-box accuracy",np.mean(ground_truth == black_preds))
    print("Maximim possible accuracy (%) on combining both", max_together / len(ground_truth))

    gain = (max_together - max(np.sum(white_right),
                               np.sum(black_right))) / len(ground_truth)
    print("Potential accuracy gain (%) in combining both accuracies:", gain)
