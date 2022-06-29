"""
    Meta-classifier experiment using Permutation Invariant Networks for direct regression.
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import utils
from data_utils import SUPPORTED_RATIOS
from model_utils import get_model_features, BASE_MODELS_DIR
import argparse
import numpy as np
import os
import pandas as pd
import torch as ch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


def load_stuff(model_dir, args, max_read):
    dims, vecs = get_model_features(model_dir, first_n=3,
                                    shift_to_gpu=False, max_read=max_read)
    return dims, np.array(vecs, dtype=object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RSNA BoneAge')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--testing', action="store_true", help="Testing mode")
    parser.add_argument('--ratios', help="Ratios to use", default=None)
    args = parser.parse_args()
    utils.flash_utils(args)

    ratios_to_use = set(args.ratios.strip().split(','))
    model_dir = f"log/meta/regression_models/"

    X_test, Y_test = [], []
    X_test_og, Y_test_og = [], []
    all_ratios_sorted = set(ratios_to_use).union(set(SUPPORTED_RATIOS))
    all_ratios_sorted = sorted([float(x) for x in all_ratios_sorted])
    palette = {x: f"C{i}" for i, x in enumerate(all_ratios_sorted)}
    for ratio in ratios_to_use:
        # Load up data for this ratio
        test_dir = os.path.join(BASE_MODELS_DIR, "bonemodel/victim/gender/%s/" % ratio)
        dims, vecs_test = load_stuff(test_dir, args, max_read=100)
        X_test.append(vecs_test)
        Y_test += [float(ratio)] * len(vecs_test)

        # Make sure same number of models read per distribution
        assert 100 == len(vecs_test)

    for ratio in SUPPORTED_RATIOS:
        # Load up data for this ratio
        test_dir = os.path.join(BASE_MODELS_DIR, "bonemodel/victim/gender/%s/" % ratio)
        dims, vecs_test = load_stuff(test_dir, args, max_read=1000)
        X_test_og.append(vecs_test)
        Y_test_og += [float(ratio)] * len(vecs_test)
        print(ratio, len(vecs_test))

        # Make sure same number of models read per distribution
        assert 1000 == len(vecs_test)

    X_test = np.concatenate(X_test)
    Y_test = ch.from_numpy(np.array(Y_test)).float()
    X_test_og = np.concatenate(X_test_og)
    Y_test_og = ch.from_numpy(np.array(Y_test_og)).float()

    data = []
    columns = [r"Actual $\alpha$", r"Predicted $\alpha$"]
    mses = {}
    for meta_path in os.listdir(model_dir):
        metamodel_path = os.path.join(model_dir, meta_path)

        metamodel = utils.PermInvModel(dims)
        metamodel.load_state_dict(ch.load(metamodel_path))
        metamodel = metamodel.cuda()

        losses_dict, _, preds = utils.get_ratio_info_for_reg_meta(
            metamodel, X_test, Y_test, 100, args.batch_size,
            combined=False)
        for k, v in losses_dict.items():
            k_ = float("%.2f" % float(k))
            if k_ not in mses:
                mses[k_] = []
            mses[k_].append(v)
        for k, v in preds.items():
            for pred in v:
                # data.append([float(k), pred])
                data.append([float("%.2f" % k), pred])

        losses_dict_og, _, preds_og = utils.get_ratio_info_for_reg_meta(
            metamodel, X_test_og, Y_test_og, 1000, args.batch_size,
            combined=False)
        for k, v in losses_dict_og.items():
            k_ = float("%.2f" % float(k))
            if k_ not in mses:
                mses[k_] = []
            mses[k_].append(v)
        for k, v in preds_og.items():
            for pred in v:
                # data.append([float(k), pred])
                data.append([float("%.2f" % k), pred])

    df = pd.DataFrame(data, columns=columns)
    # Seaborn Boxplot and save
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df,
        color="seagreen")

    # Plot dashed X=Y line for reference on same plot
    sns_plot = sns.lineplot(x=[0, 12], y=[0.2, 0.8], color="silver")
    sns_plot.lines[-1].set_linestyle("--")

    # Plot MSEs
    mses = {k: np.mean(v) for k, v in mses.items()}
    ax2 = plt.twinx()
    # -0.2 to account for the fact that ratios start with 0.2
    # 12 is (num x ticks) - 1
    # 5/3, since (0.8 - 0.2) = 0.6, which needs to be scaled to 1, so divide by 0.6
    sns.scatterplot(x=[5/3* 12 * (x - 0.2) for x in mses.keys()],
                    y=list(mses.values()), s=30, ax=ax2)
    ax2.set_ylabel(r"MSE")

    sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=60)
    sns_plot.figure.savefig("plots/regression_plot.pdf", bbox_inches='tight')
    # sns_plot.figure.savefig("plots/regression_plot.png", bbox_inches='tight')
