import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils
import numpy as np
from model_utils import BASE_MODELS_DIR
from data_utils import PROPERTY_FOCUS, SUPPORTED_PROPERTIES
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--legend', action="store_true",
                        help='Add legend to plots')
    parser.add_argument('--novtitle', action="store_true",
                        help='Remove Y-axis label')
    parser.add_argument('--dash', action="store_true",
                        help='Add dashed line midway?')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio', default=0.5,
                        help='test ratio')
    args = parser.parse_args()
    flash_utils(args)

    first_cat = " 0.5"

    # Set font size
    plt.rcParams.update({'font.size': 18})

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    data = []
    columns = [
        r'%s proportion of training data ($\alpha_1$)' % PROPERTY_FOCUS[args.filter],
        "Accuracy (%)"
    ]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % first_cat)
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % first_cat)

    targets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]
    targets = [str(x) for x in targets if x != args.ratio]

    if args.filter == "race":
        raw_data = [
           [91.4, 93.65, 92.35, 88.5, 91.65, 85.05, 91.65, 94.3, 94.85, 90.5]
,[64.65, 65.55, 70.3, 67.2, 66.35, 70.35, 74.15, 64.45, 70.85, 70.5]
,[56.35, 55.0, 57.55, 55.4, 54.1, 55.4, 56.2, 53.65, 53.0, 58.35]
,[50.7, 51.05, 51.35, 50.6, 50.5, 50.7, 50.7, 49.85, 51.85, 50.0]
,[49.7, 49.65, 52.95, 50.55, 51.75, 50.65, 49.55, 51.5, 50.15, 49.4]
,[50.9, 50.9, 47.7, 50.6, 51.5, 49.35, 51.6, 52.15, 50.8, 48.8]
, [55.15, 49.9, 53.05, 52.35, 52.05, 50.25, 51.7, 49.35, 53.25, 50.65]
, [52.8, 54.6, 54.15, 53.45, 53.95, 59.0, 53.5, 57.15, 58.8, 54.65]
, [66.35, 68.15, 72.45, 69.75, 65.0, 66.4, 62.85, 68.8, 67.35, 62.8]
, [100.0, 100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 100.0, 100.0, 100.0]
        ]
    else:
        raw_data = [
             [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
,[76.25, 76.8, 74.5, 63.9, 75.55, 76.1, 74.85, 73.5, 76.35, 80.15]
,[66.0, 56.2, 59.65, 55.8, 61.0, 59.9, 57.45, 56.95, 59.15, 58.9]
,[51.55, 51.15, 50.65, 52.75, 51.95, 50.9, 54.5, 54.15, 51.85, 53.5]
,[50.85, 51.95, 51.6, 49.4, 49.1, 49.7, 50.85, 51.65, 51.25, 50.4]
,[50.4, 50.1, 50.05, 51.0, 50.35, 48.6, 48.65, 50.2, 49.55, 51.1]
,[48.8, 49.9, 48.85, 50.6, 52.7, 50.15, 50.45, 49.7, 50.05, 49.85]
,[59.0, 60.85, 55.0, 57.9, 62.15, 59.65, 58.6, 56.45, 58.75, 56.75]
,[79.25, 78.95, 78.1, 79.6, 74.8, 75.75, 77.65, 73.5, 75.25, 79.55]
, [98.55, 99.25, 98.3, 98.9, 99.1, 98.45, 99.35, 98.1, 98.4, 98.95]
        ]

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([float(targets[i]), raw_data[i][j]])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    if args.dash:
        midpoint = (lower + upper) / 2
        plt.axvline(x=midpoint, color='white' if args.darkplot else 'black',
                    linewidth=1.0, linestyle='--')

    

    # Custom legend
    if args.legend:
        meta_patch = mpatches.Patch(color='C0', label=r'Meta-Classifier')
        baseline_patch = mpatches.Patch(color='C1', label=r'Loss Test')
        threshold_patch = mpatches.Patch(color='C2', label=r'Threshold Test')
        plt.legend(handles=[meta_patch, baseline_patch, threshold_patch])

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Make sure axis label not cut off
    plt.tight_layout()

    # Save plot
    sns_plot.figure.savefig("./meta_boxplot_%s.jpg" % args.filter)
