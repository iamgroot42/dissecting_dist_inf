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
        "Accuracy (%)",
        "description"
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
             [50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0]
,[50.55,50.25,49.95,50.85,50.3,50.2,50.35,52.3,50.1,50.0]
,[50.05,50.0,50.0,52.55,50.8,50.25,51.35,53.0,50.05,50.45]
,[49.95,50.0,50.05,50.05,50.0,49.75,50.15,50.05,50.4,50.0]
,[49.7,50.05,49.45,49.0,50.0,50.0,49.95,50.0,49.75,49.65]
,[49.85,50.35,48.55,49.9,49.95,48.9,49.45,49.0,50.0,48.15]
,[50.0,50.0,50.0,50.0,50.0,50.0,49.95,50.0,49.95,50.0]
,[50.1,51.05,50.0,52.0,51.25,50.1,50.0,51.55,49.85,51.2]
,[50.0,50.7,50.6,50.05,50.0,50.05,49.95,50.1,50.0,50.0]
, [50.05,50.4,50.05,50.05,50.35,55.15,50.15,50.3,50.05,50.15]
        ]

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([float(targets[i]), raw_data[i][j],"victim dp only"])

    if args.filter == "race":
        raw_data = [
           [62.55, 61.3, 61.05, 66.1, 59.3, 60.65, 60.65, 59.25, 55.0, 59.25]
,[56.75, 54.3, 53.65, 58.95, 57.9, 56.45, 53.0, 54.75, 52.5, 53.65]
,[53.15, 49.9, 49.95, 52.25, 50.35, 49.75, 50.4, 51.35, 49.95, 51.55]
,[49.6, 49.1, 50.3, 48.65, 50.45, 48.85, 50.85, 51.6, 50.45, 50.25]
,[50.7, 49.55, 50.25, 48.7, 49.5, 50.9, 49.5, 49.65, 49.75, 49.8]
,[50.5, 49.4, 51.55, 51.3, 52.2, 49.6, 51.85, 52.6, 52.65, 49.85]
,[50.55, 50.2, 50.35, 49.3, 49.7, 51.5, 51.5, 50.95, 50.85, 49.95]
,[53.45, 52.0, 51.2, 55.0, 52.8, 53.75, 52.65, 60.5, 55.1, 53.4]
,[64.4, 56.45, 58.85, 53.55, 61.85, 56.25, 63.5, 65.7, 58.9, 61.95]
,[73.5, 68.15, 72.4, 72.45, 72.8, 71.8, 73.3, 79.35, 71.05, 72.95]
        ]
    else:
        raw_data = [
              [68.8, 71.4, 56.75, 59.4, 66.0, 73.45, 72.95, 71.25, 65.15, 52.2]
,[53.6, 57.2, 50.5, 56.6, 50.45, 57.5, 56.45, 57.65, 54.75, 52.85]
,[52.4, 51.15, 50.5, 52.25, 52.5, 51.7, 53.2, 50.7, 55.35, 52.95]
,[50.05, 50.65, 52.15, 50.9, 50.1, 51.55, 51.45, 51.45, 50.5, 49.65]
,[49.15, 50.55, 49.8, 50.25, 49.55, 50.3, 49.85, 51.9, 48.55, 51.0]
,[49.55, 49.0, 50.5, 48.85, 49.5, 48.4, 49.25, 49.45, 49.6, 50.4]
,[51.1, 51.5, 51.65, 49.95, 50.65, 50.45, 50.4, 51.15, 50.45, 51.4]
,[51.45, 51.9, 52.3, 50.85, 51.0, 51.85, 50.1, 51.85, 52.7, 53.05]
,[50.7, 58.1, 56.85, 53.6, 54.8, 57.95, 60.8, 57.2, 53.3, 50.95]
,[69.4, 69.75, 76.9, 75.95, 64.05, 73.7, 67.2, 56.5, 76.35, 74.65]
        ]

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([float(targets[i]), raw_data[i][j],"both dp"])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], hue=columns[2], data=df,  showfliers=False,)

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
    sns_plot.figure.savefig("./meta_boxplot_dp_%s.jpg" % args.filter)
