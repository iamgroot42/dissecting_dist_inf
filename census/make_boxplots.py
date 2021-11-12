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
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio', default = 0.5,
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
        r'%s proportion of training data ($\alpha$)' % PROPERTY_FOCUS[args.filter],
        "Accuracy (%)"
    ]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % first_cat)
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % first_cat)

    targets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1.0]
    targets = [str(x) for x in targets if x != args.ratio]

    if args.filter != "race":
        with open("meta.log","r") as lg:
            lines = lg.readlines()
            lines = [line.strip() for line in lines]
            raw_data=[]
            for line in lines:
                raw_data.append([float(i) for i in line.split(",")])
                
        
        
        
    else:
        raw_data = [
            [100.0, 99.95, 100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 100.0, 100.0],
            [56.25, 53.75, 56.6, 56.3, 55.55, 57.55, 55.8, 55.95, 56.3, 58.7],
            [55.1, 53.8, 54.65, 53.0, 53.4, 53.7, 53.65, 54.65, 54.7, 52.2],
            [54.45 , 53.1 , 52.85 , 52.35 , 51.75 , 53.05 , 51.85 , 53.25 , 52.25 , 51.8],
            [50.25 , 51.6 , 49.2 , 49.6 , 50.05 , 49.7 , 49.35 , 50.2 , 50.4 , 50.85],
            [51.15, 49.3, 51.15, 48.7, 49.7, 49.55, 50.35, 50.3, 50.25, 51.1],
            [51.95, 51.8, 49.45, 49.9, 50.9, 51.7, 53.9, 49.45, 49.55, 49.25],
            [54.95, 52.4, 56.3, 49.9, 51.1, 56.7, 54.9, 50.4, 54.3, 49.5],
            [51.26, 51.21, 51.56, 53.39, 51.41, 54.13, 49.73, 51.81, 50.07, 53.19],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
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
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='white' if args.darkplot else 'black',
                linewidth=1.0, linestyle='--')

    # Map range to numbers to be plotted
    if args.filter == 'race':
        # For race
        baselines = [57, 50, 50, 50, 50, 50, 50.35, 50.05, 50.05, 50.05]
        thresholds = [
            [80.10, 79.70, 72.90, 78.65, 78.20, 72.10, 78.50, 77.45, 75.25, 79.90],
            [69.15, 70.70, 63.50, 63.75, 63.30, 56.80, 62.95, 72.50, 68.95, 68.25],
            [66.15, 62.00, 64.85, 60.95, 61.80, 66.10, 63.65, 64.20, 63.35, 60.85],
            [60.85, 61.00, 58.55, 67.60, 60.20, 52.85, 62.50, 65.05, 64.25, 57.75],
            [52.45, 60.75, 57.75, 58.20, 57.95, 57.85, 58.70, 58.60, 57.40, 59.60],
            [53.70, 54.55, 56.25, 57.20, 58.30, 55.15, 56.70, 53.65, 55.20, 54.65],
            [58.70, 63.05, 62.85, 60.30, 65.05, 63.70, 62.00, 62.25, 65.95, 65.80],
            [65.70, 58.70, 63.80, 62.20, 53.80, 59.30, 60.70, 56.85, 55.85, 58.75],
            [55.15, 51.35, 50.70, 54.40, 54.75, 51.15, 51.65, 52.55, 51.05, 50.35],
            [50.95, 59.00, 58.00, 59.00, 56.85, 59.00, 56.30, 59.70, 59.00, 59.70]
        ]
        # This data was for 1-ratio, so flip before plotting
        baselines = baselines[::-1]
        thresholds = thresholds[::-1]
    else:
        # For sex
        baselines = []
        thresholds = []
        for t in targets:
            with open(os.path.join(BASE_MODELS_DIR,"baseline_result"+str(args.ratio),args.filter+str(t)),"r") as r_b:
                line = r_b.readline()
                re = line.split("; ")
                [_, b_l] = re[0].split(": ")
                baselines.append(float(b_l))
                [_,t_l] = re[1].split(":")
                t_s = t_l.split(",")
                thresholds.append([float(x) for x in t_s])
                
        

    # Plot baselines
    targets_scaled = range(int((upper - lower)))
    plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    # Plot numbers for threshold-based accuracy
    means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
    plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

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
    sns_plot.figure.savefig("./meta_boxplot_%s.png" % args.filter)
