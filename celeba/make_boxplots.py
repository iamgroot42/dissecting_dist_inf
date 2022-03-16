import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import flash_utils, get_n_effective
import numpy as np
from data_utils import SUPPORTED_PROPERTIES
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
    parser.add_argument('--multimode', action="store_true",
                        help='Plots for meta-classifier methods')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        default="Male",
                        help='name for subfolder to save/load data from')
    parser.add_argument('--dash', action="store_true",
                        help='Add dashed line midway?')
    args = parser.parse_args()
    flash_utils(args)

    first_cat = "0.5"

    # Set font size
    if args.multimode:
        plt.rcParams.update({'font.size': 16})
    else:
        plt.rcParams.update({'font.size': 18})

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    # Changed male ratios to female ratios by flipping data
    # Be mindful of this change when modifying/inserting data

    data = []
    columns = [
        r'Female proportion of training data ($\alpha_1$)',
        "Accuracy (%)",
        "Feature-extraction method"
    ]

    targets = ["0.0", "0.1", "0.2", "0.3",
               "0.4", "0.6", "0.7", "0.8", "0.9", "1.0"]

    if args.filter == "Young":
        columns[0] = r'Old proportion of training data ($\alpha_1$)'
        fc_perf = [
            [78.85, 50.1, 79.8, 79.2, 75.15], #Rerun
            [70.9, 68.15, 70.1, 70.15, 69.05], #Rerun
            [60.05, 63.6, 63.55, 63.55, 63.8], #Rerun
            [54.15, 52.75, 53.0, 50.9, 56.7], #Rerun
            [48.7, 50.7, 49.35, 51.1, 49.4], #Rerun
            [49.45, 49.35, 49.55, 50.6, 51.8], #Rerun
            [50.15, 52.1, 55.4, 54.75, 49.5], #Rerun
            [61.65, 49.95, 54.55, 50.75, 61.15], #Rerun
            [66.1, 66.3, 64.45, 62.55, 67.0], #Rerun
            [73.85, 50.85, 74.95, 76.4, 74.05] #Rerun
        ]

        conv_perf = [
            [63.65, 53.6, 53.0, 56.45, 53.95], #Rerun
            [51.6, 52.65, 52.2, 54.45, 52.75], #Rerun
            [53.5, 50.15, 50.25, 50.4, 50.35], #Rerun
            [51.05, 50.6, 49.95, 51.05, 48.25], #Rerun
            [52.05, 50.8, 48.95, 50.25, 49.8], #Rerun
            [49.55, 51.75, 49.8, 48.75, 51.2], #Rerun
            [52.0, 55.45, 52.15, 53.8, 50.8], #Rerun
            [57.35, 54.4, 51.25, 59.6, 56.0], #Rerun
            [64.1, 54.9, 65.45, 67.35, 57.8], #Rerun
            [74.7, 77.7, 79.7, 81.1, 76.65] #Rerun
        ]
        combined_perf = [
            [60.45, 57.15, 64.4, 63.75, 63.1], #Rerun
            [51.75, 51.95, 51.95, 53.65, 52.3], #Rerun
            [51.4, 50.8, 49.8, 50.6, 51.0], #Rerun
            [50.75, 51.05, 49.0, 50.45, 49.2], #Rerun
            [49.25, 50.1, 49.75, 51.75, 49.55], #Rerun
            [52.5, 49.7, 50.25, 50.05, 51.45], #Rerun
            [53.15, 56.5, 50.0, 52.3, 53.55], #Rerun
            [59.05, 56.4, 56.75, 54.6, 56.7], #Rerun
            [56.4, 65.6, 55.8, 53.4, 56.5], #Rerun
            [58.2, 81.2, 81.7, 81.15, 74.25] #Rerun
        ]

        thresholds = [
            [50.27, 50.28, 50.28], # 0.0
            [49.95, 50, 49.95],  # 0.1
            [51.95, 48, 52.06],  # 0.2
            [49.93, 49.88, 49.78],  # 0.3
            [51.4, 50.05, 53.35],  # 0.4
            [51.15, 48.65, 51.87],  # 0.6
            [50.90, 50.7, 52.06],  # 0.7
            [55.34, 51.17, 51.78],  # 0.8
            [51.91, 51.96, 50.60],  # 0.9
            [53.35, 59.55, 54.2], # 1.0
        ]

        baselines = [57.7, 55.97, 59.95, 51.55, 52.08, 47.83, 50.13, 71.48, 62.82, 86.9]

        if args.multimode:
            combined_perf = combined_perf[::-1]
            for i in range(len(combined_perf)):
                for j in range(len(combined_perf[i])):
                    data.append([float(targets[i]), combined_perf[i][j],
                                 "Full-Model"])

            conv_perf = conv_perf[::-1]
            for i in range(len(conv_perf)):
                for j in range(len(conv_perf[i])):
                    data.append(
                        [float(targets[i]), conv_perf[i][j], "Only-Conv"])

        fc_perf = fc_perf[::-1]
        for i in range(len(fc_perf)):
            for j in range(len(fc_perf[i])):
                data.append([float(targets[i]), fc_perf[i][j],
                             "Only-FC" if args.multimode else "Meta-Classifier"])

    elif args.filter == "Male":
        fc_perf = [
            [76.65, 75.35, 74.7, 49.9, 70.95], #Rerun
            [65.1, 63.5, 67.75, 62.05, 60.15], #Rerun
            [59.85, 56.3, 57.0, 55.45, 57.45], #Rerun
            [50.0, 54.45, 49.8, 51.55, 55.9], #Rerun
            [55.1, 52.35, 51.25, 54.7, 54.75], #Rerun
            [50.1, 51.55, 52.65, 50.15, 51.8], #Rerun
            [53.7, 53.35, 54.95, 50.8, 56.4], #Rerun
            [51.3, 50.1, 50.05, 50.05, 50.05], #Rerun
            [55.4, 54.4, 54.85, 55.4, 52.35], #Rerun
            [53.9, 55.0, 55.7, 55.3, 55.35] #Rerun
        ]
        conv_perf = [
            [83.2, 85.5, 88.9, 82.1, 80.9], #Rerun
            [78.4, 78.55, 76.9, 83.35, 76.5], #Rerun
            [66.4, 69.5, 61.85, 62.6, 67.95], #Rerun
            [57.85, 63.5, 56.65, 57.85, 59.15], #Rerun
            [51.05, 53.95, 53.5, 54.95, 53.65], #Rerun
            [55.55, 55.2, 53.8, 53.35, 52.1], #Rerun
            [60.15, 61.55, 60.0, 62.4, 62.55], # Rerun
            [50.9, 50.25, 51.25, 53.35, 55.15], #Rerun
            [63.4, 61.35, 66.6, 72.2, 69.15], #Rerun
            [60.75, 57.7, 62.5, 61.0, 55.0] #Rerun
        ]
        combined_perf = [
            [93.2, 89.65, 93.25, 92.0, 91.65], #Rerun
            [68.55, 81.35, 77.5, 74.85, 80.65], #Rerun
            [67.5, 61.7, 65.9, 59.5, 59.3], #Rerun
            [60.95, 57.8, 60.45, 60.0, 56.3], #Rerun
            [53.0, 52.95, 54.0, 53.9, 51.55], #Rerun
            [52.5, 53.35, 54.75, 53.35, 51.5], #Rerun
            [59.7, 55.5, 61.35, 60.8, 62.2], #Rerun
            [51.85, 62.9, 52.1, 60.25, 51.85], #Rerun
            [70.4, 74.2, 73.1, 70.5, 69.1], #Rerun
            [67.45, 66.2, 60.1, 69.05, 65.4] #Rerun
        ]
        # Baseline method results
        baselines = [50.25, 50.15, 50.36, 50.0, 50, 50, 51.0, 51.5, 50.14, 49.9]
        thresholds = [
            [58.204, 57.25, 62.47],
            [56.32, 53.81, 52.91],
            [54.56, 55.12, 50.93],
            [53.46, 52.16, 52.44],
            [52.71, 52.51, 52.09],
            [49.7, 54.01, 51.25],
            [59.38, 60.48,  60.18],
            [54.48, 55.09, 50.0],
            [68.41, 66.8, 67.63],
            [55.07, 61.75, 62.05]
        ]

        # Threshol/Loss tests for adv-16 models
        baselines_both_robust = [50, 50, 50, 50, 50, 52, 50, 50, 50, 54]
        thresholds_both_robust = [
            [83, 84, 85.15],
            [84, 74, 80.97],
            [60, 68, 74.18],
            [67, 64, 64.92],
            [54, 58, 61.89],
            [54, 55, 62.61],
            [64, 64, 61.42],
            [64, 59, 67.54],
            [73, 71, 74.39],
            [51, 55, 57.82],
        ]

        # Threshol/Loss tests for adv-8 models
        baselines_both_robust = [50.35, 50.65, 53.35, 52.35, 50.2, 49.75, 55.15, 58.2, 67.25, 72.7]
        thresholds_both_robust = [
            [79.7, 71.9, 75.15],
            [66.1, 70.6, 64.2],
            [59.9, 59.2, 57.55],
            [57.5, 53.7, 54.05],
            [52.25, 49.7, 51],
            [62.55, 64.8, 63.75],
            [62.7, 67.5, 59],
            [65.1, 73.65, 70.15],
            [84.7, 81.8, 75.25],
            [79.6, 91.65, 76.3],
        ]

        # Ratios we want to plot are opposite of what we compute
        combined_perf = combined_perf[::-1]
        for i in range(len(combined_perf)):
            for j in range(len(combined_perf[i])):
                data.append([float(targets[i]), combined_perf[i][j],
                            "Full-Model" if args.multimode else "Meta-Classifier"])

        if args.multimode:
            conv_perf = conv_perf[::-1]
            for i in range(len(conv_perf)):
                for j in range(len(conv_perf[i])):
                    data.append([float(targets[i]), conv_perf[i][j], "Only-Conv"])

            fc_perf = fc_perf[::-1]
            for i in range(len(fc_perf)):
                for j in range(len(fc_perf[i])):
                    data.append([float(targets[i]), fc_perf[i][j], "Only-FC"])

    else:
        raise ValueError("Requested data not available")

    df = pd.DataFrame(data, columns=columns)
    if args.multimode:
        sns_plot = sns.boxplot(
            x=columns[0], y=columns[1], data=df,
            hue=columns[2], showfliers=False,)
    else:
        sns_plot = sns.boxplot(
            x=columns[0], y=columns[1], data=df,
            color='C0', showfliers=False)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add legend if requested
    if not args.legend and args.multimode:
        sns_plot.get_legend().remove()

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    if args.dash:
        midpoint = (lower + upper) / 2
        plt.axvline(x=midpoint,
                    color='white' if args.darkplot else 'black',
                    linewidth=1.0, linestyle='--')

    # This data was for 1-ratio, so flip before plotting
    baselines = baselines[::-1]
    thresholds = thresholds[::-1]

    if not args.multimode:
        # Plot baselines
        targets_scaled = range(int((upper - lower)))
        plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

        # Plot numbers for threshold-based accuracy
        means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
        plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

    fc_perf = np.mean(fc_perf, 1)
    conv_perf = np.mean(conv_perf, 1)
    combined_perf = np.mean(combined_perf, 1)
    thresholds = np.mean(thresholds, 1)
    ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    fc_n_eff = [get_n_effective(fc_perf[i] / 100, 0.5, ratios[i]) for i in range(len(ratios))]
    conv_n_eff = [get_n_effective(conv_perf[i] / 100, 0.5, ratios[i]) for i in range(len(ratios))]
    combined_n_eff = [get_n_effective(combined_perf[i] / 100, 0.5, ratios[i]) for i in range(len(ratios))]
    threshold_n_eff = [get_n_effective(thresholds[i] / 100, 0.5, ratios[i]) for i in range(len(ratios))]
    baselines_n_eff = [get_n_effective(baselines[i] / 100, 0.5, ratios[i]) for i in range(len(ratios))]
    print("FC Meta:",np.mean(fc_n_eff))
    print("Conv Meta:",np.mean(conv_n_eff))
    print("Full Meta:",np.mean(combined_n_eff))
    print("Threshold Test n_eff:",np.mean(threshold_n_eff))
    print("Loss Test n_eff:",np.mean(baselines_n_eff))

    # Custom legend
    if args.legend and not args.multimode:
        meta_patch = mpatches.Patch(color='C0', label='Meta-Classifier')
        baseline_patch = mpatches.Patch(color='C1', label='Loss Test')
        threshold_patch = mpatches.Patch(color='C2', label='Threshold Test')
        plt.legend(handles=[meta_patch, baseline_patch, threshold_patch])

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Make sure axis label not cut off
    plt.tight_layout()

    # Save plot
    suffix = "_multi" if args.multimode else ""
    sns_plot.figure.savefig("./plots/celeba_meta_boxplot_%s%s.pdf" %
                            (args.filter, suffix))
