"""
    Draw scatter plots for given models to correlate inference risk with
    their accuracies or robustness (for robust models).
"""
import argparse
import json
from data_utils import SUPPORTED_PROPERTIES
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        default="Male",
                        help='name for subfolder to save/load data from')
    args = parser.parse_args()
    # Fetch relevant accuracies (all trials) for different methods
    # From results JSON file
    # Draw scatter plot for each experiment, with inference accurac
    # One one axis, and normal accuracy/robustness
    # On another exis
    # USe different colors/shapes for each eps trial

    # Load data
    with open("./log/meta/vary_n_boxplots.json", 'r') as f:
        raw_data = json.load(f)
        if args.filter not in raw_data:
            raise ValueError(
                "Requested data not found for filter %s" % args.filter)
        raw_data = raw_data[args.filter]
    
    # Pick a particular ratio index
    ratio_index = 0

    meta = {
        0: "All",
        8: "Robust (eps=8)",
        16: "Robust (eps=16)",
    }
    # Get meta-classifier accuracies
    meta_accs = {}
    for k, v in meta.items():
        meta_accs[k] = raw_data[v]["1600"][ratio_index]

    threshold = {
        0: "Threshold Test (eps=0)",
        8: "Threshold Test (eps=8)",
        16: "Threshold Test (eps=16)" 
    }
    # Get threshold-test accuracies
    threshold_accs = {}
    for k, v in threshold.items():
        threshold_accs[k] = raw_data[v]["100"][ratio_index]
    
    X, Y, colors = [], [], []
    for k, v in meta_accs.items():
        for acc in v:
            X.append(k)
            Y.append(acc)
            colors.append("C0")
    for k, v in threshold_accs.items():
        for acc in v:
            X.append(k)
            Y.append(acc)
            colors.append("C1")
    
    # Plot
    plt.scatter(X, Y, c=colors)
    plt.savefig("./plots/scatter.png")