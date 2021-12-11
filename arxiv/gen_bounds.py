from data_utils import get_zipf_params_for_degree, compute_bound, find_n_eff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 200


def get_arxiv_data():
    raw_data_meta = [
        [
            [51.5, 54.6, 52.7],
            [93.9, 76.8, 97.55],
            [100, 100, 99.8, 93.4, 94.55],
            [99.7, 99.5, 97.7],
            [99.35, 100, 99.7, 97.7],
            [97.65, 95.9, 100, 100, 100],
            [100, 100, 99.95, 100],
            [99.75, 100, 100, 100],
        ],
        [
            [56.4, 97.65, 97.55, 50.1],
            [91, 79.15, 98.15, 99.95, 92.55],
            [99.05, 94.4, 93.75],
            [91.75, 99.5, 96.1, 89.4, 100],
            [100, 100, 99.95, 100, 97.45],
            [100, 100, 97.55, 100, 100],
            [99.07, 100, 100, 99.85, 100]
        ],
        [
            [78.65, 99.1, 50.6],
            [93, 90.5, 93.75],
            [99.95, 100, 98.5, 99.95, 83.15],
            [100, 100, 92.45, 100],
            [99.7, 100, 99.7, 100],
            [99.6, 100, 98.29, 100]
        ],
        [
            [96.1, 84.1, 93.45],
            [100, 100, 100],
            [87.15, 100, 100, 100, 92.15],
            [99, 100, 100, 100, 100],
            [100, 100, 100, 100]
        ],
        [
            [87.2, 92.9, 90.5],
            [98.34, 89.45, 99.95, 99.95, 96.93, 100, 92.3, 100],
            [99.9, 99.9, 95.4],
            [100, 100, 99.9, 99.3, 99.5, 98.3, 100, 98.5, 100]
        ],
        [
            [99.9, 98.95, 100, 73.8, 99.6],
            [96.75, 100, 96, 97.15, 100],
            [100, 100, 100, 100]
        ],
        [
            [53.75, 58.4, 86.85, 60.15],
            [99.5, 74.21, 64.22, 76.92, 83.14]
        ],
        [
            [100, 70.8, 99.25, 96.04, 82.24]
        ]
    ]
    raw_data_threshold = [
        [
            [51.25, 52.25, 51.6],
            [52.8, 52.25, 54],
            [51.85, 53.55, 54.1],
            [51.1, 51.95, 50.2],
            [59.3, 56.05, 64.15],
            [67.05, 61.9, 66.5],
            [76.05, 72.4, 75.05],
            [55.04, 52.99, 53.39]
        ],
        [
            [49.85, 49.85, 49.85],
            [51.05, 50.35, 51.4],
            [50.4, 50.05, 52.95],
            [54.45, 54.75, 54.3],
            [58.35, 50.65, 56.85],
            [72.4, 65.9, 74],
            [57.75, 61.67, 53.79]
        ],
        [
            [51.85, 51.45, 55.65],
            [50.45, 50.5, 50.6],
            [54.65, 51.8, 72.1],
            [55.15, 56.2, 55.75],
            [67.05, 76.55, 64.45],
            [61.97, 60.41, 54.39]
        ],
        [
            [50.4, 50.1, 50.65],
            [52.45, 53.9, 51.65],
            [59.45, 59.8, 54.6],
            [67, 60.7, 67.55],
            [54.69, 62.42, 57.4]
        ],
        [
            [50.7, 50.95, 50.15],
            [50.8, 50.15, 50.3],
            [51.1, 51.95, 50.2],
            [53.79, 53.94, 51.43]
        ],
        [
            [57.2, 51.9, 54.7],
            [57.75, 60.8, 52.95],
            [57.85, 53.54, 60.46]
        ],
        [
            [52.4, 52.35, 52.3],
            [51.93, 51.98, 51.58]
        ],
        [
            [53.59, 53.49, 52.23]
        ]
    ]

    raw_data_loss = [
        [53.07, 54.33, 53.28, 50, 53.13, 50.07, 50, 50],
        [54.57, 58.68, 64.1, 60.5, 53.82, 50, 50],
        [54.38, 56.63, 66.55, 50.65, 50, 50],
        [53.9, 68.1, 50.38, 50, 50],
        [55.8, 50, 50, 50],
        [50, 50, 50],
        [50, 50],
        [50]
    ]

    degrees = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    return (raw_data_meta, raw_data_threshold, raw_data_loss, degrees)


def process_data(meta, threshold, raw, degrees, picked_degree):
    for i in range(len(meta)):
        for j in range(len(meta[i])):
            meta[i][j] = np.max(meta[i][j])

    for i in range(len(threshold)):
        for j in range(len(threshold[i])):
            threshold[i][j] = np.max(threshold[i][j])

    for i in range(len(raw)):
        for j in range(len(raw[i])):
            raw[i][j] = max(raw[i][j], max(meta[i][j], threshold[i][j]))

    collection = {}
    for i in range(len(degrees)):
        hmm = 0
        for j in range(i + 1, len(degrees)):
            d1, d2 = degrees[i], degrees[j]

            if picked_degree != d1 and picked_degree != d2:
                continue

            if picked_degree == d1:
                other_degree = d2
            else:
                other_degree = d1

            wanted_r = "%d" % picked_degree
            collection[wanted_r] = collection.get(
                wanted_r, []) + [(other_degree, raw[i][hmm] / 100)]
            hmm += 1

    return collection


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=13)

    # plt.style.use('dark_background')

    # Plot curves corresponding to ratios
    colors = ['deepskyblue']
    curve_colors = ['mediumpurple']
    n_effs = []

    # Mapping between distributions and their (n, s) values
    mapping = {
        9: (86, 1.843366),
        10: (133, 1.881514),
        11: (214, 1.929022),
        12: (257, 1.927125),
        13: (253, 1.918591),
        14: (267, 1.914586),
        15: (265, 1.903623),
        16: (263, 1.886148),
        17: (263, 1.876854)
    }

    # Get n_eff values for given accuracies
    count, sum_neff, max_neff = 0, 0, 0
    
    # picked_degrees = [9, 10, 11, 12, 13, 14, 15, 16]
    picked_degrees = [13]
    for picked_degree in picked_degrees:

        arxiv = process_data(*get_arxiv_data(), picked_degree)
        print(arxiv)
        exit(0)
        for (k, v) in arxiv['%d' % picked_degree]:
            n_eff = find_n_eff(mapping[picked_degree], mapping[k], v)
            if n_eff == np.inf:
                n_eff = "infinity"
            else:
                sum_neff += n_eff
                count += 1

                if n_eff > max_neff:
                    max_neff = n_eff

                # Check if computed n_eff makes sense
                bound_computed = compute_bound(mapping[k], mapping[picked_degree], n_eff)
                assert (np.round(bound_computed, 4) >= np.round(v, 4)), "Bound computation is off!"
                n_eff = "%.2f" % n_eff
            print("Degrees: (%d, %d) | Accuracy: %.2f | N-effective: %s" % (picked_degree, k, v, n_eff))
    print("Average N-effective: %.1f" % (sum_neff / count))
    print("Maximum N-effective: %.1f" % max_neff)

    # Plot individual points
    arxiv = process_data(*get_arxiv_data(), picked_degree)
    x_all, y_all = [], []
    for k, v in arxiv.items():
        if k != "%d" % picked_degree:
            continue
        x = [p[0] for p in v if p[1] < 1]
        y = [p[1] for p in v if p[1] < 1]
        x_all.append(x)
        y_all.append(y)
        # Skip the ones with 100% accuracy
        plt.scatter(x, y, color=colors[0], edgecolors='face', marker='o')

    # Draw curves
    granularity = 0.05
    x_axis = np.arange(9, 17 + granularity, granularity)
    # x_axis = [ 9., 11., 13., 15., 17.]
    # y_axis = [1.0, 0.9422685113466328, 0.5053644290872348, 0.9675056045831436, 0.9992632319843795]

    n_eff = 493
    # For each degree on x-axis, find corresponding (n, s) values and generate curve
    y_axis = []
    for deg in x_axis:
        m, N_max, _, _ = get_zipf_params_for_degree('adv', deg)
        # Get corresponding bound on accuracy:
        acc = compute_bound((N_max, m), mapping[picked_degree], n_eff)
        y_axis.append(acc)
    plt.plot(x_axis, y_axis, color=curve_colors[0], label=r"$n_{leaked}=%d$" % n_eff)

    print(x_axis)
    print(y_axis)

    # Trick to get desired legend
    plt.plot([], [], color=colors[0], marker="o",
             ms=10, ls="", label="ogbn-arxiv")

    plt.xlabel(r'Mean Node-Degree')
    plt.ylabel(r'Accuracy')
    plt.ylim(0.5, 1)
    plt.xticks(np.arange(9, 17 + 1, 1))
    # plt.grid()
    plt.style.use('seaborn')
    plt.legend()
    plt.savefig("./bound_curves_%d.png" % picked_degree)
