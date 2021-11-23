"""
    FInd parameters of Zipf's law that fit the given data
"""
from data_utils import get_zipf_params_for_degree, compute_bound
import numpy as np


if __name__ == "__main__":
    split = "victim"
    degree = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    # order = None
    # for i, d in enumerate(degree):
        # m, N_max, order, calc_deg = get_zipf_params_for_degree(split, d, order)
        # m, N_max, _, calc_deg = get_zipf_params_for_degree(split, d, order)
        # print("%d (%.2f): (%d, %.6f)" % (d, calc_deg, N_max, m))

    # Without-filtering, sorted by freqs
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

    first, second = 13, 9
    n_points = 100
    bound = compute_bound(mapping[first], mapping[second], n_points)
    
    print("Upper bound on accuracy: %.2f" % (100 * bound))
