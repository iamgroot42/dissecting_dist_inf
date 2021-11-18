"""
    FInd parameters of Zipf's law that fit the given data
"""
from data_utils import ArxivNodeDataset
from collections import Counter
import numpy as np


def best_fit_no_intercept(x, y):
    num = np.sum(x * y)
    den = np.sum(x * x)
    return num / den


def run_for_deg(split, degree):
    ds = ArxivNodeDataset(split)

    # Change mean-node degree
    ds.change_mean_degree(degree)

    # Compute degrees
    ds.precompute_degrees()

    # Get degs
    degs = ds.degs

    # Get counts of degrees
    deg_counts = Counter(degs).items()

    # Filter out 0 frequency nodes
    deg_counts = [dc for dc in deg_counts if dc[1] > 0]

    # Get sorting order of degrees according to frequency
    deg_counts.sort(key=lambda x: x[1], reverse=True)

    # Extract frequencies (to fit Zipf's law)
    freqs = np.array([dc[1] for dc in deg_counts])

    # Filter out extremely-low-frequency nodes
    # freqs = freqs[freqs > 1]

    # Normalize frequencies
    x_range = np.arange(1, len(freqs)+1)
    freqs = freqs / np.sum(freqs)

    # Convert to log-log scale to find parameters
    log_freqs = np.log(freqs)
    log_x_range = np.log(x_range)

    # Get slope for log-log line
    m_log = best_fit_no_intercept(log_x_range, log_freqs)

    # Get corresponding slope for linear scale
    m = np.exp(m_log)

    # Print out relevant parameters
    N_max = len(freqs)

    return m, N_max


def compute_kl(d0, d1):
    # Determine which of two Ns are larger:
    if d0[0] >= d1[0]:
        return compute_kl(d1, d0)

    n0, m0 = d0
    n1, m1 = d1

    c0 = np.sum(np.arange(1, n0 + 1) ** -m0)
    c1 = np.sum(np.arange(1, n1 + 1) ** -m1)

    if m1 > m0:
        return np.log(c1 / c0) + (m1 - m0) * np.log(n0)
    else:
        return np.log(c1 / c0)


def compute_kl_actual(d0, d1):
    if d0[0] >= d1[0]:
        return compute_kl_actual(d1, d0)
    
    n0, m0 = d0
    n1, m1 = d1

    def sum_0(x): return np.sum(np.arange(1, x + 1) ** -m0)
    def sum_1(x): return np.sum(np.arange(1, x + 1) ** -m1)
    exes = np.arange(1, n0 + 1)
    logsum0 = np.sum((exes ** -m0) * np.log(exes))

    c0 = sum_0(n0)
    c1 = sum_1(n1)

    return np.log(c1 / c0) + ((m1 - m0) / c0) * logsum0


def get_n_bound(n, kl0, kl1):
    first_term = np.sqrt(1 - np.exp(-n * kl0))
    second_term = np.sqrt(1 - np.exp(-n * kl1))
    return min(first_term, second_term)


if __name__ == "__main__":
    split = "victim"
    degree = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    # for d in degree:
        # m, N_max = run_for_deg(split, d)
        # print("Degree %d | N was %d, s was %.6f" % (d, N_max, m))

    # Without-filtering
    mapping = {
        9: (87, 0.159476),
        10: (134, 0.153103),
        11: (226, 0.143644),
        12: (371, 0.138641),
        13: (504, 0.139911),
        14: (541, 0.141643),
        15: (552, 0.142441),
        16: (551, 0.144336),
        17: (552, 0.146009),
    }

    def harmonic_n_s(x, y):
        return np.sum(np.arange(1, x + 1) ** -y)

    # TODO: See how good the fit actually are
    # Currently FAR off- filter out more data from distribution
    for k, v in mapping.items():
        n, s = v
        # Compute mean degree for this distribution
        computed_mean = harmonic_n_s(n, s)
        print(k, computed_mean)

    # TODO: See what KL they correspond to
    first, second = 14, 13
    kl0 = compute_kl(mapping[first], mapping[second])
    kl1 = compute_kl(mapping[second], mapping[first])

    # kl0 = compute_kl_actual(mapping[first], mapping[second])
    # kl1 = compute_kl_actual(mapping[second], mapping[first])

    n_points = 20
    print("Upper bound on accuracy: %.2f" %
          (100 * get_n_bound(n_points, kl0, kl1)))
