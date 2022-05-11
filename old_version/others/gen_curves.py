import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def bound_1(x, y, n):
    # Handle 0/0 form gracefully
    # if x == 0 and y == 0:
    #     return 0
    ratio = min(x, y) / max(x, y)
    return np.sqrt(1 -(ratio ** n))


def bound_2(x, y, n):
    ratio = (1 - max(x, y)) / (1 - min(x, y))
    return np.sqrt(1 - (ratio ** n))


def bound(x, y, n):
    l1 = bound_1(x, y, n)
    l2 = bound_2(x, y, n)
    pick = min(l1, l2) / 2
    return 0.5 + pick


def get_census_sex():
    raw_data_meta = [
        [
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [99.95, 99.95, 99.95, 99.95, 100, 99.95, 100, 100, 100, 99.95],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 99.5],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [99.95, 99.95, 99.95, 100, 100, 99.95, 99.95, 99.95, 97.85, 99.95]
        ],
        [
            [51.55, 52.55, 51.4, 54.1, 51.9, 52.05, 52.55, 51.55, 52.3, 52.5],
            [61.45, 60.4, 59.1, 58.9, 57.35, 60.9, 59.55, 58.5, 58.85, 59.4],
            [68.0, 64.25, 65.85, 66.8, 67.55, 65.4, 67.1, 66.15, 65.0, 66.0],
            [75.1, 78.23, 71.83, 68.567, 75.73,
             75.2, 74.767, 74.467, 67.3, 71.03],
            [77.7, 77.45, 76.15, 73.9, 76.15, 76.65, 77.05, 75.3, 75.55, 78.15],
            [80.85, 78.95, 80.65, 79.65, 79.3,
             82.75, 80.75, 67.55, 77.5, 80.4],
            [80.3, 81.75, 75.3, 71.65, 81.0, 78.9, 81.1, 79.5, 83.15, 84.1],
            [78.6, 86.05, 81.15, 82.9, 84.4, 82.3, 87.85, 84.8, 84.75, 85.1],
            [72.05, 72.8, 73.95, 75.7, 70.05, 74.25, 70.3, 65.95, 83.9, 85.2]
        ],
        [
            [52.5, 51.65, 51.2, 50.85, 52.0, 48.7, 52.3, 50.75, 50.05, 49.8],
            [55.7, 57.1, 56.3, 55.9, 57.9, 55.85, 57.3, 58.3, 57.6, 53.9],
            [63.836, 56.7676, 63.2676, 57.567, 50.6764,
             59.234, 60.367, 57.13, 56.53, 54.367],
            [63.1, 64.9, 63.5, 64.4, 66.0, 66.25, 65.9, 64.45, 61.6, 65.0],
            [62.35, 64.05, 69.45, 72.95, 65.75,
             61.55, 66.25, 61.8, 69.4, 66.55],
            [60.4, 64.3, 70.55, 62.55, 66.1, 61.6, 68.45, 66.6, 64.95, 68.95],
            [72.7, 64.35, 72.2, 73.65, 68.3, 68.95, 68.05, 73.9, 66.8, 71.6],
            [100.0, 98.6, 66.65, 78.05, 80.65,
             75.6, 69.95, 100.0, 69.95, 100.0],
        ],
        [
            [51.3, 50.8, 50.75, 49.75, 49.1, 49.95, 50.8, 49.05, 51.15, 49.3],
            [55.067, 52.734, 56.13, 52.5, 50.2, 55.734,
             52.7676, 51.03, 52.867, 50.367],
            [58.05, 57.25, 59.2, 59.5, 60.35, 58.35, 60.1, 59.85, 58.6, 59.2],
            [55.45, 55.8, 58.1, 61.75, 63.45, 56.1, 62.75, 65.4, 61.85, 64.75],
            [53.05, 59.45, 66.05, 61.1, 64.5, 57.8, 59.4, 54.05, 57.8, 61.45],
            [64.65, 65.4, 57.1, 70.4, 60.9, 67.75, 65.35, 63.65, 64.95, 64.2],
            [81.4, 69.15, 100.0, 100.0, 69.65,
             74.55, 71.25, 77.9, 67.95, 80.85],
        ],
        [
            [47.3, 50.867, 46.1674, 51.2676, 47.13,
             49.6, 49.336, 51.467, 47.1, 53.367],
            [51.45, 53.25, 52.95, 53.75, 51.95,
             54.3, 54.85, 54.1, 56.05, 53.1],
            [55.75, 56.55, 55.1, 53.95, 51.9, 56.3, 58.15, 53.3, 57.65, 53.45],
            [55.25, 53.95, 54.5, 53.6, 55.05, 53.75, 56.2, 53.7, 56.75, 58.0],
            [61.1, 66.15, 62.75, 58.95, 64.9, 63.65, 60.35, 61.55, 63.2, 58.45],
            [72.4, 67.5, 74.95, 67.5, 67.45, 100.0, 99.95, 100.0, 72.35, 71.65]
        ],
        [
            [52.734, 55.234, 53.53, 52.1, 59.336,
             55.2676, 58.0, 57.7, 61.93, 51.43],
            [64.667, 62.567, 61.63, 62.6, 63.7676,
             63.93, 64.267, 58.43, 63.7, 63.067],
            [67.73, 68.067, 67.3, 66.9, 66.83,
             67.067, 67.367, 67.8, 67.7, 68.2],
            [71.467, 72.1, 70.53, 70.23, 71.0,
             70.867, 70.367, 70.934, 70.3, 70.8],
            [82.067, 99.73, 99.0, 81.267, 80.1,
             77.867, 78.634, 77.73, 78.8, 82.367]
        ],
        [
            [49.9, 51.1, 50.95, 50.45, 50.25, 50.75, 49.45, 50.3, 51.3, 52.35],
            [52.6, 53.3, 51.6, 53.5, 50.35, 51.05, 54.5, 53.0, 52.25, 52.0],
            [56.2, 55.25, 55.9, 59.95, 55.15, 56.9, 55.65, 55.8, 58.8, 58.1],
            [72.25, 71.4, 69.5, 100.0, 83.6, 100.0, 72.65, 68.75, 66.05, 66.9],
        ],
        [
            [49.65, 52.2, 51.95, 51.6, 50.75, 52.75, 52.65, 52.2, 51.6, 51.1],
            [60.7, 61.35, 59.3, 61.05, 57.05, 58.75, 60.05, 59.0, 56.75, 59.95],
            [68.3, 73.9, 71.95, 66.4, 99.95, 84.1, 72.55, 73.0, 83.8, 66.5]
        ],
        [
            [56.5, 56.35, 55.05, 55.25, 56.7, 55.45, 55.2, 56.3, 55.0, 54.6],
            [87.0, 66.45, 72.5, 79.4, 70.4, 71.95, 100.0, 65.95, 76.3, 72.65]
        ],
        [
            [100.0, 72.9, 73.85, 71.75, 73.2, 72.15, 72.4, 75.95, 66.05, 100.0]
        ]
    ]
    raw_data_threshold = [
        [
            [78.5, 74.8, 74.35, 78.75, 76.8],
            [89.2, 88.75, 88.5, 85.75, 88.6],
            [91.9, 92.65, 91.7, 90.4, 92.65],
            [91.2, 90.35, 92, 85.1, 90.65],
            [64.05, 61.60, 63.45, 63.60, 59.00,
             61.65, 64.35, 63.35, 62.25, 63.55],
            [84.8, 82, 84, 87.05, 85.5],
            [90.85, 89.5, 91.35, 90.3, 91.4],
            [94.1, 92.9, 93.35, 93.85, 93.85],
            [94.5, 93.3, 93.05, 94.55, 94.7],
            [98.05, 98.05, 94.6, 98.1, 98.1]
        ],
        [
            [55.6, 54.65, 53.55, 54.2, 54.2],
            [50.8, 56.85, 56.5, 52.15, 57.1],
            [50.95, 54.2, 53.55, 55.75, 53.95],
            [55.35, 60.35, 65.00, 59.65, 61.20,
             54.35, 60.90, 56.70, 60.55, 57.60],
            [83.35, 81, 82.2, 80.9, 81.75],
            [92.5, 92.65, 93.3, 91, 92.55],
            [95.75, 95, 94.9, 94.65, 94.5],
            [95.65, 95.75, 94.15, 96.3, 96.15],
            [58.3, 58.65, 98.35, 56.65, 60.3]
        ],
        [
            [50.85, 50.85, 50.9, 50, 52.9],
            [59.3, 55.5, 58.2, 56.4, 57.65],
            [65.45, 62.85, 55.05, 61.35, 59.15,
             60.00, 59.40, 66.25, 60.90, 61.40],
            [81.85, 83.65, 81.95, 84.55, 83.85],
            [93.25, 93.65, 93.05, 93.05, 80.06],
            [95.75, 95.45, 95.9, 96.05, 95.2],
            [96.6, 95.75, 71.7, 96.05, 95.85],
            [98.85, 62.75, 64.55, 99.2, 98.85],
        ],
        [
            [55.4, 57.2, 56.05, 55.35, 57.7],
            [60.95, 67.40, 61.25, 64.10, 66.60,
             58.00, 62.40, 64.60, 62.60, 66.00],
            [82.5, 82.1, 81.6, 82, 83.5],
            [92.75, 92.75, 91.65, 91.45, 92.75],
            [92.2, 94.85, 94.75, 95.35, 94],
            [95.25, 95.2, 94.9, 94.65, 95.45],
            [98.45, 66.1, 97.45, 98.7, 68.15]
        ],
        [
            [61.85, 59.10, 54.95, 56.65, 60.55,
             61.95, 60.80, 61.20, 61.75, 59.90],
            [73.65, 72.35, 76.2, 73.35, 76.1],
            [89.1, 87.5, 88, 88.7, 88.1],
            [89.65, 92.25, 90.4, 90.85, 92.05],
            [90.85, 91.6, 92.4, 91.15, 19.95],
            [95.65, 91.95, 90.8, 89.9, 91.95]
        ],
        [
            [64.05, 63.65, 63.90, 62.35, 64.20,
             62.90, 64.15, 61.50, 62.85, 63.15],
            [68.35, 76.40, 69.65, 74.05, 76.30,
             76.20, 74.85, 75.30, 75.80, 75.90],
            [76.65, 77.90, 77.15, 74.05, 78.50,
             74.90, 80.55, 82.80, 74.10, 74.00],
            [70.80, 71.55, 70.25, 68.50, 70.30,
             65.45, 70.15, 70.45, 66.00, 71.35],
            [60.30, 62.20, 61.30, 53.80, 57.15,
             62.75, 62.50, 62.00, 60.40, 61.85],
        ],
        [
            [67.4, 65, 65.25, 64.15, 63.3],
            [73.95, 73.2, 68.5, 74.7, 73.95],
            [64, 70.6, 70.25, 70.95, 69.4],
            [86.25, 86.35, 83.45, 84.45, 79.8]
        ],
        [
            [57.15, 56.45, 55.5, 57.2, 55.05],
            [55.3, 51.1, 52.9, 50.75, 51.9],
            [69.75, 65, 58.55, 57, 72.6]
        ],
        [
            [55.2, 51.05, 56.2, 56.7, 56.45],
            [61.85, 63.1, 59.45, 63.7, 62.6],
        ],
        [
            [57.3, 60.35, 50.1, 54.2, 55.75]
        ]
    ]
    raw_data_loss = [
        [57.16, 56.82, 57.18, 57.14, 56.37, 57.1, 56.39, 57.13, 56.99, 56.8],
        [50, 50, 50, 50, 50, 50, 50, 49.95, 50],
        [50, 50, 50, 50, 50, 50, 49.95, 50],
        [50, 50, 50, 50, 50, 49.95, 50],
        [50, 50, 50, 50, 49.95, 50],
        [49.97, 50, 50, 49.95, 50],
        [50.06, 50, 49.95, 50],
        [49.98, 49.95, 50],
        [49.95, 50],
        [50.05]
    ]

    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return (raw_data_meta, raw_data_threshold, raw_data_loss, ratios)


def get_census_race():
    raw_data_meta = [
        [
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 99.95, 100.0, 100.0, 100.0,
             100.0, 99.95, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 99.95, 100.0, 100.0, 99.95]
        ],
        [
            [51.8, 51.35, 51.4, 52.0, 52.3],
            [55.8, 55.6, 53.9, 55.65, 55.1],
            [55.9, 56.55, 56.45, 54.55, 55.9],
            [56.25, 53.75, 56.6, 56.3, 55.55, 57.55, 55.8, 55.95, 56.3, 58.7],
            [50.6, 53.15, 52.55, 51.95, 55.15],
            [51.05, 51.25, 50.85, 49.4, 50.25],
            [51.0, 50.9, 50.9, 51.4, 50.55],
            [51.46, 52, 51.86, 51.01, 50.47],
            [99.95, 99.9, 100.0, 100.0, 99.95,
             100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [52.5, 56.05, 52.95, 54.7, 54.45],
            [58.3, 55.25, 58.95, 58.95, 56.25],
            [55.1, 53.8, 54.65, 53.0, 53.4, 53.7, 53.65, 54.65, 54.7, 52.2],
            [54.1, 53.8, 53.8, 52.5, 53.0],
            [50.85, 51.1, 51.1, 51.65, 50.9],
            [51.0, 49.55, 50.6, 50.05, 49.9],
            [51.31, 50.42, 49.78, 52.7, 50.96],
            [99.95, 99.9, 100.0, 100.0, 99.95,
             100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [50.0, 49.1, 53.4, 49.05, 49.4, 48.75, 50.35, 50.45, 49.9, 50.6],
            [54.45, 53.1, 52.85, 52.35, 51.75, 53.05, 51.85, 53.25, 52.25, 51.8],
            [50.4, 51.3, 53.05, 52.45, 50.35, 51.0, 52.55, 50.05, 51.95, 51.65],
            [50.55, 49.85, 49.85, 49.9, 50.65, 50.25, 49.55, 49.0, 49.6, 49.6],
            [49.4, 49.9, 50.8, 49.55, 49.85, 49.6, 49.95, 50.0, 48.95, 49.5],
            [52.25, 50.22, 49.68, 51.86, 50.12,
             51.81, 50.82, 50.62, 51.31, 50.42],
            [100.0, 100.0, 100.0, 100.0, 99.95,
             100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [50.25, 51.6, 49.2, 49.6, 50.05, 49.7, 49.35, 50.2, 50.4, 50.85],
            [51.55, 49.4, 50.45, 50.1, 51.1, 51.4, 51.55, 49.65, 51.25, 51.75],
            [51.15, 52.1, 48.75, 49.75, 53.35,
             51.05, 53.95, 48.95, 51.9, 53.7],
            [48.6, 48.65, 48.65, 50.75, 50.35, 54.1, 49.25, 52.6, 54.1, 50.9],
            [52.30, 53.19, 52.70, 55.32, 49.04,
             48.44, 54.82, 55.22, 54.38, 51.61],
            [100.0, 100.0, 100.0, 100.0, 99.95,
             100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [51.15, 49.3, 51.15, 48.7, 49.7, 49.55, 50.35, 50.3, 50.25, 51.1],
            [51.95, 51.8, 49.45, 49.9, 50.9, 51.7, 53.9, 49.45, 49.55, 49.25],
            [54.95, 52.4, 56.3, 49.9, 51.1, 56.7, 54.9, 50.4, 54.3, 49.5],
            [51.26, 51.21, 51.56, 53.39, 51.41,
             54.13, 49.73, 51.81, 50.07, 53.19],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [50.55, 51.15, 50.7, 50.4, 52.15, 50.75, 51.8, 51.9, 50.7, 51.05],
            [50.5, 50.1, 50.55, 52.95, 51.15,
             53.05, 52.15, 54.55, 51.75, 52.55],
            [59.03, 51.41, 56.70, 56.51, 57.74,
             58.63, 54.92, 54.73, 55.47, 56.61],
            [99.95, 100.0, 100.0, 100.0, 99.95,
             99.95, 99.95, 100.0, 100.0, 99.95]
        ],
        [
            [51.9, 56.55, 48.4, 53.25, 53.3, 55.7, 53.95, 52.85, 54.15, 54.2],
            [65.22, 67.49, 59.77, 59.52, 57.15,
             63.14, 64.67, 62.59, 65.46, 60.47],
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0],
        ],
        [
            [59.43, 55.81, 60.91, 57.40, 56.56,
             55.76, 54.18, 55.22, 57.01, 57.25],
            [99.95, 99.95, 100, 100, 99.95, 99.95, 99.95, 100, 100, 99.95],
        ],
        [
            [100.0, 100.0, 100.0, 100.0, 100.0,
             100.0, 100.0, 100.0, 100.0, 100.0]
        ]
    ]
    raw_data_threshold = [
        [
            [65.75, 58.90, 53.65, 56.10, 69.95],
            [74.70, 71.65, 59.25, 73.55, 72.25],
            [75.95, 75.40, 77.25, 76.85, 74.80],
            [80.15, 71.25, 68.10, 73.55, 71.55],
            [80.10, 79.70, 72.90, 78.65, 78.20,
             72.10, 78.50, 77.45, 75.25, 79.90],
            [86.00, 81.95, 83.80, 83.35, 85.40],
            [86.15, 90.40, 85.70, 87.75, 91.15],
            [77.75, 90.40, 87.90, 83.30, 91.15],
            [76.75, 88.35, 85.40, 88.10, 88.20],
            [94.40, 96.45, 93.95, 93.95, 89.10]
        ],
        [
            [46.50, 54.35, 46.55, 55.05, 54.95],
            [56.80, 59.95, 52.85, 60.20, 60.35],
            [64.85, 64.65, 66.35, 67.00, 68.05],
            [69.15, 70.70, 63.50, 63.75, 63.30,
             56.80, 62.95, 72.50, 68.95, 68.25],
            [84.40, 80.35, 75.80, 82.45, 78.35],
            [80.85, 89.75, 89.95, 88.20, 90.65],
            [87.20, 83.45, 86.70, 88.30, 86.55],
            [83.25, 87.45, 83.20, 85.10, 82.20],
            [92.60, 93.00, 89.85, 96.60, 96.25]
        ],
        [
            [53.45, 54.85, 51.95, 53.25, 51.05],
            [59.65, 60.90, 58.15, 59.35, 60.40],
            [66.15, 62.00, 64.85, 60.95, 61.80,
             66.10, 63.65, 64.20, 63.35, 60.85],
            [75.15, 83.70, 79.70, 83.65, 81.30],
            [85.65, 87.95, 82.90, 86.80, 87.80],
            [84.15, 68.30, 86.20, 85.00, 86.15],
            [84.15, 79.50, 81.90, 86.20, 88.00],
            [88.10, 93.45, 93.70, 91.85, 88.45]
        ],
        [
            [56.45, 55.95, 55.90, 55.20, 56.25],
            [60.85, 61.00, 58.55, 67.60, 60.20,
             52.85, 62.50, 65.05, 64.25, 57.75],
            [68.80, 71.90, 75.65, 68.20, 76.05],
            [80.60, 85.80, 82.30, 84.00, 81.15],
            [78.70, 82.15, 76.40, 76.05, 77.50],
            [78.45, 80.65, 70.85, 77.55, 81.40],
            [87.85, 88.70, 89.80, 86.30, 89.80]
        ],
        [
            [52.45, 60.75, 57.75, 58.20, 57.95,
             57.85, 58.70, 58.60, 57.40, 59.60],
            [68.50, 62.60, 57.15, 64.95, 67.00],
            [74.60, 78.35, 72.95, 68.15, 78.65],
            [74.75, 75.55, 74.30, 73.05, 69.70],
            [72.50, 59.20, 71.90, 67.75, 65.45],
            [83.25, 81.85, 78.60, 81.50, 80.75]
        ],
        [
            [53.70, 54.55, 56.25, 57.20, 58.30,
             55.15, 56.70, 53.65, 55.20, 54.65],
            [58.70, 63.05, 62.85, 60.30, 65.05,
             63.70, 62.00, 62.25, 65.95, 65.80],
            [65.70, 58.70, 63.80, 62.20, 53.80,
             59.30, 60.70, 56.85, 55.85, 58.75],
            [55.15, 51.35, 50.70, 54.40, 54.75,
             51.15, 51.65, 52.55, 51.05, 50.35],
            [50.95, 59.00, 58.00, 59.00, 56.85,
             59.00, 56.30, 59.70, 59.00, 59.70]
        ],
        [
            [55.60, 57.70, 58.85, 56.20, 58.10],
            [55.40, 54.40, 53.95, 53.35, 56.35],
            [51.15, 53.05, 47.75, 51.50, 55.20],
            [69.45, 61.15, 67.05, 64.10, 68.40]
        ],
        [
            [54.00, 53.65, 53.65, 47.30, 52.45],
            [57.60, 60.35, 58.30, 56.75, 60.35],
            [66.65, 59.60, 58.75, 65.45, 61.40]
        ],
        [
            [54.90, 55.95, 55.85, 55.45, 55.85],
            [56.75, 61.85, 53.95, 56.50, 59.95]
        ],
        [
            [56.20, 56.80, 58.05, 52.40, 43.85]
        ]
    ]
    raw_data_loss = [
        [72.09, 70.36, 68.51, 71.42, 57, 69.31, 70.73, 69.73, 70.72, 71.1],
        [50.31, 50.29, 50.12, 50, 50, 50.14, 50.04, 50, 50],
        [50.23, 50, 50, 49.95, 50, 49.95, 49.95, 49.95],
        [49.21, 50, 50.05, 52.74, 50, 50, 50],
        [50, 47.28, 51.41, 50.01, 49.97, 49.95],
        [50, 50.35, 50.05, 50.05, 50.05],
        [50.62, 50.63, 51.22, 50],
        [49.18, 50, 50.02],
        [51.92, 50.06],
        [51.74]
    ]

    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return (raw_data_meta, raw_data_threshold, raw_data_loss, ratios)


def get_celeba_young_data():
    raw_data_meta = [
        [
            [50.35, 51.1, 49.65, 53.3, 49.9],
            [50.4, 57.25, 49.75, 59.85, 55.25],
            [68.15, 56.05, 65.15, 63.5, 69.2],
            [61.45, 72.8, 75.9, 75.4, 74.85],
            [78.85, 50.1, 79.8, 79.2, 75.15],
            [84.6, 84.2, 83.85, 78.05, 79.9],
            [84.8, 49.9, 88.45, 89.2, 88.75],
            [84.25, 84.05, 90.65, 91.4, 88.1],
            [93.7, 89.6, 94.5, 93.4, 93.75],
            [94.5, 94.8, 95.3, 95.0, 95.9]
        ],
        [
            [50.95, 49.7, 49.15, 48.25, 47.85],
            [58.2, 50.85, 57.95, 57.25, 48.95],
            [50.25, 65.1, 50.8, 62.2, 63.65],
            [70.9, 68.15, 70.1, 70.15, 69.05],
            [75.1, 62.25, 71.8, 73.25, 76.2],
            [79.6, 80.1, 79.0, 80.45, 78.6],
            [84.4, 51.05, 80.45, 76.15, 84.95],
            [89.55, 89.75, 83.35, 84.95, 89.8],
            [92.0, 92.05, 92.35, 93.05, 92.05]
        ],
        [
            [51.7, 51.45, 48.5, 51.3, 49.8],
            [55.15, 54.0, 56.65, 58.65, 52.15],
            [60.05, 63.6, 63.55, 63.55, 63.8],
            [68.15, 66.05, 50.35, 49.0, 65.15],
            [74.25, 75.5, 51.0, 73.65, 73.7],
            [80.15, 79.2, 79.3, 80.05, 78.75],
            [83.9, 80.9, 77.9, 84.75, 83.7],
            [89.85, 88.7, 90.6, 90.55, 90.2],
        ],
        [
            [51.2, 51.1, 49.7, 51.0, 49.5],
            [54.15, 52.75, 53.0, 50.9, 56.7],
            [48.55, 59.4, 57.9, 57.2, 60.1],
            [67.4, 67.15, 53.55, 51.0, 66.6],
            [71.65, 71.95, 72.15, 58.5, 59.15],
            [74.0, 76.35, 79.5, 78.05, 79.95],
            [84.2, 85.05, 85.5, 49.6, 84.7],
        ],
        [
            [48.7, 50.7, 49.35, 51.1, 49.4],
            [53.85, 56.1, 53.9, 51.1, 54.9],
            [55.7, 59.55, 50.0, 60.15, 60.25],
            [65.3, 63.55, 65.05, 65.35, 67.35],
            [67.4, 73.65, 73.9, 50.0, 71.35],
            [80.45, 79.95, 67.65, 69.75, 80.05]
        ],
        [
            [49.45, 49.35, 49.55, 50.6, 51.8],
            [50.15, 52.1, 55.4, 54.75, 49.5],
            [61.65, 49.95, 54.55, 50.75, 61.15],
            [66.1, 66.3, 64.45, 62.55, 67.0],
            [73.85, 50.85, 74.95, 76.4, 74.05]
        ],
        [
            [52.1, 48.8, 51.4, 51.55, 50.0],
            [51.1, 52.1, 52.6, 51.8, 48.85],
            [55.2, 61.15, 62.45, 63.15, 58.15],
            [69.95, 69.8, 68.1, 50.0, 65.6]
        ],
        [
            [49.85, 50.4, 50.3, 49.55, 52.0],
            [54.0, 53.8, 49.75, 56.6, 53.55],
            [64.05, 66.2, 59.95, 59.6, 55.35]
        ],
        [
            [51.7, 51.5, 50.5, 51.25, 50.15],
            [50.0, 56.75, 58.35, 54.45, 50.85]
        ],
        [
            [49.85, 52.75, 52.45, 49.8, 51.3]
        ]
    ]
    raw_data_threshold = [
        [
            [49.77, 49.72, 50.28],
            [48.37, 54.76, 48.27],
            [50.25, 50.35, 50.25],
            [48.32, 50.3, 50.25],
            [50.27, 50.28, 50.28],
            [49.07, 49.02, 48.92],
            [50.03, 50.03, 50.41],
            [39.52, 49.52, 49.52],
            [76.06, 50.33, 50.23],
            [50.48, 88.89, 50.63]
        ],
        [
            [47.94, 52.1, 52.06],
            [49.98, 49.98, 49.98],
            [50.65, 49.75, 42.7],
            [49.95, 50, 49.95],
            [48.13, 52.07, 48.08],
            [49.15, 49.55, 49.65],
            [49.14, 49.19, 49.14],
            [58.83, 49.85, 67.85],
            [80.5, 75.35, 80.3]
        ],
        [
            [52.03, 52.03, 47.97],
            [52.06, 48.52, 52.0],
            [51.95, 48, 52.06],
            [49.79, 55.81, 50.16],
            [55.31, 51.28, 50.97],
            [51.24, 52.79, 55.37],
            [53.12, 60.78, 58.07],
            [70.8, 66.06, 56.33]
        ],
        [
            [50.03, 49.1, 48.9],
            [49.93, 49.88, 49.78],
            [51.33, 47.13, 41.48],
            [50.63, 51.03, 50.33],
            [49.16, 49.21, 51.04],
            [52.43, 54.41, 55.14],
            [56.08, 50.03, 64.18]
        ],
        [
            [51.4, 50.05, 53.35],
            [55.75, 60.92, 61.59],
            [65.76, 71.35, 60.99],
            [61.74, 67.38, 68.24],
            [72.02, 74.52, 83.25],
            [90.85, 90.8, 83.2]
        ],
        [
            [51.15, 48.65, 51.87],
            [50.90, 50.7, 52.06],
            [55.34, 51.17, 51.78],
            [51.91, 51.96, 50.60],
            [53.35, 59.55, 54.2],
        ],
        [
            [52.9, 52.49, 49.36],
            [57.36, 56.01, 54.29],
            [51.62, 58.08, 55.88],
            [67.32, 70.99, 60.05]
        ],
        [
            [51.02, 51.12, 50.92],
            [53.42, 50.50, 50.6],
            [52.41, 54.02, 52.41]
        ],
        [
            [49.54, 49.38, 49.38],
            [49.29, 49.24, 49.39]
        ],
        [
            [50.1, 49.85, 49.85]
        ]
    ]
    raw_data_loss = [
        [52.56, 56.11, 61.73, 59.91, 57.7, 83.25, 84.07, 81.94, 85.51, 98.4],
        [52.56, 55.37, 50.6, 55.97, 79.34, 88.53, 83.97, 81.04, 96.67],
        [51.89, 47.61, 59.95, 64.5, 79.35, 75.67, 85.25, 93.88],
        [50.51, 51.55, 53.56, 67.96, 66.36, 84.08, 90.47],
        [52.08, 62.8, 77.01, 70.3, 85.99, 95.85],
        [47.83, 50.13, 71.48, 62.82, 86.9],
        [57.09, 66.22, 73.07, 88.55],
        [55.68, 58.78, 85.23],
        [50.03, 73.89],
        [63.5]
    ]

    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return (raw_data_meta, raw_data_threshold, raw_data_loss, ratios)


def get_celeba_sex_data():
    raw_data_meta = [
        [
            [58.3, 56.8, 57.6, 58.35, 53.95],
            [63.35, 65.8, 66.05, 61.2, 65.35],
            [76.7, 69.8, 76.25, 67.5, 74.0],
            [77.85, 78.2, 82.3, 87.15, 85.2],
            [93.2, 89.65, 93.25, 92.0, 91.65],
            [95.25, 90.45, 88.45, 91.5, 93.3],
            [95.15, 95.75, 94.35, 94.25, 95.2],
            [87.4, 85.25, 65.25, 90.4, 89.85],
            [95.45, 87.15, 93.45, 96.7, 86.45],
            [75.25, 61.55, 62.1, 67.1, 76.15]
        ],
        [
            [52.55, 55.35, 52.0, 51.55, 53.95],
            [61.7, 61.85, 60.2, 63.5, 60.6],
            [68.6, 74.75, 66.15, 70.9, 72.9],
            [68.55, 81.35, 77.5, 74.85, 80.65],
            [74.45, 85.95, 86.55, 83.5, 86.25],
            [88.6, 91.2, 91.15, 89.05, 89.2],
            [80.8, 69.15, 74.25, 70.6, 65.05],
            [80.7, 86.7, 83.55, 86.25, 89.55],
            [80.4, 67.95, 61.0, 81.9, 68.85]
        ],
        [
            [52.7, 52.85, 51.6, 52.85, 52.75],
            [57.8, 58.85, 57.75, 58.45, 58.0],
            [67.5, 61.7, 65.9, 59.5, 59.3],
            [68.15, 74.1, 77.4, 73.75, 73.8],
            [80.6, 69.15, 79.85, 76.95, 78.35],
            [81.2, 77.4, 77.75, 74.5, 81.85],
            [83.0, 82.7, 80.4, 86.75, 84.4],
            [76.95, 67.9, 67.9, 74.25, 67.2]
        ],
        [
            [54.9, 54.85, 54.15, 55.5, 51.75],
            [60.95, 57.8, 60.45, 60.0, 56.3],
            [70.55, 74.5, 71.95, 71.85, 64.0],
            [79.7, 80.45, 72.25, 72.8, 62.85],
            [75.8, 76.95, 62.05, 62.75, 60.95],
            [80.4, 80.9, 72.75, 86.05, 85.25],
            [67.2, 65.65, 65.5, 67.85, 71.45]
        ],
        [
            [53.0, 52.95, 54.0, 53.9, 51.55],
            [62.35, 60.8, 60.8, 61.5, 55.7],
            [67.2, 69.65, 67.95, 68.3, 67.75],
            [51.95, 74.95, 51.65, 60.65, 52.0],
            [76.55, 72.25, 66.25, 65.8, 64.85],
            [59.85, 65.25, 60.35, 62.2, 59.65]
        ],
        [
            [52.5, 53.35, 54.75, 53.35, 51.5],
            [59.7, 55.5, 61.35, 60.8, 62.2],
            [51.85, 62.9, 52.1, 60.25, 51.85],
            [70.4, 74.2, 73.1, 70.5, 69.1],
            [67.45, 66.2, 60.1, 69.05, 65.4]
        ],
        [
            [52.2, 50.05, 52.3, 51.85, 53.6],
            [48.1, 49.75, 48.15, 51.3, 51.4],
            [60.05, 63.3, 65.45, 65.9, 61.65],
            [58.85, 59.25, 62.4, 58.0, 59.25]
        ],
        [
            [49.55, 49.3, 49.55, 49.95, 49.5],
            [55.95, 54.5, 57.9, 58.8, 56.55],
            [55.0, 52.55, 58.2, 58.25, 57.9]
        ],
        [
            [48.45, 47.2, 47.7, 48.8, 48.5],
            [52.8, 54.85, 54.5, 56.5, 54.3]
        ],
        [
            [50.55, 51.4, 50.9, 53.6, 54.0]
        ],
    ]
    raw_data_threshold = [
        [
            [50.68, 50.03, 51.24],
            [56.29, 55.24],  # ?], # Running
            [55.56, 52.84, 55.20],
            [56.25, 57.01, 61.31],
            [54.2, 56.9, 58.05],
            [59.09, 60.98, 61.03],
            [67.37, 61.94, 63.95],
            [56.09, 55.24],  # ?], # Running
            [76.65, 76.49, 75.68],
            [76.1, 76.4, 74.8]
        ],
        [
            [51.15, 48.9, 52.36],
            [50.36, 51.98, 52.74],
            [53.83, 51.43, 53.67],
            [50.48, 55.57, 53.66],
            [58.63, 59.97, 57.71],
            [66.79, 59.99, 64.71],
            [48.05, 50.1, 44.74],
            [74.50, 73.43, 71.79],
            [73.58, 75.59, 75.79]
        ],
        [
            [51.05, 51.65, 50.65],
            [51.63, 51.95, 50.5],
            [54.56, 55.12, 50.93],
            [59.67, 58.67, 59.82],
            [62.68, 64.83, 65.93],
            [48.8, 46.94, 49.2],
            [72.8, 72.24, ],  # ?], #Running
            [77.49, 74.34, 78.0]
        ],
        [
            [50.15, 50.92, 50.61],
            [52.09, 53.34, 52.44],
            [54.05, 56.76, 55.17],
            [59, 63.2, 62],
            [41.58, 44.49],  # ?], #Running
            [67.65, 66.89, 71.57],
            [73.71, 71.54, 73.91]
        ],
        [
            [51.14, 52.05, 50.97],
            [56.34, 52.89, 56.91],
            [61.34, 58.24, 61.24],
            [43.99, 43.84, 39.68],
            [71.91, 65.71, 67.5],
            [68.64, 74.05, 73.04]
        ],
        [
            [55.68, 55.41, 51.96],
            [60.68, 60.43, 60.18],
            [54.48, 55.09, 50.0],
            [66.26, 68.39, 67.63],
            [72.45, 72.4, 69.05]
        ],
        [
            [53.79, 55.99, 56.3],
            [45.09, 47.14, 46.09],
            [62.1, 65.05, 59.94],
            [66.43, 69.59, 67.19]
        ],
        [
            [48.4, 48.3, 48.7],
            [58.64, 55.63, 57.52],
            [59.23, 62.75, 56.66]
        ],
        [
            [49.4, 49.5, 49.9],
            [52.31, 51.43, 51.33],
        ],
        [
            [54.86, 54.51, 53.7]
        ]
    ]
    raw_data_loss = [
        [53.12, 50.75, 52.13, 51.44, 51.15, 52.92, 50.86, 50.55, 51.74, 50.4],
        [50.1, 52.01, 51.59, 50.53, 50.14, 50.6, 50.5, 52.63, 50.19],
        [51.6, 50.85, 50.36, 50.83, 50.19, 50.84, 49.95, 50.14],
        [50.06, 50.01, 54.65, 51.06, 48.75, 50.32, 49.64],
        [50.44, 56.9, 51.91, 50.1, 52.43, 50.19],
        [50.78, 57.26, 51.5, 46.62, 50.1],
        [52.19, 50.95, 51.53, 48.62],
        [49.15, 50.1, 49.85],
        [50.4, 49.15],
        [50.42]
    ]

    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return (raw_data_meta, raw_data_threshold, raw_data_loss, ratios)


def get_boneage_data():
    raw_data_meta = [
        [
            [50.0, 86.55, 87.55, 85.45, 79.6, 75.5, 87.1, 85.05, 86.5, 88.35],
            [98.8, 97.7, 98.0, 99.2, 99.7, 96.65, 99.6, 98.3, 98.2, 98.6],
            [99.55, 99.7, 99.7, 99.55, 99.9, 99.95, 99.7, 98.05, 99.95, 99.8],
            [99.9, 99.95, 99.95, 99.85, 99.95, 99.9, 99.85, 99.9, 99.9, 99.95],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [86.65, 88.2, 83.7, 61.0, 85.25, 76.05, 54.1, 85.15, 77.2, 63.4],
            [94.7, 95.7, 93.45, 92.4, 95.35, 97.25, 96.6, 94.25, 95.9, 98.0],
            [50.0, 99.1, 98.15, 99.5, 99.7, 99.2, 95.75, 99.15, 98.65, 98.85],
            [100.0, 99.9, 100.0, 99.95, 99.9, 99.85, 99.9, 99.95, 99.95, 99.95],
            [100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 99.9, 100.0, 100.0, 100.0]
        ],
        [
            [70.95, 63.85, 63.25, 69.75, 59.2, 66.3, 74.6, 63.4, 69.45, 65.55],
            [79.8, 93.55, 88.1, 79.35, 93.25, 95.5, 83.1, 84.35, 95.55, 94.85],
            [97.95, 98.65, 97.35, 93.8, 99.25, 97.85, 92.8, 91.25, 98.1, 99.55],
            [98.4, 99.95, 99.5, 99.65, 100.0, 100.0, 100.0, 100.0, 100.0, 99.95]
        ],
        [
            [50.0, 58.9, 61.55, 67.0, 58.55, 62.15, 59.85, 58.75, 62.65, 65.65],
            [70.2, 90.3, 75.7, 83.75, 86.0, 89.45, 85.1, 83.7, 78.35, 72.35],
            [99.45, 99.2, 98.75, 98.4, 97.1, 92.05, 96.0, 96.75, 96.6, 97.35]
        ],
        [
            [50.25, 56.6, 57.55, 60.75, 53.8, 57.45, 56.35, 54.2, 53.35, 59.1],
            [76.5, 77.65, 89.95, 65.15, 66.6, 74.0, 76.0, 69.8, 80.25, 77.9]
        ],
        [
            [58.55, 54.9, 57.55, 56.65, 52.45, 54.45, 51.2, 50.75, 53.35, 50.95]
        ]
    ]
    raw_data_threshold = [
        [
            [50.85, 52.35, 50.4, 52.5, 52.3],
            [51, 57.6, 55.35, 50.4, 51.2],
            [58.10, 68.35, 57.50, 63.55, 60.10, 63.05, 62.10, 57.75, 63.60, 65.20],
            [72.75, 66.95, 80.55, 64.4, 62.95],
            [81.55, 70.5, 82.2, 78.25, 75.25],
            [88.8, 86.8, 80.75, 82.1, 89.1]
        ],
        [
            [54.75, 52.8, 50.4, 54.75, 51.1],
            [58.05, 59.40, 51.30, 59.25, 54.30, 56.55, 54.80, 58.95, 55.50, 57.85],
            [62.95, 66.2, 64.65, 63.3, 60.05],
            [67.75, 72.15, 71, 69.15, 69.05],
            [83.8, 69.45, 66.05, 80.05, 77.15]
        ],
        [
            [55.55, 51.95, 53.15, 51.60, 52.85, 51.50, 50.35, 52.50, 51.25, 50.60],
            [56.35, 63, 55.25, 57.75, 58],
            [67.1, 63.55, 64.25, 64.85, 66.65],
            [67.75, 72, 78, 70.35, 67.55]
        ],
        [
            [50.10, 52.10, 50.45, 51.45, 50.80, 53.75, 50.45, 53.00, 50.35, 53.95],
            [52.75, 56.20, 51.60, 51.50, 52.75, 58.70, 53.70, 55.60, 53.25, 54.10],
            [55.85, 64.45, 58.40, 56.40, 59.05, 64.65, 50.45, 57.25, 54.60, 69.95]
        ],
        [
            [52.45, 53.7, 52.4, 50.45, 50.3],
            [58.75, 58.25, 62.7, 51.4, 60.35]
        ],
        [
            [54.25, 51.15, 56.6, 52.25, 55.5]
        ]
    ]
    raw_data_loss = [
        [53.5, 60.05, 64.9, 80.8, 86.35, 84.6],
        [50.4, 64.3, 71.55, 69.8, 87.35],
        [59.0, 61.5, 65.65, 73.65],
        [57.0, 56.8, 66.9],
        [50.45, 65.1],
        [54.8]
    ]

    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    return (raw_data_meta, raw_data_threshold, raw_data_loss, ratios)


def process_data(meta, threshold, raw, ratios):
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
    for i in range(len(ratios)):
        hmm = 0
        for j in range(i + 1, len(ratios)):
            r1, r2 = ratios[i], ratios[j]
            # If > 0.5, consider 1 - ratios
            if min(r1, r2) > 0.5:
                r1 = 1 - r1
                r2 = 1 - r2
            r1, r2 = round(r1, 1), round(r2, 1)

            if picked_ratio != r1 and picked_ratio != r2:
                continue

            if picked_ratio == r1:
                other_ratio = r2
            else:
                other_ratio = r1

            wanted_r = "%.1f" % picked_ratio
            collection[wanted_r] = collection.get(
                wanted_r, []) + [(other_ratio, raw[i][hmm] / 100)]
            hmm += 1

    return collection


def find_n_eff(points_x, points_y, ratio):
    start_n = 1
    n_max = 10000
    # Find minimum N that corresponds to an upper-bound for ratios
    for i in range(start_n, n_max + 1):
        uppers = np.array([bound(x, ratio, i) for x in points_x])
        satisfied = np.all(points_y <= uppers)
        if satisfied:
            return i


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=14)

    # plt.style.use('dark_background')

    # Plot curves corresponding to ratios
    picked_ratio = 0.2
    colors = ['blue', 'orange', 'green', 'lightcoral', 'purple']
    # curve_colors = ['mediumpurple', 'olive', 'firebrick']
    curve_colors = ['mediumpurple', 'darkgray', 'lightgray']
    n_effs = []
    x_axis = np.linspace(0, 1, 1000)

    # BoneAge
    boneage = process_data(*get_boneage_data())
    x_all, y_all = [], []
    for k, v in boneage.items():
        if k != "%.1f" % picked_ratio:
            continue
        x = [p[0] for p in v]
        y = [p[1] for p in v]
        x_all.append(x)
        y_all.append(y)
        plt.scatter(x, y, color=colors[0], edgecolors='face', marker='D')
    x_all = np.concatenate(x_all, 0)
    y_all = np.concatenate(y_all, 0)
    n_eff = find_n_eff(x_all, y_all, picked_ratio)
    n_effs.append(n_eff)
    print("N-eff Boneage: %d" % n_eff)

    # CelebA: female
    celeba_sex = process_data(*get_celeba_sex_data())
    x_all, y_all = [], []
    for k, v in celeba_sex.items():
        if k != "%.1f" % picked_ratio:
            continue
        x = [p[0] for p in v]
        y = [p[1] for p in v]
        x_all.append(x)
        y_all.append(y)
        plt.scatter(x, y, color=colors[1], edgecolors='face', marker='*')
    x_all = np.concatenate(x_all, 0)
    y_all = np.concatenate(y_all, 0)
    n_eff = find_n_eff(x_all, y_all, picked_ratio)
    n_effs.append(n_eff)
    print("N-eff Celeba (female): %d" % n_eff)

    # Celeba: old
    celeba_young = process_data(*get_celeba_young_data())
    x_all, y_all = [], []
    for k, v in celeba_young.items():
        if k != "%.1f" % picked_ratio:
            continue
        x = [p[0] for p in v]
        y = [p[1] for p in v]
        x_all.append(x)
        y_all.append(y)
        plt.scatter(x, y, color=colors[2], edgecolors='face', marker='*')
    x_all = np.concatenate(x_all, 0)
    y_all = np.concatenate(y_all, 0)
    n_eff = find_n_eff(x_all, y_all, picked_ratio)
    n_effs.append(n_eff)
    print("N-eff Celeba (old): %d" % n_eff)

    # Census: sex
    census_sex = process_data(*get_census_sex())
    x_all, y_all = [], []
    for k, v in census_sex.items():
        if k != "%.1f" % picked_ratio:
            continue
        x = [p[0] for p in v]
        y = [p[1] for p in v]
        x_all.append(x)
        y_all.append(y)
        plt.scatter(x, y, color=colors[3], edgecolors='face', marker='s')
    x_all = np.concatenate(x_all, 0)
    y_all = np.concatenate(y_all, 0)
    n_eff = find_n_eff(x_all, y_all, picked_ratio)
    n_effs.append(n_eff)
    print("N-eff Census (sex): %d" % n_eff)

    # Census: race
    census_race = process_data(*get_census_race())
    x_all, y_all = [], []
    for k, v in census_race.items():
        if k != "%.1f" % picked_ratio:
            continue
        x = [p[0] for p in v]
        y = [p[1] for p in v]
        x_all.append(x)
        y_all.append(y)
        plt.scatter(x, y, color=colors[4], edgecolors='face', marker='s')
    x_all = np.concatenate(x_all, 0)
    y_all = np.concatenate(y_all, 0)
    n_eff = find_n_eff(x_all, y_all, picked_ratio)
    n_effs.append(n_eff)
    print("N-eff Census (race): %d" % n_eff)

    n_effs = sorted(list(set(n_effs)))[::-1]
    # Swap out
    n_effs[0] = 8
    n_effs = sorted(n_effs)[::-1]

    for cc, n_eff in zip(curve_colors, n_effs):
        if n_eff == 37:
            continue
        plt.plot(x_axis, [bound(x_, picked_ratio, n_eff)
                          for x_ in x_axis], '--', color=cc, label=r"$n_{leaked}=%d$" % n_eff)

    # Trick to get desired legend
    plt.plot([], [], color=colors[0], marker="D", ms=10, ls="", label="RSNA Bone Age")
    plt.plot([],[], color=colors[1], marker="*", ms=10, ls="", label="CelebA (female)")
    plt.plot([], [], color=colors[2], marker="*", ms=10, ls="", label="CelebA (young)")
    plt.plot([], [], color=colors[3], marker="s", ms=10, ls="", label="Census (female)")
    plt.plot([], [], color=colors[4], marker="s", ms=10, ls="", label="Census (white)")

    plt.xlabel(r'$\alpha_1$')
    plt.ylabel(r'Accuracy')
    plt.ylim(0.5, 1)
    plt.xticks(np.arange(min(x), max(x)+0.1, 0.1))
    # plt.grid()
    plt.style.use('seaborn')
    plt.legend()
    plt.savefig("./bound_curves_%.1f.pdf" % picked_ratio)

    # print(bound(0.1, 0.2, 2))
    # print(bound(0.5, 0.6, 2))