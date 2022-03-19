import numpy as np


def order_points(p1s, p2s):
    """
        Estimate utility of individual points, done by taking
        absolute difference in their predictions.
    """
    abs_dif = np.absolute(np.sum(p1s, axis=0) - np.sum(p2s, axis=0))
    inds = np.argsort(abs_dif)
    return inds


def get_threshold_acc(X, Y, threshold, rule=None):
    """
        Get accuracy of predictions using given threshold,
        considering both possible (<= and >=) rules. Also
        return which of the two rules gives a better accuracy.
    """
    # Rule-1: everything above threshold is 1 class
    acc_1 = np.mean((X >= threshold) == Y)
    # Rule-2: everything below threshold is 1 class
    acc_2 = np.mean((X <= threshold) == Y)

    # If rule is specified, use that
    if rule == 1:
        return acc_1
    elif rule == 2:
        return acc_2

    # Otherwise, find and use the one that gives the best acc
    if acc_1 >= acc_2:
        return acc_1, 1
    return acc_2, 2


def find_threshold_acc(accs_1, accs_2, granularity: float = 0.1):
    """
        Find thresholds and rules for differentiating between two
        sets of predictions.
    """
    lower = min(np.min(accs_1), np.min(accs_2))
    upper = max(np.max(accs_1), np.max(accs_2))
    combined = np.concatenate((accs_1, accs_2))
    # Want to predict first set as 0s, second set as 1s
    classes = np.concatenate((np.zeros_like(accs_1), np.ones_like(accs_2)))
    best_acc = 0.0
    best_threshold = 0
    best_rule = None
    while lower <= upper:
        best_of_two, rule = get_threshold_acc(combined, classes, lower)
        if best_of_two > best_acc:
            best_threshold = lower
            best_acc = best_of_two
            best_rule = rule

        lower += granularity

    return best_acc, best_threshold, best_rule


def find_threshold_pred(pred_1, pred_2, granularity: float = 0.005, verbose: bool = True):
    """
        Find thresholds and rules for differentiating between two
        sets of predictions.
    """
    if pred_1.shape[0] != pred_2.shape[0]:
        raise ValueError('Dimension Mismatch')
    thres, rules = [], []
    iterator = range(pred_1.shape[0])
    if verbose:
        iterator = tqdm(iterator)
    for i in tqdm(iterator):
        _, t, r = find_threshold_acc(pred_1[i], pred_2[i], granularity)
        while r is None:
            granularity /= 10
            _, t, r = find_threshold_acc(pred_1[i], pred_2[i], granularity)
        thres.append(t)
        rules.append(r - 1)
    thres = np.array(thres)
    rules = np.array(rules)
    predictions_combined = np.concatenate((pred_1, pred_2), axis=1)
    ground_truth = np.concatenate((np.zeros(pred_1.shape[1]), np.ones(pred_2.shape[1])))
    acc = get_threshold_pred(predictions_combined, ground_truth), thres, rules)
    return acc, thres, rules
