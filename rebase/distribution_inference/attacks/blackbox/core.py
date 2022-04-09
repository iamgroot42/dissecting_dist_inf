from typing import List, Tuple, Callable
import numpy as np
from tqdm import tqdm

from distribution_inference.config import BlackBoxAttackConfig


class PredictionsOnOneDistribution:
    def __init__(self,
                 preds_property_1: List,
                 preds_property_2: List):
        """
            Wrapper to store predictions for models
            with two different training distributions.
        """
        self.preds_property_1 = preds_property_1
        self.preds_property_2 = preds_property_2


class PredictionsOnDistributions:
    """
        Wrapper to store predictions on two distributions,
        for models trained on two different training distributions.
    """
    def __init__(self,
                 preds_on_distr_1: PredictionsOnOneDistribution,
                 preds_on_distr_2: PredictionsOnOneDistribution):
        self.preds_on_distr_1 = preds_on_distr_1
        self.preds_on_distr_2 = preds_on_distr_2


class Attack:
    def __init__(self, config: BlackBoxAttackConfig):
        self.config = config

    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False):
        """
            Preds contain predictions on either of the two distributions
            on the top level. Inside, they contain predictions from models
            on first and second distributuions.
        """
        raise NotImplementedError("Attack not implemented")


def threshold_test_per_dist(cal_acc: Callable,
                            preds_adv: PredictionsOnOneDistribution,
                            preds_victim: PredictionsOnOneDistribution,
                            y_gt: np.ndarray,
                            config: BlackBoxAttackConfig,
                            epochwise_version: bool = False):
    """
        Perform threshold-test on predictions of adversarial and victim models,
        for each of the given ratios. Returns statistics on all ratios.
    """
    # Predictions made by the adversary's models
    p1, p2 = preds_adv.preds_property_1, preds_adv.preds_property_2
    # Predictions made by the  victim's models
    pv1, pv2 = preds_victim.preds_property_1, preds_victim.preds_property_2

    # Get optimal order of point
    order = order_points(p1, p2)

    # Order points according to computed utility
    p1 = np.transpose(p1)[order][::-1]
    p2 = np.transpose(p2)[order][::-1]
    if epochwise_version:
        pv1 = [np.transpose(x)[order][::-1] for x in pv1]
        pv2 = [np.transpose(x)[order][::-1] for x in pv2]
    else:
        pv1 = np.transpose(pv1)[order][::-1]
        pv2 = np.transpose(pv2)[order][::-1]
    yg = y_gt[order][::-1]

    adv_accs, allaccs_1, allaccs_2, f_accs = [], [], [], []
    # For all given percentile ratios
    for ratio in config.ratios:
        # Get first <ratio> percentile of points
        leng = int(ratio * p1.shape[0])
        p1_use, p2_use, yg_use = p1[:leng], p2[:leng], yg[:leng]
        if epochwise_version:
            pv1_use = [x[:leng] for x in pv1]
            pv2_use = [x[:leng] for x in pv2]
        else:
            pv1_use, pv2_use = pv1[:leng], pv2[:leng]

        # Calculate accuracies for these points in [0,100]
        accs_1 = 100 * cal_acc(p1_use, yg_use)
        accs_2 = 100 * cal_acc(p2_use, yg_use)

        # Find a threshold on these accuracies that maximizes
        # distinguishing accuracy
        tracc, threshold, rule = find_threshold_acc(
            accs_1, accs_2, granularity=config.granularity)
        adv_accs.append(100 * tracc)
        if epochwise_version:
            accs_victim_1 = [100 * cal_acc(pv1_use_inside, yg_use)
                             for pv1_use_inside in pv1_use]
            accs_victim_2 = [100 * cal_acc(pv2_use_inside, yg_use)
                             for pv2_use_inside in pv2_use]
        else:
            accs_victim_1 = 100 * cal_acc(pv1_use, yg_use)
            accs_victim_2 = 100 * cal_acc(pv2_use, yg_use)
        allaccs_1.append(accs_victim_1)
        allaccs_2.append(accs_victim_2)

        # Get accuracy on victim models using these thresholds
        if epochwise_version:
            combined = [np.concatenate((x, y)) for (
                x, y) in zip(accs_victim_1, accs_victim_2)]
            classes = [np.concatenate((np.zeros_like(x), np.ones_like(y))) for (
                x, y) in zip(accs_victim_1, accs_victim_2)]
            specific_acc = [get_threshold_acc(
                x, y, threshold, rule) for (x, y) in zip(combined, classes)]
            f_accs.append([100 * x for x in specific_acc])
        else:
            combined = np.concatenate((accs_victim_1, accs_victim_2))
            classes = np.concatenate(
                (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
            specific_acc = get_threshold_acc(
                combined, classes, threshold, rule)
            f_accs.append(100 * specific_acc)

    allaccs_1 = np.array(allaccs_1)
    allaccs_2 = np.array(allaccs_2)
    f_accs = np.array(f_accs)
    if epochwise_version:
        allaccs_1 = np.transpose(allaccs_1, (1, 0, 2))
        allaccs_2 = np.transpose(allaccs_2, (1, 0, 2))
        f_accs = f_accs.T

    return np.array(adv_accs), f_accs, (allaccs_1, allaccs_2)


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


def find_threshold_pred(pred_1, pred_2,
                        granularity: float = 0.005,
                        verbose: bool = True):
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
    acc = get_threshold_pred(predictions_combined, ground_truth, thres, rules)
    return acc, thres, rules


def get_threshold_pred(X, Y, threshold, rule,
                       get_pred: bool = False,
                       confidence: bool = False):
    """
        Get distinguishing accuracy between distributions, given predictions
        for models on datapoints, and thresholds with prediction rules.
        Args:
            X: predictions for models on datapoints
            Y: ground truth for datapoints
            threshold: thresholds for prediction rules
            rule: prediction rules
            get_pred: whether to return predictions
            confidence: currently no real value. May be removed soon
    """
    # X Shape: (n_samples, n_models)
    # Y Shape: (n_models)
    # threshold shape: (n_samples)
    if X.shape[1] != Y.shape[0]:
        raise ValueError('Dimension mismatch between X and Y: %d and %d should match' % (
            X.shape[1], Y.shape[0]))
    if X.shape[0] != threshold.shape[0]:
        raise ValueError('Dimension mismatch between X and threshold: %d and %d should match' % (
            X.shape[0], threshold.shape[0]))
    res = []

    # For each model
    for i in range(X.shape[1]):
        # Compute expected P[distribution=1] using given data, threshold, rules
        prob = np.average((X[:, i] <= threshold) == rule)

        if confidence:
            # Store direct average P[distribution=1]
            res.append(prob)
        else:
            # If majority (>0.5) points indicate some distribution,
            # it must be that one indeed
            res.append(prob >= 0.5)

    res = np.array(res)
    if confidence:
        acc = np.mean((res >= 0.5) == Y)
    else:
        acc = np.mean(res == Y)

    # Return predictions, if requested
    if get_pred:
        return res, acc
    return acc


def order_points(p1s, p2s):
    """
        Estimate utility of individual points, done by taking
        absolute difference in their predictions.
    """
    abs_dif = np.absolute(np.sum(p1s, axis=0) - np.sum(p2s, axis=0))
    inds = np.argsort(abs_dif)
    return inds
