from typing import List, Tuple, Callable, Union
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
        self.supports_saving_preds = False

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

    def wrap_preds_to_save(self, result: List):
        if self.supports_saving_preds:
            raise NotImplementedError("Method should be implemented if attack generated soft-labels")


def multi_model_sampling(arr, multi):
    res = np.zeros(arr.shape)
    if len(arr.shape) == 2:
        leng = arr.shape[1]

        for i in range(leng):
            use = arr[:, np.random.permutation(leng)[:multi]]
            use = np.average(use, axis=1)
            res[:, i] = use
    elif len(arr.shape) == 1:
        leng = arr.shape[0]
        for i in range(leng):
            use = arr[np.random.permutation(leng)[:multi]]
            use = np.average(use)
            res[i] = use
    else:
        raise ValueError("Dimension mismatch")
    return res

def _acc_per_dis(preds_d1: PredictionsOnOneDistribution,
               preds_d2: PredictionsOnOneDistribution,
               ground_truth,
               calc_acc: Callable,
               t = False,
               multi_class:bool=False):
        #pi means ith epoch
    p1 = [preds_d1.preds_property_1, preds_d1.preds_property_2]
    p2 = [preds_d2.preds_property_1, preds_d2.preds_property_2]
    transpose_order = (1, 0, 2) if multi_class else (1, 0)
    if not t:
        for i in range(2):
            p1[i] = np.transpose(p1[i],multi_class)
            p2[i] = np.transpose(p2[i],multi_class)
    acc1 = [100*calc_acc(p,ground_truth,multi_class=multi_class) for p in p1]
    acc2 = [100*calc_acc(p,ground_truth,multi_class=multi_class) for p in p2]
    return (np.array(acc1),np.array(acc2))
def threshold_test_per_dist(calc_acc: Callable,
                            preds_adv: PredictionsOnOneDistribution,
                            preds_victim: PredictionsOnOneDistribution,
                            y_gt: np.ndarray,
                            config: BlackBoxAttackConfig,
                            epochwise_version: bool = False):
    """
        Perform threshold-test on predictions of adversarial and victim models,
        for each of the given ratios. Returns statistics on all ratios.
    """
    assert not (
        epochwise_version and config.multi), "No implementation for both epochwise and multi model"
    assert not (config.multi2 and config.multi), "No implementation for both multi model"
    assert not (
        epochwise_version and config.multi2), "No implementation for both epochwise and multi model"
    # Predictions made by the adversary's models
    p1, p2 = preds_adv.preds_property_1, preds_adv.preds_property_2
    # Predictions made by the  victim's models
    pv1, pv2 = preds_victim.preds_property_1, preds_victim.preds_property_2

    # Get optimal order of point
    order = order_points(p1, p2)

    # Order points according to computed utility
    multi_class = config.multi_class
    transpose_order = (1, 0, 2) if multi_class else (1, 0)
    p1 = np.transpose(p1, transpose_order)[order][::-1]
    p2 = np.transpose(p2, transpose_order)[order][::-1]
    if epochwise_version:
        pv1 = [np.transpose(x, transpose_order)[order][::-1] for x in pv1]
        pv2 = [np.transpose(x, transpose_order)[order][::-1] for x in pv2]
    else:
        pv1 = np.transpose(pv1, transpose_order)[order][::-1]
        pv2 = np.transpose(pv2, transpose_order)[order][::-1]
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
        accs_1 = 100 * calc_acc(p1_use, yg_use, multi_class=multi_class)
        accs_2 = 100 * calc_acc(p2_use, yg_use, multi_class=multi_class)

        # Find a threshold on these accuracies that maximizes
        # distinguishing accuracy
        tracc, threshold, rule = find_threshold_acc(
            accs_1, accs_2, granularity=config.granularity)
        adv_accs.append(100 * tracc)
        if epochwise_version:
            accs_victim_1 = [100 * calc_acc(pv1_use_inside, yg_use, multi_class=multi_class)
                             for pv1_use_inside in pv1_use]
            accs_victim_2 = [100 * calc_acc(pv2_use_inside, yg_use, multi_class=multi_class)
                             for pv2_use_inside in pv2_use]
        else:
            accs_victim_1 = 100 * \
                calc_acc(pv1_use, yg_use, multi_class=multi_class)
            accs_victim_2 = 100 * \
                calc_acc(pv2_use, yg_use, multi_class=multi_class)
        if config.multi:
            accs_victim_1 = multi_model_sampling(accs_victim_1, config.multi)
            accs_victim_2 = multi_model_sampling(accs_victim_2, config.multi)
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
            if config.multi2:
                specific_acc = get_threshold_acc_multi(
                    accs_victim_1, accs_victim_2, threshold, config.multi2, rule)
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
    if len(accs_1.shape) == 2:
        # TODO: Implement the rest of this variant
        # For now, we will use loss values for the multi-class case
        # Multi-dimension vase with raw logit values
        # Convert to probabilities, then compute class-wise
        # threshold scores
        combined = np.concatenate((accs_1, accs_2), 0)
        lower, upper = np.min(combined, 0), np.max(combined, 0)
    else:
        # Binary-classification, so values are
        # either actual accuracies, or single logit values
        combined = np.concatenate((accs_1, accs_2))
        lower, upper = np.min(combined), np.max(combined)
    # Want to predict first set as 0s, second set as 1s
    classes = np.concatenate(
        (np.zeros(accs_1.shape[0]), np.ones(accs_2.shape[0])))
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


def get_threshold_acc_multi(X1, X2, threshold, multi2: int, rule=None):
    l1 = X1.shape[0]
    l2 = X2.shape[0]
    Y1 = np.zeros(l1)
    Y2 = np.ones(l2)
    X1_use1 = []
    X2_use1 = []
    X1_use2 = []
    X2_use2 = []
    # For first set of models
    for i in range(l1):
        # For random sample of models
        x = X1[np.random.permutation(l1)[:multi2]]
        # Try both classification rules
        X1_use1.append(np.mean(x >= threshold) >= 0.5)
        X1_use2.append(np.mean(x <= threshold) >= 0.5)
    # For second set of models
    for i in range(l2):
        # For random sample of models
        x = X2[np.random.permutation(l2)[:multi2]]
        # Try both classification rules
        X2_use1.append(np.mean(x >= threshold) >= 0.5)
        X2_use2.append(np.mean(x <= threshold) >= 0.5)
    Y = np.concatenate((Y1, Y2))
    X1_use1 = np.array(X1_use1)
    X1_use2 = np.array(X1_use2)
    X2_use1 = np.array(X2_use1)
    X2_use2 = np.array(X2_use2)
    # Concatenate predictions from both sets of models
    M1 = np.concatenate((X1_use1, X2_use1))
    M2 = np.concatenate((X1_use2, X2_use2))
    # Rule-1: everything above threshold is 1 class
    acc_1 = np.mean(M1 == Y)
    # Rule-2: everything below threshold is 1 class
    acc_2 = np.mean(M2 == Y)

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
    ground_truth = np.concatenate(
        (np.zeros(pred_1.shape[1]), np.ones(pred_2.shape[1])))
    acc, _ = get_threshold_pred(predictions_combined, ground_truth, thres, rules)
    return acc, thres, rules


def get_threshold_pred(X, Y, threshold, rule,
                       get_pred: bool = False,
                       tune_final_threshold: Union[bool, float] = False):
    """
        Get distinguishing accuracy between distributions, given predictions
        for models on datapoints, and thresholds with prediction rules.
        Args:
            X: predictions for models on datapoints
            Y: ground truth for datapoints
            threshold: thresholds for prediction rules
            rule: prediction rules
            get_pred: whether to return predictions
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
        # Store direct average P[distribution=1]
        res.append(prob)
    res = np.array(res)

    # Default threshold value- 0.5
    final_threshold = 0.5
    if type(tune_final_threshold) != bool:
        # Use provided threshold
        final_threshold = tune_final_threshold
    elif tune_final_threshold is True:
        # Requested generation of threshold
        final_threshold = find_max_acc_threshold(res, Y)
    acc = np.mean((res >= final_threshold) == Y)

    # Return predictions, if requested
    if get_pred:
        return res, acc, final_threshold
    return acc, final_threshold


def get_threshold_pred_multi(
        X1, X2, threshold,
        rule, multi2: int,
        get_pred: bool = False):
    # X Shape: (n_samples, n_models)
    # Y Shape: (n_models)
    # threshold shape: (n_samples)
    res = []
    l1 = X1.shape[1]
    l2 = X2.shape[1]
    Y1 = np.zeros(l1)
    Y2 = np.ones(l2)
    res1 = []
    res2 = []
    r1 = []
    r2 = []
    # For each model
    for i in range(X1.shape[1]):
        # Compute expected P[distribution=1] using given data, threshold, rules
        prob = np.average((X1[:, i] <= threshold) == rule)

        # If majority (>0.5) points indicate some distribution,
        # it must be that one indeed
        res1.append(prob >= 0.5)
    for i in range(X2.shape[1]):
        # Compute expected P[distribution=1] using given data, threshold, rules
        prob = np.average((X2[:, i] <= threshold) == rule)

        # If majority (>0.5) points indicate some distribution,
        # it must be that one indeed
        res2.append(prob >= 0.5)
    res1 = np.array(res1)
    res2 = np.array(res2)
    for i in range(l1):
        x = res1[np.random.permutation(l1)[:multi2]]
        r1.append(np.mean(x) >= 0.5)
    for i in range(l1):
        x = res2[np.random.permutation(l2)[:multi2]]
        r2.append(np.mean(x) >= 0.5)
    r1 = np.array(r1)
    r2 = np.array(r2)
    acc = (np.mean(r1 == Y1)+np.mean(r2 == Y2))/2
    res = np.concatenate((r1, r2))
    # Return predictions, if requested
    if get_pred:
        return res, acc
    return acc


def order_points(p1s, p2s):
    """
        Estimate utility of individual points, done by taking
        absolute difference in their predictions.
    """
    if p1s.shape != p2s.shape:
        raise ValueError(
            f"Both predictions should be same shape, got {p1s.shape} and {p2s.shape}")
    # Simple binary-classification case
    abs_diff = np.absolute(np.sum(p1s, axis=0) - np.sum(p2s, axis=0))
    # TODO: Is this really the best way to rank points?
    if len(p1s.shape) == 3:
        # Handle multi-class case
        abs_diff = np.mean(abs_diff, 1)

    inds = np.argsort(abs_diff)
    return inds

def epoch_order_p(p11,p12,p21,p22):
        #pij: ith epoch, jth distribution
        #TODO: find a better order rather than only using the last epoch
        return order_points(p21,p22)[::-1]
def find_max_acc_threshold(preds, labels, granularity=0.01):
    """
        Find the threshold that maximizes accuracy.
        Assume preds are in [0, 1]
    """
    best_threshold = 0
    best_acc = 0
    qualifying_thresholds_list = []
    # Should pick the threshold with maximum gap between predictions
    # Not just the one that provides best accuracy
    for threshold in np.arange(0.0, 1.0, granularity):
        y_pred = (preds >= threshold).astype(int)
        acc = np.mean(y_pred == labels)
        if acc > best_acc:
            qualifying_thresholds_list = []
            best_acc = acc
            best_threshold = threshold
        if acc == best_acc:
            qualifying_thresholds_list.append(threshold)
    # Of all thresholds that have 'best' accuracy, pick median
    # Most likely to be robust to changes in predictions
    best_threshold = np.median(qualifying_thresholds_list)
    return best_threshold
