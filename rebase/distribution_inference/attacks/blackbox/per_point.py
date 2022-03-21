import numpy as np
from typing import List, Tuple, Callable

from distribution_inference.attacks.blackbox.core import Attack, find_threshold_pred, get_threshold_pred, order_points, PredictionsOnOneDistribution, PredictionsOnDistributions
from distribution_inference.config import BlackBoxAttackConfig


class PerPointThresholdAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None):
        """
        Take predictions from both distributions and run attacks.
        Pick the one that works best on adversary's models
        """
        # Get data for first distribution
        adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1 = perpoint_threshold_test_per_dist(
            preds_adv.preds_on_distr_1,
            preds_vic.preds_on_distr_1,
            self.config)
        # Get data for second distribution
        adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2 = perpoint_threshold_test_per_dist(
            preds_adv.preds_on_distr_2,
            preds_vic.preds_on_distr_2,
            self.config)

        # Get best adv accuracies for both distributions and compare
        chosen_distribution = 0
        if np.max(adv_accs_1) > np.max(adv_accs_2):
            adv_accs_use, adv_preds_use, victim_accs_use, victim_preds_use = adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1
        else:
            adv_accs_use, adv_preds_use, victim_accs_use, victim_preds_use = adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2
            chosen_distribution = 1

        # Out of the best distribution, pick best ratio according to accuracy on adversary's models
        chosen_ratio_index = np.argmax(adv_accs_use)
        victim_acc_use = victim_accs_use[chosen_ratio_index]
        victim_pred_use = victim_preds_use[chosen_ratio_index]
        adv_acc_use = adv_accs_use[chosen_ratio_index]
        adv_pred_use = adv_preds_use[chosen_ratio_index]

        choice_information = (chosen_distribution, chosen_ratio_index)
        return (victim_acc_use, victim_pred_use), (adv_acc_use, adv_pred_use), choice_information


def _perpoint_threshold_on_ratio(preds_1, preds_2, classes, threshold, rule):
    """
        Run perpoint threshold test (confidence)
        for a given "quartile" ratio
    """
    # Combine predictions into one vector
    combined = np.concatenate((preds_1, preds_2), axis=1)

    # Compute accuracy for given predictions, thresholds, and rules
    preds, acc = get_threshold_pred(
        combined, classes, threshold, rule, get_pred=True,
        confidence=True)

    return 100 * acc, preds


def perpoint_threshold_test_per_dist(preds_adv: PredictionsOnOneDistribution,
                                     preds_victim: PredictionsOnOneDistribution,
                                     config: BlackBoxAttackConfig):
    """
        Compute thresholds (based on probabilities) for each given datapoint,
        search for thresholds using given adv model's predictions.
        Compute accuracy and predictions using given data and predictions
        on victim model's predictions.
        Try this out with different values of "quartiles", where points
        are ranked according to some utility estimate.
    """
    # Predictions by adversary's models
    p1, p2 = preds_adv.preds_property_1, preds_adv.preds_property_2
    # Predictions by victim's models
    pv1, pv2 = preds_victim.preds_property_1, preds_victim.preds_property_2

    # Optimal order of point
    order = order_points(p1, p2)

    # Order points according to computed utility
    p1 = np.transpose(p1)[order][::-1]
    p2 = np.transpose(p2)[order][::-1]
    pv1 = np.transpose(pv1)[order][::-1]
    pv2 = np.transpose(pv2)[order][::-1]

    # Get thresholds for all points
    _, thres, rs = find_threshold_pred(p1, p2, granularity=config.granularity)

    # Ground truth
    classes_adv = np.concatenate(
        (np.zeros(p1.shape[1]), np.ones(p2.shape[1])))
    classes_victim = np.concatenate(
        (np.zeros(pv1.shape[1]), np.ones(pv2.shape[1])))

    adv_accs, victim_accs, victim_preds, adv_preds = [], [], [], []
    for ratio in config.ratios:
        # Get first <ratio> percentile of points
        leng = int(ratio * p1.shape[0])
        p1_use, p2_use = p1[:leng], p2[:leng]
        pv1_use, pv2_use = pv1[:leng], pv2[:leng]
        thres_use, rs_use = thres[:leng], rs[:leng]

        # Compute accuracy for given data size on adversary's models
        adv_acc, adv_pred = _perpoint_threshold_on_ratio(
            p1_use, p2_use, classes_adv, thres_use, rs_use)
        adv_accs.append(adv_acc)
        # Compute accuracy for given data size on victim's models
        victim_acc, victim_pred = _perpoint_threshold_on_ratio(
            pv1_use, pv2_use, classes_victim, thres_use, rs_use)
        victim_accs.append(victim_acc)
        # Keep track of predictions on victim's models
        victim_preds.append(victim_pred)
        adv_preds.append(adv_pred)

    adv_accs = np.array(adv_accs)
    victim_accs = np.array(victim_accs)
    victim_preds = np.array(victim_preds, dtype=object)
    adv_preds = np.array(adv_preds, dtype=object)
    return adv_accs, adv_preds, victim_accs, victim_preds
