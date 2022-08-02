import numpy as np
import torch as ch
from typing import List, Tuple, Callable, Union

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnOneDistribution, PredictionsOnDistributions,multi_model_sampling
from distribution_inference.config import BlackBoxAttackConfig


class PerPointChooseAttack(Attack):

    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False):
        """
        Take predictions from both distributions and run attacks.
        Pick the one that works best on adversary's models
        """
        # For the multi-class case, we do not want to work with direct logit values
        # Scale them to get post-softmax probabilities
        # TODO: Actually do that

        # Get data for first distribution
        adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1, rs_1 = perpoint_threshold_test_per_dist(
            preds_adv.preds_on_distr_1,
            preds_vic.preds_on_distr_1,
            self.config,
            epochwise_version=epochwise_version,
            ground_truth=ground_truth[0])
        # Get data for second distribution
        adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2, rs_2 = perpoint_threshold_test_per_dist(
            preds_adv.preds_on_distr_2,
            preds_vic.preds_on_distr_2,
            self.config,
            epochwise_version=epochwise_version,
            ground_truth=ground_truth[1])

        # Get best adv accuracies for both distributions and compare
        chosen_distribution = 0
        if np.max(adv_accs_1) > np.max(adv_accs_2):
            adv_accs_use, adv_preds_use = adv_accs_1, adv_preds_1
            victim_accs_use, victim_preds_use = victim_accs_1, victim_preds_1
            final_rs_use = rs_1
        else:
            adv_accs_use, adv_preds_use = adv_accs_2, adv_preds_2
            victim_accs_use, victim_preds_use = victim_accs_2, victim_preds_2
            final_rs_use = rs_2
            chosen_distribution = 1
          
        # Out of the best distribution, pick best ratio according to accuracy on adversary's models
        
        choice_information = (chosen_distribution,  final_rs_use)
        return [(victim_accs_use, victim_preds_use), (adv_accs_use, adv_preds_use), choice_information]

    def wrap_preds_to_save(self, result: List):
        victim_preds = result[0][1]
        adv_preds = result[1][1]
        save_dic = {'victim_preds': victim_preds, 'adv_preds': adv_preds}
        return save_dic

def pair_order(leng:int):
    """
    Get different pairs
    """
    order_set = set()
    while len(order_set)<leng:
        order_set.add(frozenset(np.random.choice(leng,size=2,replace=False)))
    order_set = [np.array(list(x)) for x in order_set]
    return np.array(order_set)

def order_pairs(p,order):
    """
    Simple wrapper to get a list of pairs
    """
    return np.array([p[o] for o in order])

def find_rules(p1,p2,order):
    p1o = order_pairs(p1,order)
    p2o = order_pairs(p2,order)
    r1 = np.array([np.average(p[0,:]>=p[1,:])>=0.5 for p in p1o])
    r2 = np.array([np.average(p[0,:]<p[1,:])>=0.5 for p in p2o])
    
    #discard points that have same rules for both distributions
    rs = r1==r2
    assert len(rs)>len(r1)/10, "Not enough useful pairs"
    ord = order[rs]
    rs = r1[rs] 
    """1 in rs means that for in the pair, value for first point being larger
     than the one of the second indicates first distribution, and otherwise
    """
    preds, acc = acc_rule(p1,p2,ord,rs)
    return rs, ord,acc,preds

def acc_rule(p1,p2,order,rule):
    p1 = order_pairs(p1,order)
    p2 = order_pairs(p2,order)
    r1 =  np.repeat(np.expand_dims(rule,1),p1.shape[2],axis=1)
    r2 = np.repeat(np.expand_dims(rule,1),p2.shape[2],axis=1)
    preds1 =np.array( [p[0,:]>=p[1,:] for p in p1])
    preds1 =  preds1==r1
    preds1 = np.average(preds1,axis=0)
    preds2 =np.array( [p[0,:]<p[1,:] for p in p2])
    preds2 =  preds2==r2
    preds2 = np.average(preds2,axis=0)
    preds = np.hstack((1-preds1,preds2))
    accs = (np.average(preds1)+np.average(preds2))/2
    return preds,accs




def perpoint_threshold_test_per_dist(
        preds_adv: PredictionsOnOneDistribution,
        preds_victim: PredictionsOnOneDistribution,
        config: BlackBoxAttackConfig,
        epochwise_version: bool = False,
        ground_truth: Tuple[List, List] = None):
    """
        Compute the rule (based on probabilities) for each pair of datapoints by adv models.
        Compute accuracy and predictions using given data and predictions
        on victim model's predictions.
        If preds_victim is None, computes metrics and datapoints
        only for the adversary.
    """
    assert not epochwise_version, "No implmentation for epochwise"
    assert not config.multi, "No implementation for multi model"
    assert not config.multi2, "No implementation for multi model"
    assert not config.multi_class, "No impplementation for multi class"
    victim_preds_present = (preds_victim is not None)

    # Predictions by adversary's models
    p1 = preds_adv.preds_property_1.copy()
    p2 = preds_adv.preds_property_2.copy()

    if victim_preds_present:
        # Predictions by victim's models
        pv1 = preds_victim.preds_property_1.copy()
        pv2 = preds_victim.preds_property_2.copy()

    if config.loss_variant:
        # Use loss values instead of raw logits (or prediction probabilities)
        assert ground_truth is not None, "Need ground-truth for loss_variant setting"
        p1 = np_compute_losses(p1, ground_truth, multi_class=False)
        p2 = np_compute_losses(p2, ground_truth, multi_class=False)
        if victim_preds_present:
            pv1 = np_compute_losses(pv1, ground_truth, multi_class=False)
            pv2 = np_compute_losses(pv2, ground_truth, multi_class=False)

    
    order = np.random.permutation(p1.shape[1])
    # Order points according to computed utility
    transpose_order = (1, 0, 2) if config.multi_class else (1, 0)
    p1 = np.transpose(p1, transpose_order)[order][::-1]
    p2 = np.transpose(p2, transpose_order)[order][::-1]

    if victim_preds_present:
        
        pv1 = np.transpose(pv1, transpose_order)[order][::-1]
        pv2 = np.transpose(pv2, transpose_order)[order][::-1]
        if config.multi:
            pv1 = multi_model_sampling(pv1, config.multi)
            pv2 = multi_model_sampling(pv2, config.multi)

    if config.multi_class:
        # If multi-class, replace predictions with loss values
        assert ground_truth is not None, "Need ground-truth for multi-class setting"
        y_gt = ground_truth[order][::-1]
        p1, p2 = np_compute_losses(p1, y_gt), np_compute_losses(p2, y_gt)
        if victim_preds_present:
            pv1, pv2 = np_compute_losses(pv1, y_gt), np_compute_losses(pv2, y_gt)

    # Scale thresholds with mean/std across data, if variant selected
    if config.relative_threshold:
        reduce_indices = (1, 2) if config.multi_class else 1
        mean1, std1 = np.mean(p1, reduce_indices, keepdims=True), np.std(
            p1, reduce_indices, keepdims=True)
        mean2, std2 = np.mean(p2, reduce_indices, keepdims=True), np.std(
            p2, reduce_indices, keepdims=True)
        # Scale prediction values with mean/std
        p1_use = (p1 - mean1) / std1
        p2_use = (p2 - mean2) / std2
        reduce_indices = (1, 2) if config.multi_class else 1
        mean1, std1 = np.mean(p1_use, reduce_indices, keepdims=True), np.std(
                p1_use, reduce_indices, keepdims=True)
        mean2, std2 = np.mean(p2_use, reduce_indices, keepdims=True), np.std(
                p2_use, reduce_indices, keepdims=True)
            # Scale prediction values with mean/std
        p1_use = (p1_use - mean1) / std1
        p2_use = (p2_use - mean2) / std2
        if victim_preds_present:
            mean1, std1 = np.mean(pv1_use, reduce_indices, keepdims=True), np.std(
                    pv1_use, reduce_indices, keepdims=True)
            mean2, std2 = np.mean(pv2_use, reduce_indices, keepdims=True), np.std(
                    pv2_use, reduce_indices, keepdims=True)
            pv1_use = (pv1 - mean1) / std1
            pv2_use = (pv2 - mean2) / std2

    else:

        p1_use, p2_use = p1, p2
        if victim_preds_present:
            pv1_use = pv1
            pv2_use = pv2

    ord = pair_order(p1.shape[0])
    rs,ord,adv_acc,adv_preds = find_rules(p1_use,p2_use,ord)
    vic_preds,vic_acc = acc_rule(pv1_use,pv2_use,ord,rs)
    return adv_acc, adv_preds, vic_acc, vic_preds, rs


def np_compute_losses(preds: np.ndarray,
                      labels: np.ndarray,
                      multi_class: bool = True):
    """
        Convert to PyTorch tensors, compute crossentropyloss
        Convert back to numpy arrays, return.
    """
    # Convert to PyTorch tensors
    preds_ch = ch.from_numpy(preds.copy())
    labels_ch = ch.from_numpy(labels.copy())

    if multi_class:
        loss = ch.nn.CrossEntropyLoss(reduction='none')
        labels_ch = labels_ch.long()
        preds_ch = preds_ch.transpose(0, 1)
    else:
        loss = ch.nn.BCEWithLogitsLoss(reduction='none')
        labels_ch = labels_ch.float()

    loss_vals = [loss(pred_ch, labels_ch).numpy() for pred_ch in preds_ch]
    loss_vals = np.array(loss_vals)
    if multi_class:
        loss_vals = loss_vals.T
    return loss_vals
