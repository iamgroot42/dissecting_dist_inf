import numpy as np
from scipy.stats import entropy
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, threshold_test_per_dist, PredictionsOnDistributions,PredictionsOnOneDistribution,order_points
from distribution_inference.config import BlackBoxAttackConfig

class KLAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False):
        """
            Perform Threshold-Test and Loss-Test attacks using
            given accuracies of models.
        """
        assert calc_acc is not None, "Must provide function to compute accuracy"
        assert ground_truth is not None, "Must provide ground truth to compute accuracy"
        assert not (self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        # Get accuracies on first data distribution
        acc_1,preds_1 = KL_test_per_dist(preds_adv.preds_on_distr_1,preds_vic.preds_on_distr_1,self.config,epochwise_version)
        acc_2,preds_2 = KL_test_per_dist(preds_adv.preds_on_distr_2,preds_vic.preds_on_distr_2,self.config,epochwise_version)
        # Get best adv accuracies for both distributions, across all ratios
        chosen_distribution = 0
        if acc_1>acc_2:
            acc_use,preds_use = acc_1,preds_1
        else:
            acc_use,preds_use = acc_2,preds_2
            chosen_distribution = 1

        # Of the chosen distribution, pick the one with the best accuracy
        # out of all given ratios
        
        choice_information = (chosen_distribution, None)
        return [[(acc_use,preds_use)], (None,None), choice_information]

def KL_test_per_dist(preds_adv: PredictionsOnOneDistribution,
                     preds_victim: PredictionsOnOneDistribution,
                     config: BlackBoxAttackConfig,
                     epochwise_version: bool = False,
                     KL_func: Callable=entropy):
    """
        Perform threshold-test on predictions of adversarial and victim models,
        for each of the given ratios. Returns statistics on all ratios.
    """
    assert not (
        epochwise_version and config.multi), "No implementation for both epochwise and multi model"
    assert not (config.multi2 and config.multi), "No implementation for both multi model"
    assert not (
        epochwise_version and config.multi2), "No implementation for both epochwise and multi model"
    assert not config.multi_class, "No implementation for multi class"
    assert not epochwise_version, "No implememtation for epochwise"
    # Predictions made by the adversary's models
    p1, p2 = preds_adv.preds_property_1, preds_adv.preds_property_2
    # Predictions made by the  victim's models
    pv1, pv2 = preds_victim.preds_property_1, preds_victim.preds_property_2
    KL1 = (np.array([np.average([KL_func(p1_,pv1_) for p1_ in p1]) for pv1_ in pv1]),
    np.array([np.average([KL_func(p2_,pv1_) for p2_ in p2]) for pv1_ in pv1]))
    KL2 =  (np.array([np.average([KL_func(p1_,pv2_) for p1_ in p1]) for pv2_ in pv2]),
    np.array([np.average([KL_func(p2_,pv2_) for p2_ in p2]) for pv2_ in pv2]))
    res1 = KL1[1] - KL1[0]
    res2 = KL2[0] - KL2[1]
    acc1 = np.average(res1>=0)
    acc2 = np.average(res2>=0)
    
    return 100*(acc1+acc2)/2, np.hstack((res1,res2))
