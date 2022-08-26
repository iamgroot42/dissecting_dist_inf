import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions


class KLAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False):
        """
            Perform Threshold-Test and Loss-Test attacks using
            given accuracies of models.
        """
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        self.not_using_logits = not_using_logits

        # Get values using data from first distribution
        preds_1_first, preds_1_second = self._get_kl_preds(
            preds_adv.preds_on_distr_1.preds_property_1,
            preds_adv.preds_on_distr_1.preds_property_2,
            preds_vic.preds_on_distr_1.preds_property_1,
            preds_vic.preds_on_distr_1.preds_property_2)
        # Get values using data from second distribution
        preds_2_first, preds_2_second = self._get_kl_preds(
            preds_adv.preds_on_distr_2.preds_property_1,
            preds_adv.preds_on_distr_2.preds_property_2,
            preds_vic.preds_on_distr_2.preds_property_1,
            preds_vic.preds_on_distr_2.preds_property_2)

        # Combine data
        preds_first = np.concatenate((preds_1_first, preds_2_first), 1)
        preds_second = np.concatenate((preds_1_second, preds_2_second), 1)
        preds = np.concatenate((preds_first, preds_second))

        if not self.config.kl_voting:
            preds -= np.min(preds, 0)
            preds /= np.max(preds, 0)

        preds = np.mean(preds, 1)        
        gt = np.concatenate((np.zeros(preds_first.shape[0]), np.ones(preds_second.shape[0])))
        acc = 100 * np.mean((preds >= 0.5) == gt)

        # No concept of "choice" (are we in the Matrix :P)
        choice_information = (None, None)
        return [(acc, preds), (None, None), choice_information]
    
    def _get_kl_preds(self, ka, kb, kc1, kc2):
        # Apply sigmoid to ones that are not already sigmoided
        ka_, kb_ = ka, kb
        kc1_, kc2_ = kc1, kc2
        if not self.not_using_logits:
            ka_, kb_ = sigmoid(ka), sigmoid(kb)
            kc1_, kc2_ = sigmoid(kc1), sigmoid(kc2)

        # Consider all unique pairs of models
        xx, yy = np.triu_indices(ka.shape[0], k=1)
        # Randomly pick pairs of models
        random_pick = np.random.permutation(
            xx.shape[0])[:int(self.config.kl_frac * xx.shape[0])]
        xx, yy = xx[random_pick], yy[random_pick]

        # Compare the KL divergence between the two distributions
        # For both sets of victim models
        KL_vals_1_a = np.array([KL(ka_, x,
            multi_class=self.config.multi_class) for x in kc1_])
        KL_vals_1_b = np.array(
            [KL(kb_, x, multi_class=self.config.multi_class) for x in kc1_])
        KL_vals_2_a = np.array([KL(ka_, x,
            multi_class=self.config.multi_class) for x in kc2_])
        KL_vals_2_b = np.array([KL(kb_, x,
            multi_class=self.config.multi_class) for x in kc2_])

        preds_first = self._pairwise_compare(
            KL_vals_1_a, KL_vals_1_b, xx, yy)
        preds_second = self._pairwise_compare(
            KL_vals_2_a, KL_vals_2_b, xx, yy)

        # Compare KL values
        return preds_first, preds_second
    
    def _pairwise_compare(self, x, y, xx, yy):
        x_ = np.expand_dims(x, 2)
        y_ = np.expand_dims(y, 2)
        y_ = np.transpose(y_, (0, 2, 1))
        if self.config.kl_voting:
            pairwise_comparisons = (x_ > y_)
        else:
            pairwise_comparisons = (x_ - y_)
        preds = np.array([z[xx, yy] for z in pairwise_comparisons])
        return preds
"""
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False):
        
            Perform Threshold-Test and Loss-Test attacks using
            given accuracies of models.
        
        #assert calc_acc is not None, "Must provide function to compute accuracy"
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        frac = 0.3
        # Get values using data from first distribution
        preds_1_first, preds_1_second = get_kl_preds(
            preds_adv.preds_on_distr_1.preds_property_1,
            preds_adv.preds_on_distr_1.preds_property_2,
            preds_vic.preds_on_distr_1.preds_property_1,
            preds_vic.preds_on_distr_1.preds_property_2,
            frac=frac)
        # Get values using data from second distribution
        preds_2_first, preds_2_second = get_kl_preds(
            preds_adv.preds_on_distr_2.preds_property_1,
            preds_adv.preds_on_distr_2.preds_property_2,
            preds_vic.preds_on_distr_2.preds_property_1,
            preds_vic.preds_on_distr_2.preds_property_2,
            frac=frac)

        # Combine data
        preds_first = np.concatenate((preds_1_first, preds_2_first), 1)
        preds_second = np.concatenate((preds_1_second, preds_2_second), 1)

        # Get predictions (voting)
        preds_first = np.mean(preds_first, 1)
        preds_second = np.mean(preds_second, 1)
        preds = np.concatenate((preds_first, preds_second))
        
        gt = np.concatenate((np.zeros_like(preds_first),
                            np.ones_like(preds_second)))
        acc = np.mean((preds >= 0.5) == gt)

        # No concept of "choice" (are we in the Matrix :P)
        choice_information = (None, None)
        return [(acc, preds), (None, None), choice_information]
    """

def sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)


def KL(x, y, multi_class: bool = False):
    small_eps = 1e-6
    x_ = np.clip(x, small_eps, 1 - small_eps)
    y_ = np.clip(y, small_eps, 1 - small_eps)
    if multi_class:
        return np.mean(np.sum(x * (np.log(x) - np.log(y)),axis=2),axis=1)
    else:
        # Strategy 1: Add (or subtract) small noise to avoid NaNs/INFs
        # Get preds for other class as well
        x_, y_ = 1 - x, 1 - y
        first_term = x * (np.log(x) - np.log(y))
        second_term = x_ * (np.log(x_) - np.log(y_))
        return np.mean(first_term + second_term, 1)
