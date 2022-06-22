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
               epochwise_version: bool = False):
        """
            Perform Threshold-Test and Loss-Test attacks using
            given accuracies of models.
        """
        assert calc_acc is not None, "Must provide function to compute accuracy"
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"

        # Get values using data from first distribution
        vic_1_with_1, vic_2_with_1 = get_kl_preds(
            preds_adv.preds_on_distr_1.preds_property_1,
            preds_adv.preds_on_distr_1.preds_property_2,
            preds_vic.preds_on_distr_1.preds_property_1,
            preds_vic.preds_on_distr_1.preds_property_2,)
        # Get values using data from second distribution
        vic_1_with_2, vic_2_with_2 = get_kl_preds(
            preds_adv.preds_on_distr_2.preds_property_1,
            preds_adv.preds_on_distr_2.preds_property_2,
            preds_vic.preds_on_distr_2.preds_property_1,
            preds_vic.preds_on_distr_2.preds_property_2,)

        # Combine data
        KL_for_1_with_1 = np.concatenate((vic_1_with_1[0], vic_1_with_2[0]))
        KL_for_1_with_2 = np.concatenate((vic_1_with_1[1], vic_1_with_2[1]))
        KL_for_2_with_1 = np.concatenate((vic_2_with_1[0], vic_2_with_2[0]))
        KL_for_2_with_2 = np.concatenate((vic_2_with_1[1], vic_2_with_2[1]))

        # Get predictions corresponding to the KL values
        preds_first = np.mean(KL_for_1_with_1 > KL_for_1_with_2, 1)
        preds_second = np.mean(KL_for_2_with_1 > KL_for_2_with_2, 1)
        preds = np.concatenate((preds_first, preds_second))
        gt = np.concatenate((np.zeros_like(preds_first),
                            np.ones_like(preds_second)))
        acc = np.mean((preds >= 0.5) == gt)

        # No concept of "choice"
        choice_information = (None, None)
        return [(acc, preds), (None, None), choice_information]


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def KL(x, y, multi_class: bool = False):
    if multi_class:
        raise NotImplementedError("Not implemented multi-class model yet")
    else:
        # Get preds for other class as well
        x_, y_ = 1 - x, 1 - y
        first_term = x * (np.log(x) - np.log(y))
        second_term = y_ * (np.log(x_) - np.log(y_))
    return np.mean(first_term + second_term, 1)


def get_kl_preds(adv_1_preds, adv_2_preds, vic_1_preds, vic_2_preds):
    # Apply sigmoid on all of them
    p1, p2 = sigmoid(adv_1_preds), sigmoid(adv_2_preds)
    pv1, pv2 = sigmoid(vic_1_preds), sigmoid(vic_2_preds)

    # Compare the KL divergence between the two distributions
    # For both sets of victim models
    KL_for_1_with_1 = np.array([KL(p1, x) for x in pv1])
    KL_for_1_with_2 = np.array([KL(p2, x) for x in pv1])
    KL_for_2_with_1 = np.array([KL(p1, x) for x in pv2])
    KL_for_2_with_2 = np.array([KL(p2, x) for x in pv2])

    # Compare KL values
    return (KL_for_1_with_1, KL_for_1_with_2),  (KL_for_2_with_1, KL_for_2_with_2)
