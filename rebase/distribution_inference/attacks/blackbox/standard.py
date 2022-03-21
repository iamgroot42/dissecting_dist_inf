import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, threshold_test_per_dist, PredictionsOnDistributions


class LossAndThresholdAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None):
        """
            Perform Threshold-Test and Loss-Test attacks using
            given accuracies of models.
        """
        assert calc_acc is not None, "Must provide function to compute accuracy"
        assert ground_truth is not None, "Must provide ground truth to compute accuracy"

        # Get accuracies on first data distribution
        adv_accs_1, victim_accs_1, acc_1 = threshold_test_per_dist(
            calc_acc,
            preds_adv.preds_on_distr_1,
            preds_vic.preds_on_distr_1,
            ground_truth[0], self.config)
        # Get accuracies on second data distribution
        adv_accs_2, victim_accs_2, acc_2 = threshold_test_per_dist(
            calc_acc,
            preds_adv.preds_on_distr_2,
            preds_vic.preds_on_distr_2,
            ground_truth[1], self.config)

        # Get best adv accuracies for both distributions, across all ratios
        chosen_distribution = 0
        if np.max(adv_accs_1) > np.max(adv_accs_2):
            adv_accs_use, victim_accs_use = adv_accs_1, victim_accs_1
        else:
            adv_accs_use, victim_accs_use = adv_accs_2, victim_accs_2
            chosen_distribution = 1

        # Of the chosen distribution, pick the one with the best accuracy
        # out of all given ratios
        chosen_ratio_index = np.argmax(adv_accs_use)
        victim_acc_use = victim_accs_use[chosen_ratio_index]
        # Loss test
        basic = self._loss_test(acc_1, acc_2, self.config)

        choice_information = (chosen_distribution, chosen_ratio_index)
        return (victim_acc_use, basic[chosen_ratio_index]), choice_information

    def _loss_test(self, acc_1, acc_2):
        basic = []
        for r in range(len(self.config.ratios)):
            preds_1 = (acc_1[0][r, :] > acc_2[0][r, :])
            preds_2 = (acc_1[1][r, :] <= acc_2[1][r, :])
            basic.append(100*(np.mean(preds_1) + np.mean(preds_2)) / 2)
        return basic
