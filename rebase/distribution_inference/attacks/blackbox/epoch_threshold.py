import numpy as np
from typing import Callable

from distribution_inference.attacks.blackbox.core import Attack, epoch_order_p, PredictionsOnDistributions, PredictionsOnOneDistribution, find_threshold_acc, get_threshold_acc, order_points
from distribution_inference.attacks.blackbox.core import _acc_per_dis
DUMPING = 10


class Epoch_ThresholdAttack(Attack):
    def attack(self,
               preds_vic1: PredictionsOnDistributions,
               preds_vic2: PredictionsOnDistributions,  # epoch 2
               preds_adv1: PredictionsOnDistributions,
               preds_adv2: PredictionsOnDistributions,
               ground_truth,
               calc_acc: Callable,
               get_preds: bool = False,
               ratio: bool = False):
        assert calc_acc is not None, "Must provide function to compute accuracy"
        assert ground_truth is not None, "Must provide ground truth to compute accuracy"
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        self.ratio = ratio
        adv_accs_1, victim_accs_1 = self._thresh_per_dis(
            (preds_vic1.preds_on_distr_1, preds_vic2.preds_on_distr_1),
            (preds_adv1.preds_on_distr_1, preds_adv2.preds_on_distr_1),
            ground_truth[0],
            calc_acc,
        )
        # Get accuracies on second data distribution
        adv_accs_2, victim_accs_2 = self._thresh_per_dis(
            (preds_vic1.preds_on_distr_2, preds_vic2.preds_on_distr_2),
            (preds_adv1.preds_on_distr_2, preds_adv2.preds_on_distr_2),
            ground_truth[1],
            calc_acc,
        )
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
        choice_information = (chosen_distribution, chosen_ratio_index)
        return ((victim_acc_use, None), (adv_accs_use[chosen_ratio_index], None), choice_information)

    def _thresh_per_dis(self, preds_vic,
                        preds_adv,
                        ground_truth,
                        calc_acc: Callable,
                        get_preds: bool = False):  # TODO: adds get_preds
        #preds_vic: (preds_vic1.preds_on_distr_1,preds_vic2.preds_on_distr_1)
        order = epoch_order_p(preds_adv[0].preds_property_1,
                              preds_adv[0].preds_property_2,
                              preds_adv[1].preds_property_1,
                              preds_adv[1].preds_property_2)
        adv_accs, f_accs = [], []
        #pij: ith epoch, jth distribution
        p11, p12, p21, p22, y = np.transpose(preds_adv[0].preds_property_1)[order], np.transpose(preds_adv[0].preds_property_2)[
            order], np.transpose(preds_adv[1].preds_property_1)[order], np.transpose(preds_adv[1].preds_property_2)[order], ground_truth[order]
        pv11, pv12, pv21, pv22 = np.transpose(preds_vic[0].preds_property_1)[order], np.transpose(preds_vic[0].preds_property_2)[
            order], np.transpose(preds_vic[1].preds_property_1)[order], np.transpose(preds_vic[1].preds_property_2)[order]
        for ratio in self.config.ratios:
            leng = int(ratio*len(order))
            advacc = _acc_per_dis(PredictionsOnOneDistribution(p11[:leng], p12[:leng]),
                                  PredictionsOnOneDistribution(
                                      p21[:leng], p22[:leng]),
                                  y[:leng],
                                  calc_acc,
                                  t=True)
            vacc = _acc_per_dis(PredictionsOnOneDistribution(pv11[:leng], pv12[:leng]),
                                PredictionsOnOneDistribution(
                                    pv21[:leng], pv22[:leng]),
                                y[:leng],
                                calc_acc,
                                t=True)
            if self.ratio:
                vdif1 = (vacc[1][0]+DUMPING)/(vacc[0][0]+DUMPING)
                vdif2 = (vacc[1][1]+DUMPING)/(vacc[0][1]+DUMPING)
                adif1 = (advacc[1][0]+DUMPING)/(advacc[0][0]+DUMPING)
                adif2 = (advacc[1][1]+DUMPING)/(advacc[0][1]+DUMPING)
            else:
                vdif1 = vacc[1][0]-vacc[0][0]
                vdif2 = vacc[1][1]-vacc[0][1]
                adif1 = advacc[1][0]-advacc[0][0]
                adif2 = advacc[1][1]-advacc[0][1]
            tracc, threshold, rule = find_threshold_acc(
                adif1, adif2, granularity=self.config.granularity)
            combined = np.concatenate((vdif1, vdif2))
            classes = np.concatenate(
                (np.zeros_like(vdif1), np.ones_like(vdif2)))
            specific_acc = get_threshold_acc(
                combined, classes, threshold, rule)
            adv_accs.append(100 * tracc)
            f_accs.append(100 * specific_acc)

        return np.array(adv_accs), np.array(f_accs)
