import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistribution
from distribution_inference.attacks.blackbox.core import _acc_per_dis


class Epoch_LossAttack(Attack):
    def attack(self,
               preds_vic1: PredictionsOnDistributions,
               preds_vic2: PredictionsOnDistributions,
               ground_truth,
               calc_acc: Callable,
               get_preds:bool=False):
        acc_1 = _acc_per_dis(preds_vic1.preds_on_distr_1,
                            preds_vic2.preds_on_distr_1,
                            ground_truth[0],
                            calc_acc)
        acc_2 = _acc_per_dis(preds_vic1.preds_on_distr_2,
                            preds_vic2.preds_on_distr_2,
                            ground_truth[1],
                            calc_acc)
        return self._loss_test(acc_1,acc_2,get_preds)

    def _loss_test(self,acc_1,acc_2,get_preds:bool=False):
        #acc_1 is a tupple of list of accs on first distribution, tuple is epochwise, the list consists of models trained on first and second distr
        #assume the used distribution would have the largest increase in acc
        #dif1 is the differences of model trained on first distribution
        dif1 = np.array([acc_2[1][0]-acc_2[0][0],acc_1[1][0]-acc_1[0][0]])#reverse order to use index as indicator variable
        preds1 = np.argmax(dif1,axis=0)
        dif2 = np.array([acc_1[1][1]-acc_1[0][1],acc_2[1][1]-acc_2[0][1]])
        preds2 = np.argmax(dif2,axis=0)
        if get_preds:
            return [preds1,preds2]
        else:
            return (100*(np.mean(preds1)+np.mean(preds2)))/2
