import numpy as np
from typing import Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions
from distribution_inference.attacks.blackbox.core import _acc_per_dis
from sklearn import multiclass
DUMPING  = 10
class Epoch_LossAttack(Attack):
    def attack(self,
               preds_vic1: PredictionsOnDistributions,
               preds_vic2: PredictionsOnDistributions,
               preds_adv1: PredictionsOnDistributions,
               preds_adv2: PredictionsOnDistributions,
               ground_truth,
               calc_acc: Callable,
               get_preds:bool=False,
               ratio:bool=False,
               not_using_logits: bool = False):
        self.ratio = ratio
        multi_class = self.config.multi_class
        acc_1 = _acc_per_dis(preds_vic1.preds_on_distr_1,
                            preds_vic2.preds_on_distr_1,
                            ground_truth[0],
                            calc_acc,
                            multi_class=multi_class)
        acc_2 = _acc_per_dis(preds_vic1.preds_on_distr_2,
                            preds_vic2.preds_on_distr_2,
                            ground_truth[1],
                            calc_acc,multi_class=multi_class)
        return self._loss_test(acc_1,acc_2,get_preds)

    def _loss_test(self, acc_1, acc_2, get_preds: bool = False):
        #acc_1 is a tupple of list of accs on first distribution, tuple is epochwise, the list consists of models trained on first and second distr
        #assume the used distribution would have the largest increase in acc
        #dif1 is the differences of model trained on first distribution
        if self.ratio:
            dif1 = np.array([(acc_1[1][0]+DUMPING)/(acc_1[0][0]+DUMPING),(acc_2[1][0]+DUMPING)/(acc_2[0][0]+DUMPING)])#reverse order to use index as indicator variable
            preds1 = np.argmax(dif1,axis=0)
            dif2 = np.array([(acc_2[1][1]+DUMPING)/(acc_2[0][1]+DUMPING),(acc_1[1][1]+DUMPING)/(acc_1[0][1]+DUMPING)])
            preds2 = np.argmax(dif2,axis=0)
        else:
            dif1 = np.array([acc_1[1][0]-acc_1[0][0],acc_2[1][0]-acc_2[0][0]])#reverse order to use index as indicator variable
            preds1 = np.argmax(dif1,axis=0)
            dif2 = np.array([acc_2[1][1]-acc_2[0][1],acc_1[1][1]-acc_1[0][1]])
            preds2 = np.argmax(dif2,axis=0)
        if get_preds:
            return np.concatenate([preds1,preds2])
        else:
            return (((100*(np.mean(preds1)+np.mean(preds2)))/2,None),(None,None),None)
