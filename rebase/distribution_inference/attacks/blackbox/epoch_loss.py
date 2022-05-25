import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistributions

class Epoch_LossAttack(Attack):
    def attack(self,
               preds_vic1: PredictionsOnDistributions,
               preds_vic2: PredictionsOnDistributions,
               ground_truth,
               calc_acc: Callable):
        acc_1 = self._acc_per_dis(preds_vic1.preds_on_distr_1,
                            preds_vic2.preds_on_distr_1,
                            ground_truth,
                            calc_acc)
        acc_2 = self._acc_per_dis(preds_vic1.preds_on_distr_2,
                            preds_vic2.preds_on_distr_2,
                            ground_truth,
                            calc_acc)
        return self._loss_test(acc_1,acc_2)
    
    def _acc_per_dis(self,
                preds_vicd1: PredictionsOnOneDistributions,
               preds_vicd2: PredictionsOnOneDistributions,
               ground_truth,
               calc_acc: Callable):
        #pi means ith epoch
        p1 = (preds_vicd1.preds_property_1, preds_vicd1.preds_property_2)
        p2 = (preds_vicd2.preds_property_1, preds_vicd2.preds_property_2)
        for i in range(2):
            p1[i] = np.transpose(p1[i])
            p2[i] = np.transpose(p2[i])
        acc1 = [100*calc_acc(p,ground_truth) for p in p1]
        acc2 = [100*calc_acc(p,ground_truth) for p in p2]
        return (np.array(acc1),np.array(acc2))

    def _loss_test(self,acc_1,acc_2):
        #acc_1 is a tupple of list of accs on first distribution, tuple is epochwise, the list consists of models trained on first and second distr
        #assume the used distribution would have the largest increase in acc
        #dif1 is the differences of model trained on first distribution
        dif1 = np.array([acc_2[1][0]-acc_2[0][0]],acc_1[1][0]-acc_1[0][0])#reverse order to use index as indicator variable
        preds1 = np.argmax(dif1,axis=0)
        dif2 = np.array([acc_1[1][1]-acc_1[0][1],acc_2[1][1]-acc_2[0][1]])
        preds2 = np.argmax(dif2,axis=0)
        return (100*(np.mean(preds1)+np.mean(preds2)))/2
