from distribution_inference.attacks.blackbox.KL import KLAttack,sigmoid,KL
from distribution_inference.attacks.blackbox.core import Attack
from typing import List, Callable,Tuple
import numpy as np

class KLRegression(KLAttack):
    def attack(self,
               preds_adv: List[np.ndarray],
               preds_vic: List[np.ndarray],
               ground_truth: List[Tuple[List, List]] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False,
               labels: List[float] = None):
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        self.not_using_logits = not_using_logits
        p_c = [] 
        #nested for loop to test all combinations
        for pv in preds_vic:
            p_across = []
            for pa in preds_adv:
                preds =  self._get_kl_preds(pa,pv)
                p_across.append(preds)
            p_across = np.array(p_across)
            p = np.divide(np.sum([i*x for i,x in zip(labels,p_across)],axis=0),
            np.sum(p_across,axis=0)
            )
            p_c.append(p)
        gt = np.array([[i]*preds_vic[0].shape[0] for i in labels])
        return np.square(p_c-gt).mean() #MSE
    def _get_kl_preds(self, ka,kc1):
        ka_,kc1_ = ka,kc1
        if not self.not_using_logits:
            ka_, kc1_ = sigmoid(ka), sigmoid(kc1)
        KL_vals_1_a = np.array([KL(ka_, x,
            multi_class=self.config.multi_class) for x in kc1_])
        return KL_vals_1_a
