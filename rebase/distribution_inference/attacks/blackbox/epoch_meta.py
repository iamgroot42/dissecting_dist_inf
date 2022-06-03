from simple_parsing import ArgumentParser
from pathlib import Path
import os
import numpy as np
from typing import List, Callable
from sklearn.tree import DecisionTreeClassifier
from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistribution
class Epoch_Tree(Attack):
    def attack(self,
               preds_vics: List[PredictionsOnDistributions],
               preds_advs: List[PredictionsOnDistributions]):
        
        assert not (self.config.multi2 and self.config.multi), "No implementation for both multi model"
        clf = DecisionTreeClassifier(max_depth=5)
        tr1 = [np.hstack((pa.preds_on_distr_1.preds_property_1,pa.preds_on_distr_2.preds_property_1)) for pa in preds_advs]
        tr2 = [np.hstack((pa.preds_on_distr_1.preds_property_2,pa.preds_on_distr_2.preds_property_2)) for pa in preds_advs]
        te1 = [np.hstack((pv.preds_on_distr_1.preds_property_1,pv.preds_on_distr_2.preds_property_1)) for pv in preds_vics]
        te2 = [np.hstack((pv.preds_on_distr_1.preds_property_2,pv.preds_on_distr_2.preds_property_2)) for pv in preds_vics]
        
        tr1 = np.hstack(tr1)
        tr2 = np.hstack(tr2)
        te1 = np.hstack(te1)
        te2 = np.hstack(te2)
        tr = np.concatenate((tr1,tr2),axis=0)
        te = np.concatenate((te1,te2),axis=0)
        labels_adv = np.hstack((np.zeros(len(preds_advs[0].preds_on_distr_1.preds_property_1)),np.ones(len(preds_advs[0].preds_on_distr_1.preds_property_2))))
        labels_vic = np.hstack((np.zeros(len(preds_vics[0].preds_on_distr_1.preds_property_1)),np.ones(len(preds_vics[0].preds_on_distr_1.preds_property_2))))
        clf.fit(tr, labels_adv)
        return ((clf.score(te, labels_vic),clf.predict_proba(te)),(clf.score(tr, labels_adv),clf.predict_proba(tr)),clf)
    


    