"""
    Attack proposed in the paper 'Leakage of Dataset Properties in Multi-Party Machine Learning'
"""
import numpy as np
from typing import Tuple
from typing import List, Callable
from sklearn.neural_network import MLPClassifier
from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistribution,PredictionsOnOneDistribution


class ZhangAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False):
        
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        if not epochwise_version:
            return self.attack_not_epoch(preds_adv,preds_vic,ground_truth,calc_acc,not_using_logits)
        else:
            preds_v = [PredictionsOnDistributions(
                PredictionsOnOneDistribution(preds_vic.preds_on_distr_1.preds_property_1[i],preds_vic.preds_on_distr_1.preds_property_2[i]),
                PredictionsOnOneDistribution(preds_vic.preds_on_distr_2.preds_property_1[i],preds_vic.preds_on_distr_2.preds_property_2[i])
            ) for i in range(len(preds_vic.preds_on_distr_2.preds_property_1))]
            accs,preds=[],[]
            for x in preds_v:
                result = self.attack_not_epoch(preds_adv,x,ground_truth,calc_acc,not_using_logits)
                accs.append(result[0][0])
                preds.append(result[0][1])
            return [(accs, preds), (None, None), (None,None)]

    def _attack_on_data(self,
                        preds_adv: PredictionsOnOneDistribution,
                        preds_vic: PredictionsOnOneDistribution,
                        not_using_logits: bool = False):
        ka, kb = preds_adv.preds_property_1, preds_adv.preds_property_2
        kc1, kc2 = preds_vic.preds_property_1, preds_vic.preds_property_2
        # Apply sigmoid to ones that are not already sigmoided
        if not self.not_using_logits:
            if self.config.multi_class:
                ka, kb = softmax(ka), softmax(kb)
                kc1, kc2 = softmax(kc1), softmax(kc2)
            else:
                ka, kb = sigmoid(ka), sigmoid(kb)
                kc1, kc2 = sigmoid(kc1), sigmoid(kc2)
        
        # Use same architecture as in paper
        self.meta_model = MLPClassifier((20, 8), early_stopping=True)

        # Train BB meta-classifier
        X_train = np.concatenate((ka, kb), axis=0)
        y_train = np.concatenate((np.ones(len(ka)), np.zeros(len(kb))), axis=0)
        self.meta_model.fit(X_train, y_train)
        val_acc = self.meta_model.best_validation_score_

        # Predict on victim
        X_test = np.concatenate((kc1, kc2), axis=0)
        y_test = np.concatenate((np.ones(len(kc1)), np.zeros(len(kc2))), axis=0)
        y_pred = self.meta_model.predict_proba(X_test)[:, 1]
        test_acc = self.meta_model.score(X_test, y_test)

        return y_pred, (val_acc, test_acc)

    def attack_not_epoch(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               not_using_logits: bool = False):
        
        self.not_using_logits = not_using_logits
        self.meta_model = MLPClassifier((20, 8)) # Use same architecture as in paper

        # Try attack with data from both distributions
        preds_1, (tr_acc_1, te_acc_1) = self._attack_on_data(preds_adv.preds_on_distr_1, preds_vic.preds_on_distr_1)
        preds_2, (tr_acc_2, te_acc_2) = self._attack_on_data(preds_adv.preds_on_distr_2, preds_vic.preds_on_distr_2)

        # Pick the one that performs best locally
        if tr_acc_1 > tr_acc_2:
            victim_pred_use, victim_acc_use = preds_1, te_acc_1
        else:
            victim_pred_use, victim_acc_use = preds_2, te_acc_2

        # No concept of "choice" (are we in the Matrix :P)
        return [(100*victim_acc_use, victim_pred_use), (None, None), (None,None)]


def sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)


def softmax(x):
    # Numericaly stable softmax
    z = x - np.max(x, -1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, -1, keepdims=True)
