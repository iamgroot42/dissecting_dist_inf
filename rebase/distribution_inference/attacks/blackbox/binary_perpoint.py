"""
This script only uses label predictions
"""
import numpy as np
import torch as ch
from typing import List, Tuple, Callable, Union

from distribution_inference.attacks.blackbox.core import Attack, order_points, PredictionsOnOneDistribution, PredictionsOnDistributions, multi_model_sampling, get_threshold_pred_multi
from distribution_inference.config import BlackBoxAttackConfig
from distribution_inference.attacks.blackbox.per_point import PerPointThresholdAttack

class BinaryPerPointThresholdAttack(PerPointThresholdAttack):
    def __init__(self, config: BlackBoxAttackConfig):
        super().__init__(config)
        assert not config.multi_class

    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False):
        """
        Take predictions from both distributions and run attacks.
        Pick the one that works best on adversary's models
        """
        t = 0.5 if not_using_logits else 0
        pa = PredictionsOnDistributions(
            PredictionsOnOneDistribution(preds_adv.preds_on_distr_1.preds_property_1>=t,
            preds_adv.preds_on_distr_1.preds_property_2>=t),
            PredictionsOnOneDistribution(preds_adv.preds_on_distr_2.preds_property_1>=t,
            preds_adv.preds_on_distr_2.preds_property_2>=t)
        )
        pv = PredictionsOnDistributions(
            PredictionsOnOneDistribution(preds_vic.preds_on_distr_1.preds_property_1>=t,
            preds_vic.preds_on_distr_1.preds_property_2>=t),
            PredictionsOnOneDistribution(preds_vic.preds_on_distr_2.preds_property_1>=t,
            preds_vic.preds_on_distr_2.preds_property_2>=t)
        )
        return super().attack(pa,pv,ground_truth,epochwise_version=epochwise_version,not_using_logits=not_using_logits)