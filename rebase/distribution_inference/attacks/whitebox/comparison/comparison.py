import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from typing import Callable
from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from simple_parsing import ArgumentParser
from copy import deepcopy
from dataclasses import replace
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train
from distribution_inference.training.utils import save_model
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig,AttackConfig,WhiteBoxAttackConfig
from scipy.stats import entropy
from distribution_inference.attacks.blackbox.core import  PredictionsOnDistributions,PredictionsOnOneDistribution
def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))
class ComparisonAttack:
    def __init__(self,
                t_config: TrainConfig,
                wb_config: WhiteBoxAttackConfig
                ):
        self.t_config = replace(t_config)
        self.t_config.save_every_epoch = False
        assert not (wb_config.save or wb_config.load), "Not implemented"
        assert wb_config.comparison_config, "No comparison config"
        self.wb_config = replace(wb_config)
        self.t_config.epochs=self.wb_config.comparison_config.End_epoch-self.wb_config.comparison_config.Start_epoch
        
    def train(self,vic_models,ratio):
        dp_config = None
        train_config: TrainConfig = self.t_config
        data_config: DatasetConfig = replace(train_config.data_config)
        data_config.split = "adv"
        data_config.value = ratio
        num_models = len(vic_models)
        models = []
        misc_config: MiscTrainConfig = train_config.misc_config
        if misc_config is not None:
            dp_config: DPTrainingConfig = misc_config.dp_config

        # TODO: Figure out best place to have this logic in the module
            if misc_config.adv_config:
            # Scale epsilon by 255 if requested
                if train_config.misc_config.adv_config.scale_by_255:
                    train_config.misc_config.adv_config.epsilon /= 255
        ds_wrapper_class = get_dataset_wrapper(data_config.name)
        # Create new DS object
        ds = ds_wrapper_class(data_config)
        for i in range(1, num_models+1):
            print("Training classifier %d / %d" % (i, num_models))
            train_loader, val_loader = ds.get_loaders(
            batch_size=train_config.batch_size)
            model, (vloss, vacc) = train(deepcopy(vic_models[i-1]), (train_loader, val_loader),
                                     train_config=train_config,
                                     extra_options={
                "curren_model_num": i + train_config.offset,
                "save_path_fn": ds.get_save_path})
            models.append(model)
        return models

    def _attack_per_dis(self,preds_victim:PredictionsOnOneDistribution,
                preds_adv:PredictionsOnOneDistribution,
                KL_func: Callable=entropy):
        p1, p2 = sigmoid(preds_adv.preds_property_1), sigmoid(preds_adv.preds_property_2)
        # Predictions made by the  victim's models
        pv1, pv2 = sigmoid(preds_victim.preds_property_1), sigmoid(preds_victim.preds_property_2)
        KL1 = (np.array([KL_func(p1_,pv1_) for p1_,pv1_ in zip(p1,pv1)]),np.array([KL_func(p2_,pv1_) for p2_,pv1_ in zip(p2,pv1)]))
        KL2 =  (np.array([KL_func(p1_,pv2_) for p1_,pv2_ in zip(p1,pv2)]),np.array([KL_func(p2_,pv2_) for p2_,pv2_ in zip(p2,pv2)]))
        res1 = KL1[1] - KL1[0]
        res2 = KL2[0] - KL2[1]
        acc1 = np.average(res1>=0)
        acc2 = np.average(res2>=0)
        return 100*(acc1+acc2)/2, np.hstack((res1,res2))

    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions):
        acc_1,preds_1 = self._attack_per_dis(preds_adv.preds_on_distr_1,preds_vic.preds_on_distr_1)
        acc_2,preds_2 = self._attack_per_dis(preds_adv.preds_on_distr_2,preds_vic.preds_on_distr_2)
        # Get best adv accuracies for both distributions, across all ratios
        chosen_distribution = 0
        if acc_1>acc_2:
            acc_use,preds_use = acc_1,preds_1
        else:
            acc_use,preds_use = acc_2,preds_2
            chosen_distribution = 1

        # Of the chosen distribution, pick the one with the best accuracy
        # out of all given ratios
        
        choice_information = (chosen_distribution, None)
        return [(acc_use,preds_use), (None,None), choice_information]