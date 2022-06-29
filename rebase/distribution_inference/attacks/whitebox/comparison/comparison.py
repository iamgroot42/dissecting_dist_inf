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
from distribution_inference.attacks.blackbox.core import  PredictionsOnDistributions,PredictionsOnOneDistribution
from distribution_inference.attacks.blackbox.KL import sigmoid, KL
class ComparisonAttack:
    def __init__(self,
                t_config: TrainConfig,
                wb_config: WhiteBoxAttackConfig,
                info,
                save_m :bool=False,
                name:str=None
                ):
        self.t_config = replace(t_config)
        self.t_config.save_every_epoch = False
        self.name = name
        assert not (wb_config.save or wb_config.load), "Not implemented"
        assert wb_config.comparison_config, "No comparison config"
        self.wb_config = replace(wb_config)
        self.t_config.epochs=self.wb_config.comparison_config.End_epoch-self.wb_config.comparison_config.Start_epoch
        self.info_object=info
        self.trial=None
        self.v_r=None
        self.save=save_m
    def train(self,vic_models,ratio,v_r,trial:int):
        dp_config = None
        self.trial=trial
        self.v_r=v_r
        train_config: TrainConfig = replace(self.t_config)
        train_config.data_config.split="adv"
        train_config.data_config.value = ratio
        data_config: DatasetConfig = train_config.data_config
        assert data_config.split=="adv"
        assert data_config.value==ratio
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
                "save_path_fn": None})
            models.append(model)
            if self.save:
                if misc_config and misc_config.adv_config:
                    suffix = "_%.2f_adv_%.2f.ch" % (vacc[0], vacc[1])
                else:
                    suffix = "_%.2f.ch" % vacc

            # Get path to save model
                file_name = str(i + train_config.offset) + suffix
                save_path = self._get_save_path(train_config, file_name)

            # Save model
            
                save_model(model, save_path)
        return models
    def _model_save_path(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        dp_config = None
        shuffle_defense_config = None
        if train_config.misc_config is not None:
            dp_config = train_config.misc_config.dp_config
            shuffle_defense_config = train_config.misc_config.shuffle_defense_config
        # Standard logic
        if model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")
        if model_arch is None:
            model_arch = self.info_object.default_model
        base_models_dir = os.path.join(base_models_dir, "comparison_attack_adv_model",model_arch)

        if dp_config is None:
            if shuffle_defense_config is None:
                base_path = os.path.join(base_models_dir, "normal")
            else:
                base_path = os.path.join(base_models_dir, "shuffle_defense",
                                         "%.2f" % shuffle_defense_config.desired_value,
                                         "%.2f" % shuffle_defense_config.sample_ratio)
        else:
            base_path = os.path.join(
                base_models_dir, "DP_%.2f" % dp_config.epsilon)
        if train_config.label_noise:
            base_path = os.path.join(
                base_models_dir, "label_noise:{}".format(train_config.label_noise))
        assert self.v_r != None
        save_path = os.path.join(base_path,"model_from_{}".format(self.v_r))
        assert train_config.data_config.value != None
        save_path = os.path.join(save_path, "trained_on_{}".format(train_config.data_config.value))

        assert train_config.data_config.scale == 1.0
        if train_config.data_config.drop_senstive_cols:
            save_path = os.path.join(save_path, "drop")
        """
        if self.trial!=None:
            save_path = os.path.join(save_path, str(self.trial))
        """
        if self.name!=None:
            save_path = os.path.join(save_path, str(self.name))
        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path
    def _get_save_path(self, train_config: TrainConfig, name: str) -> str:
        prefix = self._model_save_path(train_config, model_arch=train_config.model_arch)
        if name is None:
            return prefix
        return os.path.join(prefix, name)
    def _attack_per_dis(self,
                preds_adv1:PredictionsOnOneDistribution,
                preds_adv2:PredictionsOnOneDistribution,
                preds_victim:PredictionsOnOneDistribution,
                KL_func: Callable=KL):
        p1, p2 = (sigmoid(preds_adv1.preds_property_1),sigmoid(preds_adv1.preds_property_2)), (sigmoid(preds_adv2.preds_property_1),sigmoid(preds_adv2.preds_property_2))
        # Predictions made by the  victim's models
        pv1, pv2 = sigmoid(preds_victim.preds_property_1), sigmoid(preds_victim.preds_property_2)
        KL1 = (np.array([np.mean(KL_func(p1_,pv1_)) for p1_,pv1_ in zip(p1[0],pv1)]),np.array([np.mean(KL_func(p2_,pv1_))  for p2_,pv1_ in zip(p1[1],pv1)]))
        KL2 =  (np.array([np.mean(KL_func(p1_,pv2_))  for p1_,pv2_ in zip(p2[0],pv2)]),np.array([np.mean(KL_func(p2_,pv2_))  for p2_,pv2_ in zip(p2[1],pv2)]))
        res1 = KL1[1] - KL1[0]
        res2 = KL2[0] - KL2[1]
        
        return (res1,res2)

    def attack(self,
               preds_adv1: PredictionsOnDistributions,
               preds_adv2: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions):
        r1 = self._attack_per_dis(preds_adv1.preds_on_distr_1,preds_adv2.preds_on_distr_1,preds_vic.preds_on_distr_1)
        r2 = self._attack_per_dis(preds_adv1.preds_on_distr_2,preds_adv2.preds_on_distr_2,preds_vic.preds_on_distr_2)
        p1 = np.hstack((r1[0],r2[0]))
        p2 = np.hstack((r1[1],r2[1]))
        preds_use = np.vstack((np.hstack(r1),np.hstack(r2)))
        acc_use = 50*(np.mean(p1)+np.mean(p2))
        choice_information = (None, None)
        return [(acc_use,preds_use), (None,None), choice_information]