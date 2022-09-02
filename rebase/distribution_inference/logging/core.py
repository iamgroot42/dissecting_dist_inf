import json
from pathlib import Path
from distribution_inference.config.core import TrainConfig
from typing import List
from copy import deepcopy
from datetime import datetime
import os
from simple_parsing.helpers import Serializable
import pickle
from distribution_inference.config import AttackConfig, DefenseConfig
from distribution_inference.utils import get_save_path


class Result:
    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self.path = path
        self.start = datetime.now()
        self.dic = {'name': name, 'start time': str(self.start)}

    def save(self, json_: bool = True):
        self.save_t = datetime.now()
        self.dic['save time'] = str(self.save_t)

        self.path.mkdir(parents=True, exist_ok=True)
        if json_:
            save_p = self.path.joinpath(f"{self.name}.json")
            with save_p.open('w') as f:
                json.dump(self.dic, f, indent=4)
        else:
            save_p = self.path.joinpath(f"{self.name}.p")
            with save_p.open('wb') as f:
                pickle.dump(self.dic, f)

    def not_empty_dic(self, dic: dict, key):
        if key not in dic:
            dic[key] = {}

    def convert_to_dict(self, dic: dict):
        for k in dic:
            if isinstance(dic[k], Serializable):
                dic[k] = dic[k].__dict__
            if isinstance(dic[k], dict):
                self.convert_to_dict(dic[k])

    def load(self):
        raise NotImplementedError("Implement method to model for logger")

    def check_rec(self, dic: dict, keys: List):
        if not keys == []:
            k = keys.pop(0)
            self.not_empty_dic(dic, k)
            self.check_rec(dic[k], keys)

    def conditional_append(self, dict, key, item):
        if key not in dict:
            dict[key] = []
        dict[key].append(item)


class AttackResult(Result):
    def __init__(self,
                 experiment_name: str,
                 attack_config: AttackConfig,
                 D0 = None,
                 aname: str = None):
        # Infer path from data_config inside attack_config
        dataset_name = attack_config.train_config.data_config.name
        # Figure out if BB attack or WB attack
        if aname:
            attack_name = aname
        else:
            attack_name = "blackbox" if attack_config.white_box is None else "whitebox"
        save_path = get_save_path()
        if D0==None:
            path = Path(os.path.join(save_path, dataset_name, attack_name))
            super().__init__(path, experiment_name)
        else:
            path = Path(os.path.join(save_path, dataset_name, attack_name,experiment_name))
            super().__init__(path, str(D0))

        self.dic["attack_config"] = deepcopy(attack_config)
        self.convert_to_dict(self.dic)

    def add_results(self, attack: str, prop, vacc, adv_acc=None):
        self.check_rec(self.dic, ['result', attack, prop])
        # Log adv acc
        self.conditional_append(self.dic['result'][attack][prop],
                                'adv_acc', adv_acc)
        # Log victim acc
        self.conditional_append(self.dic['result'][attack][prop],
                                'victim_acc', vacc)


class IntermediateResult(Result):
    def __init__(self,
                 name: str,
                 attack_config: AttackConfig):
        dataset_name = attack_config.train_config.data_config.name
        save_path = get_save_path()
        path = Path(os.path.join(
            save_path, dataset_name, "Intermediate_result"))
        super().__init__(path, name)
        self.dic["attack_config"] = deepcopy(attack_config)

    def _add_results(self, item: str, prop, value, trial: int):
        self.check_rec(self.dic, [item, prop])
        self.dic[item][prop][trial] = value

    def add_model_name(self, prop, names: List[str], trial: int):
        self._add_results("model_names", prop, names, trial)

    def add_points(self, prop, points, trial: int):
        self._add_results("points", prop, points, trial)

    def add_bb(self, prop, models_preds: List,
               preds: List, labels: List,
               trial: int,
               is_victim: bool = False):
        field_name = "blackbox_vic" if is_victim else "blackbox"
        self._add_results(field_name, prop,
                          (models_preds, preds, labels), trial)

    def add_wb(self, prop, preds: List,
               labels: List, trial: int,
               is_victim: bool = False,
               is_raw_feature: bool = False):
        field_name = "whitebox_vic" if is_victim else "whitebox"
        if is_raw_feature:
            field_name += "_features"
        self._add_results(field_name, prop, (preds, labels), trial)

    def add_model(self, prop, model, trial: int):
        self._add_results("model", prop, model, trial)

    def save(self):
        super().save(json_=False)


class DefenseResult(Result):
    def __init__(self,
                 experiment_name: str,
                 defense_config: DefenseConfig):
        # Infer path from data_config inside attack_config
        dataset_name = defense_config.train_config.data_config.name
        # Figure out if BB attack or WB attack
        defense_name = "unlearning" if defense_config.unlearning_config else "unknown_defense"
        save_path = get_save_path()
        path = Path(os.path.join(save_path, dataset_name, defense_name))
        super().__init__(path, experiment_name)

        self.dic["defense_config"] = deepcopy(defense_config)
        self.convert_to_dict(self.dic)

    def add_results(self, defense: str, prop,
                    before_acc: float,
                    after_acc: float,):
        self.check_rec(self.dic, ['result', defense, prop])
        # Log before-defense acc
        self.conditional_append(self.dic['result'][defense][prop],
                                'before_acc', before_acc)
        # Log after-defense acc
        self.conditional_append(self.dic['result'][defense][prop],
                                'after_acc', after_acc)


class TrainingResult(Result):
    def __init__(self,
                 experiment_name: str,
                 train_config: TrainConfig):
        # Infer path from data_config inside attack_config
        dataset_name = train_config.data_config.name
        save_path = get_save_path()
        path = Path(os.path.join(save_path, dataset_name, "training"))
        super().__init__(path, experiment_name)

        self.dic["train_config"] = deepcopy(train_config)
        self.convert_to_dict(self.dic)

    def add_result(self, prop, **metrics):
        self.check_rec(self.dic, ['log', prop])
        # Log loss of model
        
        
        for e in metrics.keys():
            self.conditional_append(self.dic['log'][prop],
                    e, metrics[e])
        
