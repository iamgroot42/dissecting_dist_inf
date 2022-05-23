import json
from pathlib import Path
from typing import List
from copy import deepcopy
from datetime import datetime
import os
from simple_parsing.helpers import Serializable
import pickle
from distribution_inference.config import AttackConfig
from distribution_inference.utils import get_save_path


class Result:
    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self.path = path
        self.start = datetime.now()
        self.dic = {'name': name, 'start time': str(self.start)}

    def save(self, jsob:bool=True):
        self.save_t = datetime.now()
        self.dic['save time'] = str(self.save_t)
        
        self.path.mkdir(parents=True, exist_ok=True)
        if jsob:
            save_p = self.path.joinpath(f"{self.name}.json")
            with save_p.open('w') as f:
                json.dump(self.dic, f)
        else:
            save_p = self.path.joinpath(f"{self.name}.p")
            with save_p.open('w') as f:
                pickle.dump(self.dic,f)

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
    
class AttackResult(Result):
    def __init__(self,
                 experiment_name: str,
                 attack_config: AttackConfig,
                 aname:str=None):
        # Infer path from data_config inside attack_config
        dataset_name = attack_config.train_config.data_config.name
        # Figure out if BB attack or WB attack
        if aname:
            attack_name = aname
        else:
            attack_name = "blackbox" if attack_config.white_box is None else "whitebox"
        save_path = get_save_path()
        path = Path(os.path.join(save_path, dataset_name, attack_name))
        super().__init__(path, experiment_name)

        self.dic["attack_config"] = deepcopy(attack_config)
        self.convert_to_dict(self.dic)

    def add_results(self, attack: str, prop, vacc, adv_acc=None):
        self.check_rec(self.dic, ['result', attack, prop])
        if 'adv_acc' in self.dic['result'][attack][prop]:
            self.dic['result'][attack][prop]['adv_acc'].append(adv_acc)
        else:
            self.dic['result'][attack][prop]['adv_acc'] = [adv_acc]
        if 'victim_acc' in self.dic['result'][attack][prop]:
            self.dic['result'][attack][prop]['victim_acc'].append(vacc)
        else:
            self.dic['result'][attack][prop]['victim_acc'] = [vacc]

class IntermediateResult(Result):
    def __init__(self,
                 name: str,
                 attack_config: AttackConfig):
        dataset_name = attack_config.train_config.data_config.name
        save_path = get_save_path()
        path = Path(os.path.join(save_path, dataset_name, "Intermediate_result"))
        super().__init__(path, name)
        self.dic["attack_config"] = deepcopy(attack_config)

    def _add_results(self, item:str,prop,value,trial:int):
        self.check_rec(self.dic, [item, prop])
        self.dic[item][prop][trial] = value
    
    def add_model_name(self,prop,names:List,trial:int):
        self._add_results("model_names",prop,names,trial)
    
    def add_points(self,prop,points:List,trial:int):
        self._add_results("adv points",prop,points,trial)
    def add_bb(self,prop,model_preds,preds,labels,trial:int):
        self._add_results("blackbox",prop,(models_preds,preds,labels),trial)

    def add_model(self,prop,model,trial:int):
        self._add_results("model",prop,model,trial)
    def save(self):
        super.save(jsob=False)