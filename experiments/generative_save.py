import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from simple_parsing import ArgumentParser
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions, PredictionsOnOneDistribution
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig,GenerativeAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils,ensure_dir_exists
from distribution_inference.attacks.blackbox.generative import GenerativeAttack
import pickle
if __name__ == "__main__":
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    parser.add_argument('--trial_offset',type=int,
                        default=None, help="device number")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    generative_config:GenerativeAttackConfig = bb_attack_config.generative_attack
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    assert generative_config is not None
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255
    # Do the same if adv_misc_config is present
    if attack_config.adv_misc_config is not None:
        if attack_config.adv_misc_config.adv_config:
            if attack_config.adv_misc_config.adv_config.scale_by_255:
                attack_config.adv_misc_config.adv_config.epsilon /= 255
    flash_utils(attack_config)
    BATCH_SIZE=bb_attack_config.batch_size
    
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()
   
    attacker_obj = GenerativeAttack(bb_attack_config)
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    
    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False)
    
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    
    preds_a = {}
    preds_v = {}
    x_use = {}
    
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)
        
        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)

        # Load victim models for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)
        
        
        preds_a[prop_value] = {}
        preds_v[prop_value] = {}
        x_use[prop_value] = {}
        for t in range(attack_config.tries):
            print("{}: trial {}".format(prop_value, t))
            models_adv_1 = ds_adv_1.get_models(
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch)
            models_adv_2 = ds_adv_2.get_models(
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch)
            generated = attacker_obj.gen_data(models_adv_1,models_adv_2,ds_adv_1,ds_adv_2,BATCH_SIZE,generative_config)
            preds_adv = attacker_obj.preds_wrapper(models_adv_1,models_adv_2,generated[0],generated[1])
            preds_vic = attacker_obj.preds_wrapper(models_vic_1,models_vic_2,generated[0],generated[1])
            
            preds_a[prop_value][t]= preds_adv
            preds_v[prop_value][t]= preds_vic
            x_use[prop_value][t]= generated
    preds_path = os.path.join(ds_info.base_models_dir,"generative","preds",args.en)
    ensure_dir_exists(preds_path)
    attack_config.save(os.path.join(preds_path,"config.json"),indent=4)
    with open(os.path.join(preds_path,"preds_a.p"),"wb") as f:
        pickle.dump(preds_a, f)
    with open(os.path.join(preds_path,"preds_v.p"),"wb") as f:
        pickle.dump(preds_vic, f)
    with open(os.path.join(preds_path,"generated.p"),"wb") as f:
        pickle.dump(x_use, f)