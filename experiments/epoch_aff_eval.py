from simple_parsing import ArgumentParser
from pathlib import Path
import os
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, WhiteBoxAttackConfig, TrainConfig, AttackConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult, IntermediateResult
from distribution_inference.attacks.whitebox.utils import wrap_into_loader
import distribution_inference.attacks.whitebox.utils as wu
from sklearn.tree import DecisionTreeClassifier
from distribution_inference.attacks.whitebox.affinity.utils import get_loader_for_seed_data


#a bit messy in this file. Might need to move something out to functions in rebase
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
        "--meta_path", help="path to trained meta-models directory",
        type=str, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file

    wb_attack_config: WhiteBoxAttackConfig = attack_config.white_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255
    if attack_config.adv_misc_config is not None:
        if attack_config.adv_misc_config.adv_config:
            if attack_config.adv_misc_config.adv_config.scale_by_255:
                attack_config.adv_misc_config.adv_config.epsilon /= 255

    if not wb_attack_config:
        raise ValueError(
            "This script need whitebox config")
    if wb_attack_config.attack != "affinity":
        raise ValueError("This script only takes affinity attack")
    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")
    # Print out arguments
    flash_utils(attack_config)

    # Define logger
    logger = AttackResult(args.en, attack_config)
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)(True)

    # Create new DS object for both and victim
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1, epoch=True)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True, epoch=True)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
            train_config,
            n_models=50,
            epochwise_version=True,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            model_arch=attack_config.victim_model_arch)

    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2, epoch=True)
        ds_vic_2 = ds_wrapper_class(        
            data_config_vic_2, skip_data=True, epoch=True)
        # Load victim models for other value
        models_vic_2 = ds_vic_2.get_models(
                train_config,
                n_models=50,
                epochwise_version=True,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                model_arch=attack_config.victim_model_arch)

        attack_model_path_dir = os.path.join(
            args.meta_path, str(prop_value))
        attack_model_paths = []
        for a in os.listdir(attack_model_path_dir):
            attack_model_paths.append(a)
        #in case the number of trials doesn't match the # of metaclassifier
        t = 0 
        for attack_model_path in attack_model_paths:
            if os.path.isdir(os.path.join(attack_model_path_dir, attack_model_path)):
                    continue
            
            print("Ratio: {}, Trial: {}".format(prop_value, t))
            

            attacker_obj = wu.get_attack(wb_attack_config.attack)(
                None, wb_attack_config)
            
            # Load model
            attacker_obj.load_model(os.path.join(
                attack_model_path_dir, attack_model_path))
            seed_data_loader =get_loader_for_seed_data(attacker_obj.seed_data_ds, wb_attack_config)
            
            vic_test = [wrap_into_loader(
                [mv1, mv2],
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                wrap_with_loader=False
            ) for mv1, mv2 in zip(models_vic_1, models_vic_2)]
           
            # Make affinity features for victim models
            features_vic = [attacker_obj.make_affinity_features(
                vic[0], seed_data_loader) for vic in vic_test]

            accs = [attacker_obj.eval_attack(
                    test_loader=(features_test, test_data[1]))  for features_test,test_data
                    in zip(features_vic,vic_test)]
            
            logger.add_results("affinity", prop_value,
                               accs, None)
            t+=1
    # Summarize results over runs, for each ratio and attack
    logger.save()
  
