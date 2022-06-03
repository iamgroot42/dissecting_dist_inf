from simple_parsing import ArgumentParser
from pathlib import Path
import os
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_preds_epoch_on_dis
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, WhiteBoxAttackConfig, BlackBoxAttackConfig, TrainConfig, CombineAttackConfig
from distribution_inference.utils import flash_utils, ensure_dir_exists,warning_string
from distribution_inference.logging.core import AttackResult, IntermediateResult
from distribution_inference.attacks.whitebox.utils import  wrap_into_loader
import distribution_inference.attacks.whitebox.utils as wu
from sklearn.tree import DecisionTreeClassifier
from joblib import load, dump
from distribution_inference.attacks.blackbox.epoch_meta import Epoch_Tree

from distribution_inference.attacks.whitebox.affinity.utils import get_seed_data_loader, identify_relevant_points, make_ds_and_loader
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
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: CombineAttackConfig = CombineAttackConfig.load(
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
    if wb_attack_config.attack!="affinity":
        raise ValueError("This script only takes affinity attack")
    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")
    # Print out arguments
    flash_utils(attack_config)

    # Define logger
    logger = AttackResult(args.en, attack_config,aname = "Combine")
    DataLogger = IntermediateResult(args.en,attack_config)
    if attack_config.save_bb:
        bb_logger = AttackResult(args.en+"_bb", attack_config,aname = "blackbox")
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1,epoch=True)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True,epoch=True)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    # Load victim models for first value
    models_vic_1,vic1_names = ds_vic_1.get_models(
        train_config,
        n_models=50,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=True,
                get_names=True)
    
    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2,epoch=True)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True,epoch=True)
        # Load victim models for other value
        models_vic_2 ,vic2_names= ds_vic_2.get_models(
            train_config,
            n_models=50,
            epochwise_version=True,
                get_names=True)
        attack_model_path_dir = os.path.join(attack_config.wb_path, str(prop_value))
        attack_model_paths = []
        for a in os.listdir(attack_model_path_dir):
            attack_model_paths.append(a)
        #in case the number of trials doesn't match the # of metaclassifier
        for (t,attack_model_path) in zip(range(attack_config.tries),attack_model_paths):
            print("Ratio: {}, Trial: {}".format(prop_value,t))
            models_adv_1,adv1_names = ds_adv_1.get_models(
                train_adv_config,
                n_models=50,
                epochwise_version=True,
                on_cpu=attack_config.on_cpu,
                get_names=True)
            models_adv_2, adv2_names = ds_adv_2.get_models(
                train_adv_config,
                n_models=50,
                epochwise_version=True,
                on_cpu=attack_config.on_cpu,
                get_names=True)
            labels_adv = np.hstack((np.zeros(models_adv_1.shape[1]),np.ones(models_adv_2.shape[1])))
            labels_vic = np.hstack((np.zeros(models_vic_1.shape[1]),np.ones(models_vic_2.shape[1])))
            # Get victim and adv predictions on loaders for first ratio
            #only support default random selection of points
            seed_data_ds, seed_data_loader,adv_l,raw_data = get_seed_data_loader(
                        [ds_adv_1, ds_adv_2],
                        wb_attack_config,
                        num_samples_use=wb_attack_config.affinity_config.num_samples_use,
                        also_get_raw_data=True)
            
            # Wrap predictions to be used by the attack
            
            attacker_obj = wu.get_attack(wb_attack_config.attack)(
                None, wb_attack_config)
            attacker_obj.register_seed_data(seed_data_ds)
            # Load model
            attacker_obj.load_model(os.path.join(
                attack_model_path_dir, attack_model_path))
            adv_test = [wrap_into_loader(
            [ma1, ma2],
            batch_size=wb_attack_config.batch_size,
            shuffle=False,
            wrap_with_loader=False
            ) for ma1, ma2 in zip(models_adv_1,models_adv_2)]
            vic_test =  [wrap_into_loader(
            [mv1, mv2],
            batch_size=wb_attack_config.batch_size,
            shuffle=False,
            wrap_with_loader=False
            ) for mv1, mv2 in zip(models_vic_1,models_vic_2)]
            # Make affinity features for train (adv) models
            features_adv = [attacker_obj.make_affinity_features(
                adv[0], seed_data_loader, labels=adv_test[1]) for adv in adv_test]
            # Make affinity features for victim models
            features_vic = [attacker_obj.make_affinity_features(
                vic[0], seed_data_loader) for vic in vic_test]

            
            

            wb_preds_adv = [attacker_obj.eval_attack(
                test_loader=(fadv,adv[1]),
                get_preds = True) for fadv,adv in zip(features_adv,adv_test)]

            wb_preds_vic = [attacker_obj.eval_attack(
                test_loader=(fvic,vic[1]),
                get_preds = True) for fvic,vic in zip(features_vic,vic_test)]
            #decision tree
            preds_adv = np.vstack(wb_preds_adv)
            preds_vic = np.vstack(wb_preds_vic)
            preds_adv = np.transpose(preds_adv)
            preds_vic = np.transpose(preds_vic)
            clf = DecisionTreeClassifier(max_depth=4)
            clf.fit(preds_adv, labels_adv)
            #log results
            DataLogger.add_model_name(prop_value,(adv1_names,adv2_names),t)
            DataLogger.add_model(prop_value,clf,t)
            DataLogger.add_points(prop_value,raw_data,t)
            logger.add_results("Combine", prop_value,
                                clf.score(preds_vic, labels_vic), clf.score(preds_adv, labels_adv))
    # Summarize results over runs, for each ratio and attack
    logger.save()
    DataLogger.save()
