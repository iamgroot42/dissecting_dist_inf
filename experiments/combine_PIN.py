from simple_parsing import ArgumentParser
from pathlib import Path
import os
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, WhiteBoxAttackConfig, BlackBoxAttackConfig, TrainConfig, CombineAttackConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult, IntermediateResult
from distribution_inference.attacks.whitebox.utils import wrap_into_loader
import distribution_inference.attacks.whitebox.utils as wu
from sklearn.tree import DecisionTreeClassifier

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
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
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
    if len(bb_attack_config.attack_type) > 1:
        raise ValueError(
            "This script only works with one blackbox attack")
    if bb_attack_config.attack_type[0] != "threshold_perpoint":
        raise ValueError(
            "This script only works with perpoint attack")
    if not wb_attack_config:
        raise ValueError(
            "This script need whitebox config")
    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")
    # Print out arguments
    flash_utils(attack_config)

    # Define logger
    DataLogger = IntermediateResult(args.en, attack_config)
    logger = AttackResult(args.en, attack_config, aname="Combine")
    if attack_config.save_bb:
        bb_logger = AttackResult(
            args.en+"_bb", attack_config, aname="blackbox")
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    # Load victim models for first value
    models_vic_1, vic1_names = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=attack_config.train_config.save_every_epoch,
        get_names=True)
    dims, features_vic_1 = ds_vic_1.get_features(
        train_config,
        wb_attack_config,
        models=models_vic_1)

    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)
        # Load victim models for other value
        models_vic_2, vic2_names = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            get_names=True)
        attack_model_path_dir = os.path.join(
            attack_config.wb_path, str(prop_value))
        attack_model_paths = []
        for a in os.listdir(attack_model_path_dir):
            attack_model_paths.append(a)
        #in case the number of trials doesn't match the # of metaclassifier
        for (t, attack_model_path) in zip(range(attack_config.tries), attack_model_paths):
            print("Ratio: {}, Trial: {}".format(prop_value, t))
            models_adv_1, adv1_names = ds_adv_1.get_models(
                train_adv_config,
                n_models=bb_attack_config.num_adv_models,
                on_cpu=attack_config.on_cpu,
                get_names=True)
            models_adv_2, adv2_names = ds_adv_2.get_models(
                train_adv_config,
                n_models=bb_attack_config.num_adv_models,
                on_cpu=attack_config.on_cpu,
                get_names=True)
            # Get victim and adv predictions on loaders for first ratio

            preds_adv_on_1, preds_vic_on_1, ground_truth_1 = get_vic_adv_preds_on_distr(
                models_vic=(models_vic_1, models_vic_2),
                models_adv=(models_adv_1, models_adv_2),
                ds_obj=ds_adv_1,
                batch_size=bb_attack_config.batch_size,
                epochwise_version=attack_config.train_config.save_every_epoch,
                preload=bb_attack_config.preload,
                multi_class=bb_attack_config.multi_class
            )
            # Get victim and adv predictions on loaders for second ratio
            preds_adv_on_2, preds_vic_on_2, ground_truth_2 = get_vic_adv_preds_on_distr(
                models_vic=(models_vic_1, models_vic_2),
                models_adv=(models_adv_1, models_adv_2),
                ds_obj=ds_adv_2,
                batch_size=bb_attack_config.batch_size,
                epochwise_version=attack_config.train_config.save_every_epoch,
                preload=bb_attack_config.preload,
                multi_class=bb_attack_config.multi_class
            )
            # Wrap predictions to be used by the attack
            bbm_preds_adv = PredictionsOnDistributions(
                preds_on_distr_1=preds_adv_on_1,
                preds_on_distr_2=preds_adv_on_2
            )
            bbm_preds_vic = PredictionsOnDistributions(
                preds_on_distr_1=preds_vic_on_1,
                preds_on_distr_2=preds_vic_on_2
            )

           #actually only perpoint
            for attack_type in bb_attack_config.attack_type:
                # Create attacker object
                attacker_obj = get_attack(attack_type)(bb_attack_config)

                # Launch attack
                result = attacker_obj.attack(
                    bbm_preds_adv, bbm_preds_vic,
                    ground_truth=(ground_truth_1, ground_truth_2),
                    calc_acc=calculate_accuracies,
                    epochwise_version=attack_config.train_config.save_every_epoch)
                if attack_config.save_bb:
                    bb_logger.add_results(attack_type, prop_value,
                                          result[0][0], result[1][0])
                classes_use = result[3]
                labels_adv = classes_use[0]
                labels_vic = classes_use[1]
                bb_preds_adv = result[1][1]
                bb_preds_vic = result[0][1]
               #done with bb, now wb
            _, features_vic_2 = ds_vic_2.get_features(
                train_config,
                wb_attack_config,
                models=models_vic_2)

            wb_test_loader = wrap_into_loader(
                [features_vic_1, features_vic_2],
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                wrap_with_loader=True,
                epochwise_version=attack_config.train_config.save_every_epoch)
            _, features_adv_1 = ds_adv_1.get_features(
                train_config,
                wb_attack_config,
                models=models_adv_1)
            _, features_adv_2 = ds_adv_2.get_features(
                train_config,
                wb_attack_config,
                models=models_adv_2)
            wb_train_loader = wrap_into_loader(
                [features_adv_1, features_adv_2],
                wrap_with_loader=True,
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                epochwise_version=attack_config.train_config.save_every_epoch)
            # Create attacker object
            attacker_obj = wu.get_attack(wb_attack_config.attack)(
                dims, wb_attack_config)

            # Load model
            attacker_obj.load_model(os.path.join(
                attack_model_path_dir, attack_model_path))
            wb_preds_adv = attacker_obj.eval_attack(
                test_loader=wb_train_loader,
                epochwise_version=attack_config.train_config.save_every_epoch,
                get_preds=True)
            wb_preds_vic = attacker_obj.eval_attack(
                test_loader=wb_test_loader,
                epochwise_version=attack_config.train_config.save_every_epoch,
                get_preds=True)
            preds_adv = np.vstack((wb_preds_adv, bb_preds_adv))
            preds_vic = np.vstack((wb_preds_vic, bb_preds_vic))
            preds_adv = np.transpose(preds_adv)
            preds_vic = np.transpose(preds_vic)
            #decision tree
            clf = DecisionTreeClassifier(max_depth=2)
            clf.fit(preds_adv, labels_adv)
            DataLogger.add_model_name(prop_value, (adv1_names, adv2_names), t)
            DataLogger.add_model(prop_value, clf, t)
            DataLogger.add_bb(prop_value, bbm_preds_adv,
                              bb_preds_adv, labels_adv, t)
            DataLogger.add_points(prop_value, None, t)
            logger.add_results("Combine", prop_value,
                               clf.score(preds_vic, labels_vic), clf.score(preds_adv, labels_adv))
    # Summarize results over runs, for each ratio and attack
    logger.save()
    DataLogger.save()
