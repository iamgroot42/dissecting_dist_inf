from simple_parsing import ArgumentParser
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions, PredictionsOnOneDistribution
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, WhiteBoxAttackConfig, BlackBoxAttackConfig, TrainConfig, CombineAttackConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.whitebox.utils import wrap_into_loader
import distribution_inference.attacks.whitebox.utils as wu
from sklearn.tree import DecisionTreeClassifier
from distribution_inference.attacks.whitebox.affinity.utils import  get_loader_for_seed_data


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
    if bb_attack_config.attack_type[0] != "KL":
        raise ValueError(
            "This script only works with KL attack")
    if wb_attack_config is None:
        raise ValueError(
            "This script needs a whitebox config")
    if wb_attack_config.attack != "affinity":
        raise ValueError("This script only takes affinity attack")
    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")
    # Print out arguments
    flash_utils(attack_config)

    # Define logger
    logger = AttackResult(args.en, attack_config, aname="AGA+KL")
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
        attack_model_paths = os.listdir(attack_model_path_dir)

        # In case the number of trials doesn't match the # of metaclassifiers
        for (t, attack_model_path) in zip(range(attack_config.tries), attack_model_paths):
            if attack_model_path == 'victim_local':
                continue

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
            preds_adv_on_1, preds_vic_on_1, ground_truth_1, not_using_logits = get_vic_adv_preds_on_distr(
                models_vic=(models_vic_1, models_vic_2),
                models_adv=(models_adv_1, models_adv_2),
                ds_obj=ds_adv_1,
                batch_size=bb_attack_config.batch_size,
                epochwise_version=attack_config.train_config.save_every_epoch,
                preload=bb_attack_config.preload,
                multi_class=bb_attack_config.multi_class,
                make_processed_version=attack_config.adv_processed_variant
            )
            # Get victim and adv predictions on loaders for second ratio
            preds_adv_on_2, preds_vic_on_2, ground_truth_2, _ = get_vic_adv_preds_on_distr(
                models_vic=(models_vic_1, models_vic_2),
                models_adv=(models_adv_1, models_adv_2),
                ds_obj=ds_adv_2,
                batch_size=bb_attack_config.batch_size,
                epochwise_version=attack_config.train_config.save_every_epoch,
                preload=bb_attack_config.preload,
                multi_class=bb_attack_config.multi_class,
                make_processed_version=attack_config.adv_processed_variant
            )

            # Create split on adv models
            adv_for_bb_index, adv_for_meta_index = train_test_split(list(range(len(models_adv_1))), test_size=attack_config.num_for_meta)

            # Create new PredictionsOnOneDistribution objects
            preds_adv_1_bb = PredictionsOnOneDistribution(
                preds_property_1=preds_adv_on_1.preds_property_1[adv_for_bb_index],
                preds_property_2=preds_adv_on_1.preds_property_2[adv_for_bb_index]
            )
            preds_adv_2_bb = PredictionsOnOneDistribution(
                preds_property_1=preds_adv_on_2.preds_property_1[adv_for_bb_index],
                preds_property_2=preds_adv_on_2.preds_property_2[adv_for_bb_index]
            )
            # Combine remaining BB preds with vic preds, treat as one
            preds_adv_1_combined = PredictionsOnOneDistribution(
                preds_property_1=np.concatenate((preds_adv_on_1.preds_property_1[adv_for_meta_index], preds_vic_on_1.preds_property_1), 0),
                preds_property_2=np.concatenate((preds_adv_on_1.preds_property_2[adv_for_meta_index], preds_vic_on_1.preds_property_2), 0)
            )
            preds_adv_2_combined = PredictionsOnOneDistribution(
                preds_property_1=np.concatenate((preds_adv_on_2.preds_property_1[adv_for_meta_index], preds_vic_on_2.preds_property_1), 0),
                preds_property_2=np.concatenate((preds_adv_on_2.preds_property_2[adv_for_meta_index], preds_vic_on_2.preds_property_2), 0)
            )
            
            # One for use for BB attack, other to get preds for meta-classifier
            bbm_preds_adv_for_bb = PredictionsOnDistributions(
                preds_on_distr_1=preds_adv_1_bb,
                preds_on_distr_2=preds_adv_2_bb,
            )
            split_point = len(adv_for_meta_index)
            # Wrap predictions to be used by the attack
            bbm_and_vic_preds = PredictionsOnDistributions(
                preds_on_distr_1=preds_adv_1_combined,
                preds_on_distr_2=preds_adv_2_combined
            )

            # Actually only KL
            # Create attacker object
            attacker_obj = get_attack(bb_attack_config.attack_type[0])(bb_attack_config)

            # Launch attack
            result = attacker_obj.attack(
                bbm_preds_adv_for_bb, bbm_and_vic_preds,
                ground_truth=None,
                calc_acc=calculate_accuracies,
                epochwise_version=attack_config.train_config.save_every_epoch,
                not_using_logits=not_using_logits)
                
            # From this result, extract all preds
            # And split into actual victim preds and adv preds for holdout

            # Extract predictions for bb holdout and vic from this
            bb_and_vic_preds = result[0][1]
            bb_and_vic_preds_1, bb_and_vic_preds_2 = np.split(result[0][1], 2)
            bb_meta_preds = np.concatenate((bb_and_vic_preds_1[:split_point], bb_and_vic_preds_2[:split_point]), 0)
            bb_preds_vic =  np.concatenate((bb_and_vic_preds_1[split_point:], bb_and_vic_preds_2[split_point:]), 0)
        
            # Now, come onto white-box preds
            attacker_obj = wu.get_attack(wb_attack_config.attack)(
                None, wb_attack_config)
            
            # Load model
            attacker_obj.load_model(os.path.join(
                attack_model_path_dir, attack_model_path))

            # Convert saved seed-data (from AMC meta-clf) into loader for processing            
            seed_data_loader = get_loader_for_seed_data(
                attacker_obj.seed_data_ds, wb_attack_config)

            adv_test = wrap_into_loader(
                [models_adv_1[adv_for_meta_index], models_adv_2[adv_for_meta_index]],
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                wrap_with_loader=False
            )
            vic_test = wrap_into_loader(
                [models_vic_1, models_vic_2],
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                wrap_with_loader=False
            )
            # Make affinity features for train (adv) models
            features_adv = attacker_obj.make_affinity_features(
                adv_test[0], seed_data_loader, labels=adv_test[1])
            # Make affinity features for victim models
            features_vic = attacker_obj.make_affinity_features(
                vic_test[0], seed_data_loader)

            wb_preds_adv = attacker_obj.eval_attack(
                test_loader=(features_adv, adv_test[1]),
                epochwise_version=attack_config.train_config.save_every_epoch,
                get_preds=True,
                get_latents=attack_config.use_wb_latents)

            wb_preds_vic = attacker_obj.eval_attack(
                test_loader=(features_vic, vic_test[1]),
                epochwise_version=attack_config.train_config.save_every_epoch,
                get_preds=True,
                get_latents=attack_config.use_wb_latents)
            
            def _expand_if_needed(x_):
                if type(x_) != np.ndarray or  len(x_.shape) == 1:
                    return np.expand_dims(x_, axis=1)
                return x_

            #decision tree
            preds_adv = np.concatenate(
                (_expand_if_needed(wb_preds_adv), _expand_if_needed(bb_meta_preds)), 1)
            preds_vic = np.concatenate(
                (_expand_if_needed(wb_preds_vic), _expand_if_needed(bb_preds_vic)), 1)

            clf = DecisionTreeClassifier(max_depth=2)
            labels_adv = np.concatenate(
                ([0] * len(adv_for_meta_index), [1] * len(adv_for_meta_index)))
            labels_vic = np.concatenate(
                ([0] * len(models_vic_1), [1] * len(models_vic_2)))
            clf.fit(preds_adv, labels_adv)
            logger.add_results("AGA+KL", prop_value,
                               100 * clf.score(preds_vic, labels_vic),
                               100 * clf.score(preds_adv, labels_adv))
            # Summarize results over runs, for each ratio and attack
            # Save after each run (in case scripts crash)
            logger.save()
