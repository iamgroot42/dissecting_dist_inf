"""
    Script for single uodate loss attack.
"""
import numpy as np
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_preds_epoch_on_dis
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv,get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult


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
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(attack_config)
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)(epoch_wise=True)

    # Create new DS object for both and victim
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True, epoch=True)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1,epoch=True)
    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=True)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2,epoch=True)
        ds_vic_2 = ds_wrapper_class(
            data_config_vic_2, skip_data=True, epoch=True)
        # Load victim models for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            epochwise_version=True)

        for t in range(attack_config.tries):
            print("Ratio: {}, Trial: {}".format(prop_value, t))
            
            _, loader1 = ds_adv_1.get_loaders(
                batch_size=bb_attack_config.batch_size)
            _, loader2 = ds_adv_2.get_loaders(
                batch_size=bb_attack_config.batch_size)
            models_adv_1 = ds_adv_1.get_models(
                train_adv_config,
                n_models=bb_attack_config.num_adv_models,
                on_cpu=attack_config.on_cpu,
                epochwise_version=True)
            
            models_adv_2 = ds_adv_2.get_models(
                train_adv_config,
                n_models=bb_attack_config.num_adv_models,
                on_cpu=attack_config.on_cpu,
                epochwise_version=True)
            preds_vepoch_1, ground_truth_1 = get_preds_epoch_on_dis([models_vic_1, models_vic_2],
                                                                   loader=loader1, preload=bb_attack_config.preload,
                                                                   multi_class=bb_attack_config.multi_class)
            preds_vepoch_2, ground_truth_2 = get_preds_epoch_on_dis([models_vic_1, models_vic_2],
                                                                   loader=loader2, preload=bb_attack_config.preload,
                                                                   multi_class=bb_attack_config.multi_class)
            preds_aepoch_1, g1 = get_preds_epoch_on_dis([models_adv_1, models_adv_2],
                                                                   loader=loader1, preload=bb_attack_config.preload,
                                                                   multi_class=bb_attack_config.multi_class)
            preds_aepoch_2, g2 = get_preds_epoch_on_dis([models_adv_1, models_adv_2],
                                                                   loader=loader2, preload=bb_attack_config.preload,
                                                                   multi_class=bb_attack_config.multi_class)
            assert np.array_equal(ground_truth_1,g1)
            assert np.array_equal(ground_truth_2,g2)
            preds_ve = [PredictionsOnDistributions(
                preds_on_distr_1=e1,
                preds_on_distr_2=e2
            ) for e1, e2 in zip(preds_vepoch_1, preds_vepoch_2)]
            preds_ae = [PredictionsOnDistributions(
                preds_on_distr_1=e1,
                preds_on_distr_2=e2
            ) for e1, e2 in zip(preds_aepoch_1, preds_aepoch_2)]
            for attack_type in bb_attack_config.attack_type:
                    # Create attacker object
                attacker_obj = get_attack(attack_type)(bb_attack_config)
                preds_accross_epoch = []
                for i in range(bb_attack_config.Start_epoch-1, bb_attack_config.End_epoch-5):
                    preds_ve1, preds_ve2 = preds_ve[i], preds_ve[i+5]
                    preds_ae1, preds_ae2 = preds_ae[i], preds_ae[i+5]
                

                # Launch attack
                    raw_preds = attacker_obj.attack(
                        preds_ve1, preds_ve2,preds_ae1, preds_ae2,
                        ground_truth=(ground_truth_1, ground_truth_2),
                        calc_acc=calculate_accuracies,
                        get_preds=True)
                    preds_accross_epoch.append(raw_preds)
                preds_accross_epoch = np.array(preds_accross_epoch)
                aggre_preds = np.mean(preds_accross_epoch, axis=0) >= 0.5
                result = 100*np.mean(aggre_preds)
                logger.add_results(attack_type, prop_value,
                               vacc=result)
                print("{} acc: {}".format(attack_type,result))
    # Summarize results over runs, for each ratio and attack
    logger.save()
