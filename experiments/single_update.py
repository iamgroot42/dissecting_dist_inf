"""
    Script for single uodate loss attack.
"""
import numpy as np
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
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
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True,epoch=True)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=True)
    models_vic_e1_1 = models_vic_1[bb_attack_config.Start_epoch-1]
    models_vic_e2_1 = models_vic_1[bb_attack_config.End_epoch-1]
    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)
        
        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True,epoch=True)
        # Load victim models for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            epochwise_version=True)
        models_vic_e1_2 = models_vic_2[bb_attack_config.Start_epoch-1]
        models_vic_e2_2 = models_vic_2[bb_attack_config.End_epoch-1]
        for t in range(attack_config.tries):
            print("Ratio: {}, Trial: {}".format(prop_value,t))
           
            # Get victim predictions on loaders for first ratio

            preds_e1_1, preds_e2_1, ground_truth_1 = get_vic_adv_preds_on_distr(
                models_vic=(models_vic_e1_1, models_vic_e1_2),
                models_adv=(models_vic_e2_1, models_vic_e2_1),
                ds_obj=ds_adv_1,
                batch_size=bb_attack_config.batch_size,
                epochwise_version=False,
                preload=bb_attack_config.preload,
                multi_class=bb_attack_config.multi_class
            )
            # Get victim and adv predictions on loaders for second ratio
            preds_e1_2, preds_e2_2, ground_truth_2 = get_vic_adv_preds_on_distr(
                models_vic=(models_vic_e1_1, models_vic_e1_2),
                models_adv=(models_vic_e2_1, models_vic_e2_1),
                ds_obj=ds_adv_2,
                batch_size=bb_attack_config.batch_size,
                epochwise_version=False,
                preload=bb_attack_config.preload,
                multi_class=bb_attack_config.multi_class
            )
            # Wrap predictions to be used by the attack
            preds_e1 = PredictionsOnDistributions(
                preds_on_distr_1=preds_e1_1,
                preds_on_distr_2=preds_e1_2
            )
            preds_e2 = PredictionsOnDistributions(
                preds_on_distr_1=preds_e2_1,
                preds_on_distr_2=preds_e2_2
            )
            assert np.array_equal(ground_truth_1,ground_truth_2)
            # TODO: Need a better (and more modular way) to handle
            # the redundant code above.

            # For each requested attack, only one in this script
            for attack_type in bb_attack_config.attack_type:
                # Create attacker object
                attacker_obj = get_attack(attack_type)(bb_attack_config)
                
                # Launch attack
                result = attacker_obj.attack(
                    preds_e1, preds_e2,
                    ground_truth=ground_truth_1,
                    calc_acc=calculate_accuracies)

                logger.add_results(attack_type, prop_value,
                                   vacc=result)

                # Save predictions, if requested
                if bb_attack_config.save and attacker_obj.supports_saving_preds:
                    save_dic = attacker_obj.wrap_preds_to_save(result)

    # Summarize results over runs, for each ratio and attack
    logger.save()
