"""
    Script for executing black-box inference attacks.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
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
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True)
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=attack_config.train_config.save_every_epoch)

    # For each value (of property) asked to experiment with
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
            shuffle=False,
            epochwise_version=attack_config.train_config.save_every_epoch)
        for _ in range(attack_config.tries):
            models_adv_1 = ds_adv_1.get_models(
                train_adv_config,
                n_models=bb_attack_config.num_adv_models,
                on_cpu=attack_config.on_cpu)
            models_adv_2 = ds_adv_2.get_models(
                train_adv_config,
                n_models=bb_attack_config.num_adv_models,
                on_cpu=attack_config.on_cpu)
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
            preds_adv = PredictionsOnDistributions(
                preds_on_distr_1=preds_adv_on_1,
                preds_on_distr_2=preds_adv_on_2
            )
            preds_vic = PredictionsOnDistributions(
                preds_on_distr_1=preds_vic_on_1,
                preds_on_distr_2=preds_vic_on_2
            )

            # TODO: Need a better (and more modular way) to handle
            # the redundant code above.

            # For each requested attack
            for attack_type in bb_attack_config.attack_type:
                # Create attacker object
                attacker_obj = get_attack(attack_type)(bb_attack_config)

                # Launch attack
                result = attacker_obj.attack(
                    preds_adv, preds_vic,
                    ground_truth=(ground_truth_1, ground_truth_2),
                    calc_acc=calculate_accuracies,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    multi=bb_attack_config.multi,
                    multi2=bb_attack_config.multi2)

                logger.add_results(attack_type, prop_value,
                                   result[0][0], result[1][0])

    # Summarize results over runs, for each ratio and attack
    logger.save()
