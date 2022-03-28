from distribution_inference.attacks import blackbox
from simple_parsing import ArgumentParser
from pathlib import Path
from dataclasses import replace

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_preds_for_vic_and_adv
from distribution_inference.attacks.blackbox.core import PredictionsOnOneDistribution, PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv,get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--load_config", help="Specify config file", type=Path)
    parser.add_argument("--en", help="experiment name",type=str)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = None
    if args.load_config is not None:
        config = AttackConfig.load(args.load_config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(AttackConfig, dest="attack_config", default=config)
    args = parser.parse_args(remaining_argv)

    # Extract configuration information from config file
    attack_config: AttackConfig = args.attack_config
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    logger = AttackResult(Path('./log/new_census'),args.en,attack_config)
    # Print out arguments
    flash_utils(attack_config)
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_victim_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    ds_vic_1 = ds_wrapper_class(data_config_victim_1)
    train_adv_config=get_train_config_for_adv(train_config,attack_config)
    # Load victim models for first value

    models_vic_1 = ds_vic_1.get_models(train_config,
                                       n_models=attack_config.num_victim_models,
                                       on_cpu=attack_config.on_cpu)
    


    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        data_config_other = replace(data_config)
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config_other)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2)
        # Load victim models for other value

        models_vic_2 = ds_vic_2.get_models(train_config,
                                           n_models=attack_config.num_victim_models,
                                           on_cpu=attack_config.on_cpu)
        for _ in range(attack_config.tries):
            models_adv_1 = ds_adv_1.get_models(train_adv_config,
                                       n_models=bb_attack_config.num_adv_models,
                                       on_cpu=attack_config.on_cpu)
            # Get victim and adv predictions on loaders for given ratio
            preds_vic_1_on_1, preds_adv_1_on_1, ground_truth_1 = get_preds_for_vic_and_adv(
                models_vic_1, models_adv_1,
                ds_adv_1,
                batch_size=bb_attack_config.batch_size)
            models_adv_2 = ds_adv_2.get_models(train_adv_config,
                                           n_models=bb_attack_config.num_adv_models,
                                           on_cpu=attack_config.on_cpu)
            # Get victim and adv predictions on loaders for fixed ratio
            preds_vic_1_on_2, preds_adv_1_on_2, ground_truth_2 = get_preds_for_vic_and_adv(
                models_vic_1, models_adv_1,
                ds_adv_2,
                batch_size=bb_attack_config.batch_size)
            # Get victim and adv predictions on loaders for fixed ratio
            preds_vic_2_on_1, preds_adv_2_on_1, _ = get_preds_for_vic_and_adv(
                models_vic_2, models_adv_2,
                ds_adv_1, batch_size=bb_attack_config.batch_size)
            # Get victim and adv predictions on loaders for new ratio
            preds_vic_2_on_2, preds_adv_2_on_2, _ = get_preds_for_vic_and_adv(
                models_vic_2, models_adv_2,
                ds_adv_2, batch_size=bb_attack_config.batch_size)
            # Wrap predictions to be used by the attack
            preds_adv = PredictionsOnDistributions(
                preds_on_distr_1=PredictionsOnOneDistribution(
                preds_property_1=preds_adv_1_on_1,
                preds_property_2=preds_adv_2_on_1
            ),
            preds_on_distr_2=PredictionsOnOneDistribution(
                preds_property_1=preds_adv_1_on_2,
                preds_property_2=preds_adv_2_on_2
            )
            )
            preds_vic = PredictionsOnDistributions(
                preds_on_distr_1=PredictionsOnOneDistribution(
                preds_property_1=preds_vic_1_on_1,
                preds_property_2=preds_vic_2_on_1
            ),
            preds_on_distr_2=PredictionsOnOneDistribution(
                preds_property_1=preds_vic_1_on_2,
                preds_property_2=preds_vic_2_on_2
            )
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
                    calc_acc=calculate_accuracies)

                logger.add_results(attack_type,prop_value,result[0][0],result[1][0])


     # Summarize results over runs, for each ratio and attack
    logger.save()