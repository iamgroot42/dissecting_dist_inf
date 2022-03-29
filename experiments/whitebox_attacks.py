

from simple_parsing import ArgumentParser
from pathlib import Path
from dataclasses import replace
from copy import deepcopy

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import wrap_into_x_y, get_attack, get_train_val_from_pool
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    wb_attack_config: WhiteBoxAttackConfig = attack_config.white_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config

    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")

    # Print out arguments
    flash_utils(attack_config)
    # TODO: Path below should be inferred from dataset, not fixed here in file
    # Same for the 'deepcopy' - maybe shift to the constructor itself?
    logger = AttackResult(Path("./log/new_census"),
                          args.en, deepcopy(attack_config))
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_victim_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1, skip_data=True)
    ds_vic_1 = ds_wrapper_class(data_config_victim_1, skip_data=True)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load victim and adversary's model features for first value
    _, features_vic_1 = ds_vic_1.get_model_features(
        train_config,
        wb_attack_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False)

    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        data_config_other = replace(data_config)
        data_config_other.value = prop_value
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config_other)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2, skip_data=True)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)

        # Load victim and adversary's model features for other value
        _, features_vic_2 = ds_vic_2.get_model_features(
            train_config,
            wb_attack_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)

        # Generate test set
        test_data = wrap_into_x_y(
            [features_vic_1, features_vic_2])

        for _ in range(attack_config.tries):
            # Create attacker object
            dims, features_adv_1 = ds_adv_1.get_model_features(
                train_config_adv,
                wb_attack_config,
                n_models=attack_config.num_total_adv_models,
                on_cpu=attack_config.on_cpu,
                shuffle=True)
            attacker_obj = get_attack(wb_attack_config.attack)(dims, wb_attack_config)
            _, features_adv_2 = ds_adv_2.get_model_features(
                train_config_adv,
                wb_attack_config,
                n_models=attack_config.num_total_adv_models,
                on_cpu=attack_config.on_cpu,
                shuffle=True)
            # Prepare train, val data
            train_data, val_data = get_train_val_from_pool(
                [features_adv_1, features_adv_2],
                wb_config=wb_attack_config,
            )

            # Execute attack
            chosen_accuracy = attacker_obj.execute_attack(
                train_data=train_data,
                test_data=test_data,
                val_data=val_data)

            print("Test accuracy: %.3f" % chosen_accuracy)
            logger.add_results(wb_attack_config.attack,
                               prop_value, chosen_accuracy, None)
    logger.save()
