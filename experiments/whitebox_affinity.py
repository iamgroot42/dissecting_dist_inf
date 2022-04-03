from simple_parsing import ArgumentParser
from pathlib import Path
from dataclasses import replace

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import wrap_into_loader, get_attack, get_train_val_from_pool
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.whitebox.affinity.utils import get_seed_data_loader


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
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

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
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_victim_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    ds_vic_1 = ds_wrapper_class(data_config_victim_1, skip_data=True)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False)

    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)

        # Load victim's model features for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)

        # Generate test set
        test_data = wrap_into_loader(
            [models_vic_1, models_vic_2],
            batch_size=wb_attack_config.batch_size,
            shuffle=False,
            wrap_with_loader=False
        )

        for _ in range(attack_config.tries):
            # Load adv models
            models_adv_1 = ds_vic_1.get_models(
                train_config_adv,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)

            models_adv_2 = ds_vic_2.get_models(
                train_config_adv,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)

            # Split into train, val models
            train_data, val_data = get_train_val_from_pool(
                [models_adv_1, models_adv_2],
                wb_config=wb_attack_config,
                wrap_with_loader=False
            )

            # Get seed-data
            seed_data_loader = get_seed_data_loader(
                [ds_adv_1, ds_adv_2],
                wb_attack_config,
                num_samples_use=wb_attack_config.affinity_config.num_samples_use)

            # Create attacker object
            attacker_obj = get_attack(wb_attack_config.attack)(
                wb_attack_config)

            # Make affinity features for train (adv) models
            features_train = attacker_obj.make_affinity_features(
                train_data[0], seed_data_loader)
            # Make affinity features for victim models
            features_test = attacker_obj.make_affinity_features(
                test_data[0], seed_data_loader)
            # Make affinity features for val (adv) models, if requested
            if val_data is not None:
                features_val = attacker_obj.make_affinity_features(
                    val_data[0], seed_data_loader)

            # Execute attack
            chosen_accuracy = attacker_obj.execute_attack(
                train_data=(features_train, train_data[1]),
                # TODO: Use val data as well
                # val_data = (features_val, val_data[1]),
                test_data=(features_test, test_data[1]),)

            print("Test accuracy: %.3f" % chosen_accuracy)
            logger.add_results(wb_attack_config.attack,
                               prop_value, chosen_accuracy, None)

            # Save attack parameters if requested
            if wb_attack_config.save:
                attacker_obj.save_model(
                    data_config_vic_2,
                    attack_specific_info_string=str(chosen_accuracy))

    # Save logger results
    logger.save()
