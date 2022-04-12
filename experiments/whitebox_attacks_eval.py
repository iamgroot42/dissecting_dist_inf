"""
    Script for executing white-box inference attacks.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import warnings

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from distribution_inference.attacks.whitebox.utils import get_attack, wrap_into_loader
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils, warning_string
from distribution_inference.logging.core import AttackResult


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--path", help="path to trained model",
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

    if len(attack_config.values) != 1:
        warnings.warn(warning_string(
            "\nTesting one meta-classifier against multiple ratios.\n"))

    # Print out arguments
    flash_utils(attack_config)

    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    _, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True)

    # Load victim and adversary's model features for first value
    dims, features_vic_1 = ds_vic_1.get_model_features(
        train_config,
        wb_attack_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=attack_config.train_config.save_every_epoch)

    # Create attacker object
    attacker_obj = get_attack(wb_attack_config.attack)(
        dims, wb_attack_config)

    # Load model
    attacker_obj.load_model(args.path)

    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        _, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)

        # Load victim and adversary's model features for other value
        _, features_vic_2 = ds_vic_2.get_model_features(
            train_config,
            wb_attack_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            epochwise_version=attack_config.train_config.save_every_epoch)

        # Generate test set
        test_loader = wrap_into_loader(
            [features_vic_1, features_vic_2],
            batch_size=wb_attack_config.batch_size,
            shuffle=False,
            epochwise_version=attack_config.train_config.save_every_epoch
        )

        # Execute attack
        chosen_accuracy = attacker_obj.eval_attack(
            test_loader=test_loader,
            epochwise_version=attack_config.train_config.save_every_epoch)

        if not attack_config.train_config.save_every_epoch:
            print("Test accuracy: %.3f" % chosen_accuracy)
        logger.add_results(wb_attack_config.attack,
                           prop_value, chosen_accuracy, None)

    # Save logger results
    logger.save()
