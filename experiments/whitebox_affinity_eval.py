"""
    Script for executing white-box inference attacks.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import warnings
import os

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from distribution_inference.attacks.whitebox.utils import get_attack, wrap_into_loader
from distribution_inference.attacks.whitebox.affinity.utils import get_loader_for_seed_data
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
        "--meta_path", help="path to trained meta-models directory",
        type=str, required=True)
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
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

    # Make sure a model is present for each value to be tested
    values_to_test = [str(x) for x in attack_config.values]
    folders = os.listdir(args.meta_path)
    for value in values_to_test:
        if value not in folders:
            raise ValueError(
                f"No model found for value {value} in {args.meta_path}")

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

    def single_evaluation(models_1_path=None, models_2_paths=None):

        # Load victim's model for first value
        models_vic_1 = ds_vic_1.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            full_model=attack_config.victim_full_model,
            custom_models_path=models_1_path)

        # For each value (of property) asked to experiment with
        for i, prop_value in enumerate(attack_config.values):
            # Creata a copy of the data config, with the property value
            # changed to the current value
            _, data_config_vic_2 = get_dfs_for_victim_and_adv(
                data_config, prop_value=prop_value)

            # Create new DS object for both and victim (for other ratio)
            ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)

            # Load victim's model for other value
            models_vic_2 = ds_vic_2.get_models(
                train_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                full_model=attack_config.victim_full_model,
                custom_models_path=models_2_paths[i] if models_2_paths else None)

            # Generate test set
            test_data = wrap_into_loader(
                [models_vic_1, models_vic_2],
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                wrap_with_loader=False
            )

            # Look at all models
            attack_model_path_dir = os.path.join(
                args.meta_path, str(prop_value))
            for attack_model_path in os.listdir(attack_model_path_dir):
                # Skip if directory
                if os.path.isdir(os.path.join(attack_model_path_dir, attack_model_path)):
                    continue

                # Create attacker object
                attacker_obj = get_attack(wb_attack_config.attack)(
                    None, wb_attack_config)

                # Load model
                attacker_obj.load_model(os.path.join(
                    attack_model_path_dir, attack_model_path))

                # Make seed-data loader for this attack
                seed_data_loader = get_loader_for_seed_data(
                    attacker_obj.seed_data_ds, wb_attack_config)
                # Create affinity features
                features_test = attacker_obj.make_affinity_features(
                    test_data[0], seed_data_loader)

                # Execute attack
                chosen_accuracy = attacker_obj.eval_attack(
                    test_loader=(features_test, test_data[1]))

                if not attack_config.train_config.save_every_epoch:
                    print("Test accuracy: %.3f" % chosen_accuracy)
                logger.add_results(wb_attack_config.attack,
                                   prop_value, chosen_accuracy, None)

    if args.victim_path:
        def joinpath(x, y): return os.path.join(
            args.victim_path, str(x), str(y))
        for i in range(1, 3+1):
            models_1_path = joinpath(data_config.value, i)
            model_2_paths = [joinpath(v, i) for v in attack_config.values]
            single_evaluation(models_1_path, model_2_paths)
    else:
        single_evaluation()

    # Save logger results
    logger.save()
