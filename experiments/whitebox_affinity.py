from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import wrap_into_loader, get_attack, get_train_val_from_pool
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils, warning_string
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.whitebox.affinity.utils import get_seed_data_loader, identify_relevant_points, make_ds_and_loader


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
    # Do the same if adv_misc_config is present
    if attack_config.adv_misc_config is not None:
        if attack_config.adv_misc_config.adv_config:
            if attack_config.adv_misc_config.adv_config.scale_by_255:
                attack_config.adv_misc_config.adv_config.epsilon /= 255

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
    ds_vic_1 = ds_wrapper_class(
        data_config_victim_1, skip_data=not attack_config.victim_local_attack)
    if not attack_config.victim_local_attack:
        ds_adv_1 = ds_wrapper_class(data_config_adv_1)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        full_model=attack_config.victim_full_model)

    # For each value (of property) asked to experiment with
    for prop_value in attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_vic_2 = ds_wrapper_class(
            data_config_vic_2, skip_data=not attack_config.victim_local_attack)
        if not attack_config.victim_local_attack:
            ds_adv_2 = ds_wrapper_class(data_config_adv_2)

        # Load victim's model features for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            full_model=attack_config.victim_full_model)

        # Generate test set unless victim-only mode
        # In that case, 'val' data is test data
        if not attack_config.victim_local_attack:
            test_data = wrap_into_loader(
                [models_vic_1, models_vic_2],
                batch_size=wb_attack_config.batch_size,
                shuffle=False,
                wrap_with_loader=False
            )

        # Load adv models for both ratios
        # Unless victim-only mode
        if not attack_config.victim_local_attack:
            models_adv_1 = ds_adv_1.get_models(
                train_config_adv,
                n_models=attack_config.num_total_adv_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)

            models_adv_2 = ds_adv_2.get_models(
                train_config_adv,
                n_models=attack_config.num_total_adv_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)

        for _ in range(attack_config.tries):

            # Prepare train, val data
            if attack_config.victim_local_attack:
                # Split victim models into train-val
                train_data, val_data = get_train_val_from_pool(
                    [models_vic_1, models_vic_2],
                    wb_config=wb_attack_config,
                    wrap_with_loader=False
                )
                test_data = val_data
            else:
                # Normal train-val split from adv models
                train_data, val_data = get_train_val_from_pool(
                    [models_adv_1, models_adv_2],
                    wb_config=wb_attack_config,
                    wrap_with_loader=False
                )

            # Create attacker object
            attacker_obj = get_attack(wb_attack_config.attack)(
                None, wb_attack_config)

            # Decide which models and DS to use
            models_1_use = models_vic_1 if attack_config.victim_local_attack else models_adv_1
            models_2_use = models_vic_2 if attack_config.victim_local_attack else models_adv_2
            ds_use = [ds_vic_1, ds_vic_2] if attack_config.victim_local_attack else [ds_adv_1, ds_adv_2]

            # Get seed-data
            if wb_attack_config.affinity_config.perpoint_based_selection > 0:
                if wb_attack_config.affinity_config.optimal_data_identity:
                    raise NotImplementedError("optimal_data_identity not supported with per-point selection yet")

                print(warning_string("Using Per-Point criteria for seed-data selection"))
                # Take a random sample of adv models
                models_1_sample = np.random.choice(
                    models_1_use,
                    wb_attack_config.affinity_config.perpoint_based_selection,
                    replace=False)
                models_2_sample = np.random.choice(
                    models_2_use,
                    wb_attack_config.affinity_config.perpoint_based_selection,
                    replace=False)
                # Get seed-data
                seed_data_ds, seed_data_loader = get_seed_data_loader(
                    ds_use, wb_attack_config,
                    num_samples_use=wb_attack_config.affinity_config.num_samples_use,
                    adv_models=[models_1_sample, models_2_sample])
            else:
                # Identify optimal data to use (for features) using heuristic
                if wb_attack_config.affinity_config.optimal_data_identity:
                    # Collect all data
                    seed_data_all_ds, seed_data_all_loader, seed_data_all = get_seed_data_loader(
                        ds_use, wb_attack_config,
                        also_get_raw_data=True)
                    # Take random samples from both models
                    models_1_sample = np.random.choice(
                        models_1_use,
                        wb_attack_config.affinity_config.model_sample_for_optimal_data_identity,
                        replace=False)
                    models_2_sample = np.random.choice(
                        models_2_use,
                        wb_attack_config.affinity_config.model_sample_for_optimal_data_identity,
                        replace=False)
                    models_sample = np.concatenate(
                        (models_1_sample, models_2_sample), axis=0, dtype=object)
                    # Use those to determine 'optimal' data
                    all_features = attacker_obj.make_affinity_features(
                        models_sample, seed_data_all_loader,
                        return_raw_features=True)
                    wanted_ids = identify_relevant_points(
                        all_features,
                        len(seed_data_all_ds),
                        wb_attack_config.affinity_config.num_samples_use,
                        flip_selection_logic=wb_attack_config.affinity_config.flip_selection_logic)
                    # Create loaders corresponding to these datapoints
                    seed_data_ds, seed_data_loader = make_ds_and_loader(
                        seed_data_all, wb_attack_config, wanted_ids)
                else:
                    seed_data_ds, seed_data_loader = get_seed_data_loader(
                        ds_use, wb_attack_config,
                        num_samples_use=wb_attack_config.affinity_config.num_samples_use)

            # Save seed-data in attack object, since this is needed
            # To use model later in evaluation mode, if loaded from memory
            attacker_obj.register_seed_data(seed_data_ds)

            # Make affinity features for train (adv) models
            features_train = attacker_obj.make_affinity_features(
                train_data[0], seed_data_loader, labels=train_data[1])
            # Make affinity features for victim models
            features_test = attacker_obj.make_affinity_features(
                test_data[0], seed_data_loader)
            # Make affinity features for val (adv) models, if requested
            if val_data is not None:
                if attack_config.victim_local_attack:
                    # Val is same as test
                    features_val = features_test
                else:
                    features_val = attacker_obj.make_affinity_features(
                        val_data[0], seed_data_loader)

            # Execute attack
            chosen_accuracy = attacker_obj.execute_attack(
                train_data=(features_train, train_data[1]),
                test_data=(features_test, test_data[1]),
                val_data=(features_val, val_data[1]))

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
