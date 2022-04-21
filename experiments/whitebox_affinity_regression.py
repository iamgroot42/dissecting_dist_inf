from simple_parsing import ArgumentParser
from pathlib import Path

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import get_attack, get_train_val_from_pool, wrap_into_loader
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
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Make sure regression config is not being used here
    if wb_attack_config.regression_config is None:
        raise ValueError(
            "Regression config must be provided")

    # Print out arguments
    flash_utils(attack_config)

    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv, data_config_victim = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv = ds_wrapper_class(data_config_adv)
    ds_vic = ds_wrapper_class(data_config_victim, skip_data=True)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Loading up all models altogether gives oom for system
    # Have to load all models from scratch again, but only
    # the ones needed; even victim models need to be
    # re-loaded per trial
    for _ in range(attack_config.tries):

        # Get seed-data DS objects
        models_adv_all, models_vic_all, all_ds = [], [], []
        for prop_value in attack_config.values:
            data_config_adv_specific, data_config_vic_specific = get_dfs_for_victim_and_adv(
                data_config, prop_value=prop_value)
            
            # Create new DS object for both and victim (for other ratio)
            ds_adv_specific = ds_wrapper_class(
                data_config_adv_specific)
            ds_vic_specific = ds_wrapper_class(
                data_config_vic_specific, skip_data=True)
            all_ds.append(ds_adv_specific)

            # Load up models for each of the values
            models_adv = ds_adv_specific.get_models(
                train_config_adv,
                n_models=attack_config.num_total_adv_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)
            models_vic = ds_vic_specific.get_models(
                train_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)
            models_adv_all.append(models_adv)
            models_vic_all.append(models_vic)
        
        # Look at any additional requested ratios
        additional_values = None
        if wb_attack_config.regression_config.additional_values_to_test:
            additional_values = wb_attack_config.regression_config.additional_values_to_test
        if additional_values:
            for prop_value in additional_values:
                # Creata a copy of the data config, with the property value
                # changed to the current value
                _, data_config_vic_specific = get_dfs_for_victim_and_adv(
                    data_config, prop_value=prop_value)

                # Create new DS object for victim (for other ratio)
                ds_vic_specific = ds_wrapper_class(
                    data_config_vic_specific, skip_data=True)

                models_vic = ds_vic_specific.get_models(
                    train_config,
                    n_models=attack_config.num_victim_models,
                    on_cpu=attack_config.on_cpu,
                    shuffle=False)
                models_vic_all.append(models_vic)
        
        # Generate all the seed data
        seed_data_ds, seed_data_loader = get_seed_data_loader(
            all_ds,
            wb_attack_config,
            num_samples_use=wb_attack_config.affinity_config.num_samples_use)

        # Wrap into train and test data
        base_labels = [float(x) for x in attack_config.values]

        # Wrap into test data
        test_labels = base_labels
        if additional_values:
            test_labels += [float(x) for x in additional_values]

        # Generate test set
        test_data = wrap_into_loader(
            models_vic_all,
            batch_size=wb_attack_config.batch_size,
            labels_list=test_labels,
            shuffle=False,
            wrap_with_loader=False
        )

        # Split into train, val models
        train_data, val_data = get_train_val_from_pool(
            models_adv_all,
            wb_config=wb_attack_config,
            labels_list=base_labels,
            wrap_with_loader=False
        )
        
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

        # Create attacker object
        attacker_obj = get_attack(wb_attack_config.attack)(
            dims, wb_attack_config)

        # Execute attack
        chosen_mse = attacker_obj.execute_attack(
            train_data=(features_train, train_data[1]),
            # TODO: Use val data as well
            # val_data = (features_val, val_data[1]),
            test_data=(features_test, test_data[1]),)

        print("Test MSE: %.3f" % chosen_mse)
        logger.add_results(wb_attack_config.attack,
                           "regression", chosen_mse, None)

        # Save attack parameters if requested
        if wb_attack_config.save:
            attacker_obj.save_model(
                data_config,
                attack_specific_info_string=str(chosen_mse))

        # Cleanup: delete features, etc before next trial begins
        del features_train, features_test
        del train_data, test_data
        del attacker_obj

    # Save logger results
    logger.save()
