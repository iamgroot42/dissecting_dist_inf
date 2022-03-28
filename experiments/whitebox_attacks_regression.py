# TODO: Test file

from simple_parsing import ArgumentParser
from pathlib import Path
from dataclasses import replace


from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import wrap_into_x_y, get_attack, get_train_val_from_pool
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    wb_attack_config: WhiteBoxAttackConfig = attack_config.white_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config

    # Make sure regression config is not being used here
    if wb_attack_config.regression_config is None:
        raise ValueError(
            "Regression config must be provided")

    # Print out arguments
    flash_utils(attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv, data_config_victim = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv = ds_wrapper_class(data_config_adv)
    ds_vic = ds_wrapper_class(data_config_victim)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load up model features for each of the values
    collected_features_train, collected_features_test = [], []
    for prop_value in attack_config.values:
        # Creata a copy of the data config, with the property value
        # changed to the current value
        data_config_specific = replace(data_config)
        data_config_specific.value = prop_value
        data_config_adv_specific, data_config_vic_specific = get_dfs_for_victim_and_adv(
            data_config_specific)

        # Create new DS object for both and victim (for other ratio)
        ds_adv_specific = ds_wrapper_class(data_config_adv_specific)
        ds_vic_specific = ds_wrapper_class(data_config_vic_specific)

        # Load victim and adversary's model features for other value
        dims, features_adv_specific = ds_adv_specific.get_model_features(
            train_config_adv,
            wb_attack_config,
            n_models=attack_config.num_total_adv_models,
            on_cpu=attack_config.on_cpu,
            shuffle=True)
        _, features_vic_specific = ds_vic_specific.get_model_features(
            train_config,
            wb_attack_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)

        collected_features_train.append(features_adv_specific)
        collected_features_test.append(features_vic_specific)

    # Look at any additional requested ratios
    if wb_attack_config.regression_config.additional_values_to_test is not None:
        for prop_value in wb_attack_config.regression_config.additional_values_to_test:
            # Creata a copy of the data config, with the property value
            # changed to the current value
            data_config_specific = replace(data_config)
            data_config_specific.value = prop_value
            _, data_config_vic_specific = get_dfs_for_victim_and_adv(
                data_config_specific)

            # Create new DS object for victim (for other ratio)
            ds_vic_specific = ds_wrapper_class(data_config_vic_specific)

            _, features_vic_specific = ds_vic_specific.get_model_features(
                train_config,
                wb_attack_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False)

            collected_features_test.append(features_vic_specific)

    # Wrap into train and test data
    base_labels = [float(x) for x in attack_config.values]

    # Wrap into test data
    test_labels = base_labels + \
        [float(x)
         for x in wb_attack_config.regression_config.additional_values_to_test]
    test_data = wrap_into_x_y(collected_features_test,
                              labels_list=test_labels)

    mse_vals = []
    for _ in range(attack_config.tries):
        # Create attacker object
        attacker_obj = get_attack(wb_attack_config.attack)(
            dims, wb_attack_config)

        # Prepare train, val data
        train_data, val_data = get_train_val_from_pool(
            collected_features_train,
            wb_config=wb_attack_config,
            labels_list=base_labels
        )

        # Execute attack
        chosen_mse = attacker_obj.execute_attack(
            train_data=train_data,
            test_data=test_data,
            val_data=val_data)

        print("Test MSEe: %.3f" % chosen_mse)
        mse_vals.append(chosen_mse)

        if attack_config.save:
            attacker_obj.save_model()

    print(mse_vals)

    # TODO: Implement logging
