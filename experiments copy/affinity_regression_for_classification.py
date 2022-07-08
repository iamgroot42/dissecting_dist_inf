"""
    Load trained regression model and use for binary prediction of properties.
"""
from simple_parsing import ArgumentParser
from pathlib import Path

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import get_attack, wrap_into_loader, eval_regression_preds_for_binary
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.attacks.whitebox.affinity.utils import get_loader_for_seed_data
from distribution_inference.utils import flash_utils


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--path", help="path to trained regression model",
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

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    _, data_config_victim = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic = ds_wrapper_class(data_config_victim, skip_data=True)

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Load up model features for each of the values
    collected_models_test = []
    for prop_value in attack_config.values:
        _, data_config_vic_specific = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_vic_specific = ds_wrapper_class(
            data_config_vic_specific, skip_data=True)

        # Load victim's model features for other value
        models_vic_specific = ds_vic_specific.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)

        collected_models_test.append(models_vic_specific)

    # Wrap into train and test data
    test_labels = [float(x) for x in attack_config.values]

    # Generate test set
    test_data = wrap_into_loader(
        collected_models_test,
        batch_size=wb_attack_config.batch_size,
        shuffle=False,
        wrap_with_loader=False,
        labels_list=test_labels
    )

    # Create attacker object
    attacker_obj = get_attack(wb_attack_config.attack)(
        None, wb_attack_config)

    # Load regression model
    attacker_obj.load_model(args.path)

    # Make affinity features for victim models
    seed_data_loader = get_loader_for_seed_data(
        attacker_obj.seed_data_ds, wb_attack_config)
    features_test = attacker_obj.make_affinity_features(
        test_data[0], seed_data_loader)
    test_loader = (features_test, test_data[1])

    # Evaluate attack on test loader
    reression_preds = attacker_obj.eval_attack(test_loader, get_preds=True)

    # Get evaluation matrix
    pred_matrix = eval_regression_preds_for_binary(
        reression_preds, test_data[1], attack_config.values, raw=True)
    print(pred_matrix)
