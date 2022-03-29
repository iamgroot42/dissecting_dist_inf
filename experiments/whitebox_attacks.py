

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
    logger = AttackResult(Path("./log/new_census"),args.en,deepcopy(attack_config))
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object for both and victim
    data_config_adv_1, data_config_victim_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1)
    ds_vic_1 = ds_wrapper_class(data_config_victim_1)

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
        ds_adv_2 = ds_wrapper_class(data_config_adv_2)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2)

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
            

            #if attack_config.save:
            #    attacker_obj.save_model()
            #     save_path = os.path.join(BASE_MODELS_DIR, args.filter, "meta_model", "-".join(
            #         [args.d_0, str(args.start_n), str(args.first_n)]), tg)
            #     if not os.path.isdir(save_path):
            #         os.makedirs(save_path)
            #     save_model(clf, os.path.join(save_path, str(i)+
            # "_%.2f" % tacc))
            logger.add_results(wb_attack_config.attack,prop_value,chosen_accuracy,None)
    logger.save()
    # TODO: Implement logging
    # Print data
    # log_path = os.path.join(BASE_MODELS_DIR, args.filter, "meta_result")

    # if args.scale != 1.0:
    #     log_path = os.path.join(log_path,"sample_size_scale:{}".format(args.scale))

    # if args.drop:
    #     log_path = os.path.join(log_path,'drop')
    # utils.ensure_dir_exists(log_path)
    # with open(os.path.join(log_path, "-".join([args.filter, args.d_0, str(args.start_n), str(args.first_n)])), "a") as wr:
    #     for i, tup in enumerate(data):
    #         print(targets[i], tup)
    #         wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
