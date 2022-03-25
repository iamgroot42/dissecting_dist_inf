from distribution_inference.attacks import blackbox
from pyparsing import AtStringStart
from simple_parsing import ArgumentParser
from pathlib import Path
from dataclasses import replace

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, get_preds_for_vic_and_adv
from distribution_inference.attacks.blackbox.core import PredictionsOnOneDistribution, PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteboxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils



if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--load_config", help="Specify config file", type=Path)
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
    wb_attack_config: WhiteboxAttackConfig = attack_config.white_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_configz

    # Make sure regression config is not being used here
    if wb_attack_config.regression_config:
        raise ValueError(
            "This script is not designed to be used with regression attacks")

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

    # Load victim and adversary's models for first value
    features_adv_1 = ds_adv_1.get_model_features(
            train_config,
            wb_attack_config,
            n_models=wb_attack_config.num_adv_models,
            on_cpu=attack_config.on_cpu,
            shuffle=True)
    features_vic_1 = ds_vic_1.get_model_features(
        train_config,
        wb_attack_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False)

    data = []

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

        # Load victim and adversary's models for other value
        features_adv_2 = ds_adv_2.get_models(
            train_config,
            wb_attack_config,
            n_models=wb_attack_config.num_adv_models,
            on_cpu=attack_config.on_cpu,
            shuffle=True)
        features_vic_2 = ds_vic_2.get_models(
            train_config,
            wb_attack_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)

        # TODO: Pick up from here


        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))
        Y_te = ch.cat((pos_labels_test, neg_labels_test)).cuda()

        print("Batching data: hold on")
        X_te = utils.prepare_batched_data(X_te)

        for i in range(args.ntimes):
            # Random shuffles
            shuffled_1 = np.random.permutation(len(pos_labels))
            pp_x = pos_w[shuffled_1[:args.train_sample]]
            pp_y = pos_labels[shuffled_1[:args.train_sample]]

            shuffled_2 = np.random.permutation(len(neg_labels))
            np_x = neg_w[shuffled_2[:args.train_sample]]
            np_y = neg_labels[shuffled_2[:args.train_sample]]

            # Combine them together
            X_tr = np.concatenate((pp_x, np_x))
            Y_tr = ch.cat((pp_y, np_y))

            val_data = None
            if args.val_sample > 0:
                pp_val_x = pos_w[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_x = neg_w[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                pp_val_y = pos_labels[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_y = neg_labels[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                # Combine them together
                X_val = np.concatenate((pp_val_x, np_val_x))
                Y_val = ch.cat((pp_val_y, np_val_y))

                # Batch layer-wise inputs
                print("Batching data: hold on")
                X_val = utils.prepare_batched_data(X_val)
                Y_val = Y_val.float()

                val_data = (X_val, Y_val)

            metamodel = utils.PermInvModel(dims, dropout=0.5)
            metamodel = metamodel.cuda()
            metamodel = ch.nn.DataParallel(metamodel)

            # Float data
            Y_tr = Y_tr.float()
            Y_te = Y_te.float()

            # Batch layer-wise inputs
            print("Batching data: hold on")
            X_tr = utils.prepare_batched_data(X_tr)

            # Train PIM
            clf, tacc = utils.train_meta_model(
                         metamodel,
                         (X_tr, Y_tr), (X_te, Y_te),
                         epochs=epoch_strategy(tg, args),
                         binary=True, lr=1e-3,
                         regression=False,
                         batch_size=args.batch_size,
                         val_data=val_data, combined=True,
                         eval_every=10, gpu=True)
            if args.save:
                save_path = os.path.join(BASE_MODELS_DIR, args.filter, "meta_model", "-".join(
                    [args.d_0, str(args.start_n), str(args.first_n)]), tg)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                save_model(clf, os.path.join(save_path, str(i)+
            "_%.2f" % tacc))
            tgt_data.append(tacc)
            print("Test accuracy: %.3f" % tacc)
        data.append(tgt_data)

    # Print data
    log_path = os.path.join(BASE_MODELS_DIR, args.filter, "meta_result")

    if args.scale != 1.0:
        log_path = os.path.join(log_path,"sample_size_scale:{}".format(args.scale))

    if args.drop:
        log_path = os.path.join(log_path,'drop')
    utils.ensure_dir_exists(log_path)
    with open(os.path.join(log_path, "-".join([args.filter, args.d_0, str(args.start_n), str(args.first_n)])), "a") as wr:
        for i, tup in enumerate(data):
            print(targets[i], tup)
            wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
