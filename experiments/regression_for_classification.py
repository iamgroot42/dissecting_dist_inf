"""
    Load trained regression model and use for binary prediction of properties.
"""
from simple_parsing import ArgumentParser
from pathlib import Path

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.whitebox.utils import get_attack, wrap_into_loader
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

    # Define logger
    logger = AttackResult(args.en, attack_config)

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

    # Loading up all models altogether gives OOm for system
    # Have to load all models from scratch again, but only
    # the ones needed; even victim models need to be
    # re-loaded per trial

    # Load up model features for each of the values
    collected_features_train, collected_features_test = [], []
    for prop_value in attack_config.values:
        _, data_config_vic_specific = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)

        # Create new DS object for both and victim (for other ratio)
        ds_vic_specific = ds_wrapper_class(
            data_config_vic_specific, skip_data=True)

        # Load victim's model features for other value
        dims, features_vic_specific = ds_vic_specific.get_model_features(
            train_config,
            wb_attack_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False)

        collected_features_test.append(features_vic_specific)

    # Wrap into train and test data
    test_labels = [float(x) for x in attack_config.values]

    # Generate test set
    test_loader = wrap_into_loader(
        collected_features_test,
        labels_list=test_labels,
        batch_size=wb_attack_config.batch_size,
        shuffle=False,
    )

    # Create attacker object
    attacker_obj = get_attack(wb_attack_config.attack)(
        dims, wb_attack_config)

    # Load regression model
    attacker_obj.load_model(args.path)

    # Execute attack
    # chosen_mse = attacker_obj.execute_attack(
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    
    
    # Load model
        metamodel.load_state_dict(ch.load(args.model_path))
        # Evaluate
        metamodel = metamodel.cuda()
        loss_fn = ch.nn.MSELoss(reduction='none')
        _, losses, preds = utils.test_meta(
            metamodel, loss_fn, X_test, Y_test.cuda(),
            args.batch_size, None,
            binary=True, regression=True, gpu=True,
            combined=True, element_wise=True,
            get_preds=True)
        y_np = Y_test.numpy()
        losses = losses.numpy()
        print("Mean loss: %.4f" % np.mean(losses))
        # Get all unique ratios in GT, and their average losses from model
        ratios = np.unique(y_np)
        losses_dict = {}
        for ratio in ratios:
            losses_dict[ratio] = np.mean(losses[y_np == ratio])
        print(losses_dict)
        # Conctruct a matrix where every (i, j) entry is the accuracy
        # for ratio[i] v/s ratio [j], where whichever ratio is closer to the
        # ratios is considered the "correct" one
        # Assume equal number of models per ratio, stored in order of
        # SUPPORTED_RATIOS
        acc_mat = np.zeros((len(ratios), len(ratios)))
        for i in range(acc_mat.shape[0]):
            for j in range(i + 1, acc_mat.shape[0]):
                # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
                gt_z = (Y_test[num_per_dist * i:num_per_dist *
                        (i + 1)].numpy() == float(ratios[j]))
                gt_o = (Y_test[num_per_dist * j:num_per_dist *
                        (j + 1)].numpy() == float(ratios[j]))
                # Get relevant preds
                pred_z = preds[num_per_dist * i:num_per_dist * (i + 1)]
                pred_o = preds[num_per_dist * j:num_per_dist * (j + 1)]
                pred_z = (pred_z >= (
                    0.5 * (float(ratios[i]) + float(ratios[j]))))
                pred_o = (pred_o >= (
                    0.5 * (float(ratios[i]) + float(ratios[j]))))
                # Compute accuracies and store
                acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate(
                    (pred_z, pred_o), 0)
                acc_mat[i, j] = np.mean(acc)
        print(acc_mat)
        
        #     val_loader=val_loader)

    print("Test MSE: %.3f" % chosen_mse)
    logger.add_results(wb_attack_config.attack,
                        "regression", chosen_mse, None)

    # Save logger results
    logger.save()
