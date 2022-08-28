"""
Haven't touch yet
"""
from simple_parsing import ArgumentParser
from pathlib import Path

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.attacks.blackbox.utils import _get_preds_for_vic_and_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
import os
from distribution_inference.attacks.blackbox.KL_regression import KLRegression

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument('--gpu',
                        default=None, help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    are_graph_models = data_config.name=="arxiv"
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255
    assert bb_attack_config.attack_type[0] == "KL_regression"
    # Make sure regression config is not being used here
    if bb_attack_config.regression_config is None:
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

    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)

    # Loading up all models altogether gives OOm for system
    # Have to load all models from scratch again, but only
    # the ones needed; even victim models need to be
    # re-loaded per trial
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    make_processed_version=attack_config.adv_processed_variant
    for _ in range(attack_config.tries):

        # Load up model for each of the values
        models_vic,models_adv  = [], []
        for prop_value in attack_config.values:
            data_config_adv_specific, data_config_vic_specific = get_dfs_for_victim_and_adv(
                data_config, prop_value=prop_value)

            # Create new DS object for both and victim (for other ratio)
            ds_adv_specific = ds_wrapper_class(
                data_config_adv_specific,skip_data=True)
            ds_vic_specific = ds_wrapper_class(
                data_config_vic_specific, skip_data=True)
            
            models_vic.append(ds_vic_specific.get_models(
                train_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                epochwise_version=attack_config.train_config.save_every_epoch,
                model_arch=attack_config.victim_model_arch))
            models_adv.append(ds_adv_specific.get_models(
                    train_adv_config,
                    n_models=bb_attack_config.num_adv_models,
                    on_cpu=attack_config.on_cpu,
                    model_arch=attack_config.adv_model_arch,
                    target_epoch = attack_config.adv_target_epoch))
        
        
        assert not bb_attack_config.regression_config.additional_values_to_test, "Not implmented"
        # Wrap into train and test data
        base_labels = [float(x) for x in attack_config.values]

        # Wrap into test data
        test_labels = base_labels
        preds_vic = []
        preds_adv = []
        data_config_adv_D0, _ = get_dfs_for_victim_and_adv(
                data_config, prop_value=0.5)
        ds_D0 = ds_wrapper_class(
                data_config_adv_D0)
        if are_graph_models:
            assert not make_processed_version
            data_ds, (_, test_idx) = ds_D0.get_loaders(batch_size=bb_attack_config.batch_size)
            loader_vic = (data_ds, test_idx)
            loader_adv = loader_vic
        else:
            loader_for_shape, loader_vic = ds_D0.get_loaders(batch_size=bb_attack_config.batch_size)
            adv_datum_shape = next(iter(loader_for_shape))[0].shape[1:]
            if make_processed_version:
                # Make version of DS for victim that processes data
                # before passing on
                adv_datum_shape = ds_D0.prepare_processed_data(loader_vic)
                loader_adv = ds_D0.get_processed_val_loader(batch_size=bb_attack_config.batch_size)
            else:
                # Get val data loader (should be same for all models, since get_loaders() gets new data for every call)
                loader_adv = loader_vic
        for v,a in zip(models_vic,models_adv):
            adv_p,vic_p,gt_,not_using_logits=_get_preds_for_vic_and_adv(v,a, (loader_vic, loader_adv),
                        epochwise_version=False,preload=bb_attack_config.preload,multi_class=bb_attack_config.multi_class)
            preds_vic.append(vic_p)
            preds_adv.append(adv_p)
        attacker_obj = KLRegression(bb_attack_config)
        # Execute attack
        _,chosen_mse = attacker_obj.attack(
            preds_adv,
            preds_vic,
            not_using_logits=not_using_logits,
            labels=test_labels)

        print("Test MSE:{} ".format(chosen_mse))
        for l,c in zip(test_labels,chosen_mse):
            logger.add_results("KL_regression",
                           l, c, None)

    # Save logger results
    logger.save()
