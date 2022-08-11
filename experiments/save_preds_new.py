"""
Store predictions in a dic with structure dic[model's ratio][data's ratio][trial]
"""

from simple_parsing import ArgumentParser
from pathlib import Path
import os
import pickle
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import get_attack, calculate_accuracies, _get_preds_for_vic_and_adv,_get_preds_accross_epoch
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils,ensure_dir_exists
from distribution_inference.logging.core import AttackResult
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default=None, help="device number")
    parser.add_argument(
        "--ratios", nargs='+',help="ratios", type=float,required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    attack_config: AttackConfig = AttackConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file
    bb_attack_config: BlackBoxAttackConfig = attack_config.black_box
    train_config: TrainConfig = attack_config.train_config
    data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255
    
    # Print out arguments
    flash_utils(attack_config)
    ratios = args.ratios
    EPOCH=train_config.save_every_epoch
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)(epoch_wise=EPOCH)

    
    
    def single_evaluation():
        preds_a = {}
        preds_v = {}
        gt = {}
        batch_size = bb_attack_config.batch_size
        make_processed_version=attack_config.adv_processed_variant
        for r0 in ratios:
            preds_a[r0] = {}
            preds_v[r0] = {}
            gt[r0] = {}
            # Create new DS object for both and victim
            data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
                data_config,prop_value=r0)
            ds_vic_1 = ds_wrapper_class(
                data_config_vic_1,
                skip_data=True,
                label_noise=train_config.label_noise,
                epoch=EPOCH)
            ds_adv_1 = ds_wrapper_class(data_config_adv_1,epoch=EPOCH)
            train_adv_config = get_train_config_for_adv(train_config, attack_config)
            # Load victim models for first value
            models_vic_1 = ds_vic_1.get_models(
                train_config,
                n_models=attack_config.num_victim_models,
                on_cpu=attack_config.on_cpu,
                shuffle=False,
                model_arch=attack_config.victim_model_arch,
                custom_models_path=None,
                epochwise_version=EPOCH)
            
            # For each value (of property) asked to experiment with
            for prop_value in ratios:
                data_config_adv_2, _ = get_dfs_for_victim_and_adv(
                    data_config, prop_value=prop_value)

                # Create new DS object for both and victim (for other ratio)
                ds_adv_2 = ds_wrapper_class(data_config_adv_2,epoch=EPOCH)
                preds_a[r0][prop_value] = {}
                preds_v[r0][prop_value] = {}
                gt[r0][prop_value] = {}
                for t in range(attack_config.tries):
                    print("{} on {}: trial {}".format(r0,prop_value, t))
                    models_adv_1 = ds_adv_1.get_models(
                        train_adv_config,
                        n_models=bb_attack_config.num_adv_models,
                        on_cpu=attack_config.on_cpu,
                        model_arch=attack_config.adv_model_arch,
                        epochwise_version=EPOCH)
                    loader_for_shape, loader_vic = ds_adv_2.get_loaders(batch_size=batch_size)
                    adv_datum_shape = next(iter(loader_for_shape))[0].shape[1:]
                    if make_processed_version:
                        # Make version of DS for victim that processes data
                        # before passing on
                        adv_datum_shape = ds_adv_2.prepare_processed_data(loader_vic)
                        loader_adv = ds_adv_2.get_processed_val_loader(batch_size=batch_size)
                    else:
                        # Get val data loader (should be same for all models, since get_loaders() gets new data for every call)
                        loader_adv = loader_vic
                    if EPOCH:
                        assert not make_processed_version
                        adv_p,gt = _get_preds_accross_epoch(models_adv_1,loader_vic,preload=bb_attack_config.preload,multi_class=bb_attack_config.multi_class)
                        vic_p,_ = _get_preds_accross_epoch(models_vic_1,loader_vic,preload=bb_attack_config.preload,multi_class=bb_attack_config.multi_class)

                    else:
                        adv_p,vic_p,gt,_=_get_preds_for_vic_and_adv(models_vic_1,models_adv_1, (loader_vic, loader_adv),
                        epochwise_version=EPOCH,preload=bb_attack_config.preload,multi_class=bb_attack_config.multi_class)
                    preds_a[r0][prop_value][t] = adv_p
                    preds_v[r0][prop_value][t] = vic_p
                    gt[r0][prop_value][t] = gt
        return preds_a,preds_v,gt
    
    preds = single_evaluation()
    preds_path = os.path.join(ds_info.base_models_dir,"preds",args.en)
    ensure_dir_exists(preds_path)
    attack_config.save(os.path.join(preds_path,"config.json"),indent=4)
    with open(os.path.join(preds_path,"preds_a.p"),"wb") as f:
        pickle.dump(preds[0], f)
    with open(os.path.join(preds_path,"preds_v.p"),"wb") as f:
        pickle.dump(preds[1], f)
    with open(os.path.join(preds_path,"gt.p"),"wb") as f:
        pickle.dump(preds[2], f)
        