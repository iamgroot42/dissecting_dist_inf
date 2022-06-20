import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from simple_parsing import ArgumentParser
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.blackbox.utils import  get_vic_adv_preds_on_distr
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv, get_train_config_for_adv
from distribution_inference.config import DatasetConfig, AttackConfig, WhiteBoxAttackConfig, TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.whitebox.comparison.comparison import ComparisonAttack
if __name__ == "__main__":
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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
    flash_utils(attack_config)
    BATCH_SIZE=wb_attack_config.batch_size
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()
    assert wb_attack_config.attack=="comparison", "This script is only for comparison attack"
    attacker_obj = ComparisonAttack(train_config,wb_attack_config)
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    
    # Make train config for adversarial models
    train_config_adv = get_train_config_for_adv(train_config, attack_config)
    ds_vic_1 = ds_wrapper_class(data_config_vic_1, skip_data=True,epoch=True)
    ds_adv_1 = ds_wrapper_class(data_config_adv_1,epoch=True)
    # Load victim models for first value
    models_vic_1 = ds_vic_1.get_models(
        train_config,
        n_models=attack_config.num_victim_models,
        on_cpu=attack_config.on_cpu,
        shuffle=False,
        epochwise_version=True)
    models_vic_1 = (models_vic_1[wb_attack_config.comparison_config.Start_epoch-1],models_vic_1[wb_attack_config.comparison_config.End_epoch-1])
    train_adv_config = get_train_config_for_adv(train_config, attack_config)
    for prop_value in attack_config.values:
        data_config_adv_2, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)
        
        # Create new DS object for both and victim (for other ratio)
        ds_adv_2 = ds_wrapper_class(data_config_adv_2,epoch=True)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True,epoch=True)
        # Load victim models for other value
        models_vic_2 = ds_vic_2.get_models(
            train_config,
            n_models=attack_config.num_victim_models,
            on_cpu=attack_config.on_cpu,
            shuffle=False,
            epochwise_version=True)
        models_vic_2 = (models_vic_2[wb_attack_config.comparison_config.Start_epoch-1],models_vic_2[wb_attack_config.comparison_config.End_epoch-1])
        for t in range(attack_config.tries):
            _, loader1 = ds_adv_1.get_loaders(batch_size=BATCH_SIZE)
            _, loader2 = ds_adv_2.get_loaders(batch_size=BATCH_SIZE)
            models_adv_1 = attacker_obj.train(models_vic_1[0],0.5)
            models_adv_2 = attacker_obj.train(models_vic_2[0],prop_value)
            preds_adv_on_1, preds_vic_on_1, ground_truth_1 = get_vic_adv_preds_on_distr(
                    models_vic=(models_vic_1, models_vic_2),
                    models_adv=(models_adv_1, models_adv_2),
                    ds_obj=ds_adv_1,
                    batch_size=BATCH_SIZE,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    preload=True,
                    multi_class=False,
                    make_processed_version=attack_config.victim_full_model
                )
                # Get victim and adv predictions on loaders for second ratio
            preds_adv_on_2, preds_vic_on_2, ground_truth_2 = get_vic_adv_preds_on_distr(
                    models_vic=(models_vic_1, models_vic_2),
                    models_adv=(models_adv_1, models_adv_2),
                    ds_obj=ds_adv_2,
                    batch_size=BATCH_SIZE,
                    epochwise_version=attack_config.train_config.save_every_epoch,
                    preload=True,
                    multi_class=False,
                    make_processed_version=attack_config.victim_full_model
                )
                # Wrap predictions to be used by the attack
            preds_adv = PredictionsOnDistributions(
                    preds_on_distr_1=preds_adv_on_1,
                    preds_on_distr_2=preds_adv_on_2
                )
            preds_vic = PredictionsOnDistributions(
                    preds_on_distr_1=preds_vic_on_1,
                    preds_on_distr_2=preds_vic_on_2
                )
            result = attacker_obj.attack(
                        preds_adv, preds_vic)

            logger.add_results("comparison", prop_value,
                                       result[0][0], result[1][0])
    logger.save()
