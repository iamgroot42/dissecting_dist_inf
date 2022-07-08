
import os
import pickle
from pathlib import Path
from simple_parsing import ArgumentParser
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information

from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
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
    parser.add_argument(
        "--pred_name", help="Specify preds file",
        type=Path, required=True)
    args = parser.parse_args()
    
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
    
    # Define logger
    logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)(epoch_wise=True)
    assert wb_attack_config.attack=="comparison", "This script is only for comparison attack"
    attacker_obj = ComparisonAttack(train_config,wb_attack_config,info=ds_info,save_m=False)
    data_config_adv_1, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    
    preds_path = os.path.join(ds_info.base_models_dir,"comparison","preds",args.pred_name)
    preds_a1 = pickle.load( open( os.path.join(preds_path,"preds_a1.p"), "rb" ) )
    preds_a2 = pickle.load( open( os.path.join(preds_path,"preds_a2.p"), "rb" ) )
    preds_v = pickle.load( open( os.path.join(preds_path,"preds_v.p"), "rb" ) )
    ground_truths = pickle.load( open( os.path.join(preds_path,"gt.p"), "rb" ) )
    for prop_value in attack_config.values:
        for t in range(attack_config.tries):
            preds_adv1 = preds_a1[prop_value][t]
            preds_adv2 = preds_a2[prop_value][t]
            preds_vic = preds_v[prop_value][t]
            gt = ground_truths[prop_value][t]
            result = attacker_obj.attack(preds_adv1,
                        preds_adv2, preds_vic)
            vacc= result[0][0]
            print(vacc)
            logger.add_results("comparison", prop_value,
                                       vacc, result[1][0])
    logger.save()
