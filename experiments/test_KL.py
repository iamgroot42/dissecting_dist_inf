from distribution_inference.attacks.blackbox.KL import KLAttack
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.config import DatasetConfig, AttackConfig, BlackBoxAttackConfig, TrainConfig
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    args = parser.parse_args()
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
    # Define logger
    path = "/p/adversarialml/temp/boneage_preds_0.4"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    preds_adv_on_1 = data['preds_adv_on_1']
    preds_adv_on_2 = data['preds_adv_on_2']
    preds_vic_on_1 = data['preds_vic_on_1']
    preds_vic_on_2 = data['preds_vic_on_2']
    pa = PredictionsOnDistributions(preds_adv_on_1,preds_adv_on_2)
    pv = PredictionsOnDistributions(preds_vic_on_1,preds_vic_on_2)
    attack_object = KLAttack(bb_attack_config)
    result = attack_object.attack(pa,pv)
    print(result[0][0])

