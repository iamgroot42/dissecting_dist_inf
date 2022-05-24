from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.whitebox.utils import get_attack, get_weight_layers
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from distribution_inference.config import UnlearningConfig
from distribution_inference.defenses.passive.unlearning import Unlearning


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--attacker_path", help="path to meta-classifier",
        type=Path, required=True)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    defense_config: UnlearningConfig = UnlearningConfig.load(
        args.load_config, drop_extra_fields=False)
    train_config = defense_config.train_config
    data_config = defense_config.train_config.data_config
    wb_attack_config = defense_config.wb_config

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)
    ds_info = get_dataset_information(data_config.name)()

    _, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic = ds_wrapper_class(data_config_vic_1, skip_data=True)

    # Create function to create features for given model
    def create_features(x):
        _, x_feats = get_weight_layers(
            x, wb_attack_config, detach=False,
            track_grad=True)
        return [x.cuda() for x in x_feats]

    def unlearn_models(ds):
        # Load victim's model features for given DS
        models_vic = ds.get_models(
            train_config,
            n_models=defense_config.num_models,
            on_cpu=defense_config.on_cpu,
            shuffle=False)

        # Get 'dims'
        dims, _ = get_weight_layers(models_vic[0], wb_attack_config)

        # Create and load meta-classifier
        attacker_obj = get_attack(wb_attack_config.attack)(
            dims, wb_attack_config)
        attacker_obj.load_model(args.attacker_path)

        models_vic_updated = []
        preds_old, preds_new = [], []
        defense = Unlearning(defense_config)
        for model_vic in tqdm(models_vic):
            _, (pred_old, pred_new) = defense.defend(
                attacker_obj, model_vic, create_features)
            models_vic_updated.append(models_vic)
            preds_old.append(pred_old)
            preds_new.append(pred_new)

        preds_new = np.array(preds_new)
        preds_old = np.array(preds_old)

        return preds_old, preds_new

    # Get old and new preds for first ratio of DS
    preds_old_1, preds_new_1 = unlearn_models(ds_vic)

    # Get old and new preds for second ratio of DS
    _, data_config_vic_2 = get_dfs_for_victim_and_adv(
        data_config, prop_value=defense_config.other_val)
    ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)
    preds_old_2, preds_new_2 = unlearn_models(ds_vic_2)

    # Construct ground truth
    gt = np.concatenate(
        (np.zeros_like(preds_old_1), np.ones_like(preds_old_2)))
    preds_before = np.concatenate(
        (preds_old_1, preds_old_2))
    preds_after = np.concatenate(
        (preds_new_1, preds_new_2))

    # Compute defense success rate before v/s after
    print("Attack accuracy rate before: %.3f" % (np.mean(preds_before == gt)))
    print("Attack accuracy rate after: %.3f" % (np.mean(preds_after == gt)))
    print("Success rate for defense changing predictions: %.3f" %
          (np.mean(preds_after != preds_before)))
