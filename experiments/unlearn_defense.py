from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.attacks.whitebox.utils import get_attack, get_weight_layers
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from distribution_inference.attacks.whitebox.affinity.utils import get_loader_for_seed_data
from distribution_inference.config import DefenseConfig
from distribution_inference.training.utils import save_model
from distribution_inference.utils import flash_utils, ensure_dir_exists
from distribution_inference.defenses.passive.unlearning import Unlearning
from distribution_inference.logging.core import DefenseResult


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument(
        "--path", help="path to saved meta-classifier",
        type=Path, required=True)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument(
        "--savepath", help="if not blank, save victim models to this path",
        default="", type=str)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    defense_config: DefenseConfig = DefenseConfig.load(
        args.load_config, drop_extra_fields=False)
    train_config = defense_config.train_config
    data_config = defense_config.train_config.data_config
    wb_attack_config = defense_config.wb_config

    # Make sure a model is present for each value to be tested
    values_to_test = [str(x) for x in defense_config.values]
    folders = os.listdir(args.path)
    for value in values_to_test:
        if value not in folders:
            raise ValueError(
                f"No model found for value {value} in {args.path}")

    # Print out arguments
    flash_utils(defense_config)

    # Define logger
    logger = DefenseResult(args.en, defense_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)
    ds_info = get_dataset_information(data_config.name)()

    _, data_config_vic_1 = get_dfs_for_victim_and_adv(
        data_config)
    ds_vic = ds_wrapper_class(data_config_vic_1, skip_data=True)

    def unlearn_models(ds, attacker_model_path, save_prefix=None):
        # Load victim's model features for given DS
        models_vic = ds.get_models(
            train_config,
            n_models=defense_config.num_models,
            on_cpu=defense_config.on_cpu,
            shuffle=False)

        # Get 'dims'
        if wb_attack_config.attack == "affinity":
            dims = None
        else:
            dims, _ = get_weight_layers(models_vic[0], wb_attack_config)

        # Create and load meta-classifier
        attacker_obj = get_attack(wb_attack_config.attack)(
            dims, wb_attack_config)
        attacker_obj.load_model(attacker_model_path)

        # Create seed-data loader, if AMC attack
        if wb_attack_config.attack == "affinity":
            seed_data_loader = get_loader_for_seed_data(
                attacker_obj.seed_data_ds, wb_attack_config)

            # Create function to create features for given model
            # For AMC
            def create_features_fn(x):
                affinity_feature, _, _, _ = attacker_obj._make_affinity_feature(
                    x, seed_data_loader, detach=False)
                return [x.unsqueeze(0).cuda() for x in affinity_feature]

        elif wb_attack_config.attack == "permutation_invariant":
            # Create function to create features for given model
            # For PIN
            def create_features_fn(x):
                _, x_feats = get_weight_layers(
                    x, wb_attack_config, detach=False,
                    track_grad=True)
                return [x.cuda() for x in x_feats]

        else:
            raise ValueError(f"Attack {wb_attack_config.attack} not supported")

        preds_old, preds_new = [], []
        defense = Unlearning(defense_config.unlearning_config)
        iterator = tqdm(enumerate(models_vic),
                        desc="Unlearning Property", total=len(models_vic))
        for i, model_vic in iterator:
            model_vic_new, (pred_old, pred_new) = defense.defend(
                attacker_obj, model_vic, create_features_fn,
                verbose=False)
            # Keep track of before/after predictions
            preds_old.append(pred_old)
            preds_new.append(pred_new)
            # Save new model, if requested
            if save_prefix is not None:
                file_name = f"modified_{i+1}.ch"
                save_model(model_vic_new, os.path.join(save_prefix, file_name))

        preds_new = np.array(preds_new)
        preds_old = np.array(preds_old)

        return preds_old, preds_new

    # For all ratios
    for prop_value in defense_config.values:
        # Create new dataset wrapper
        _, data_config_vic_2 = get_dfs_for_victim_and_adv(
            data_config, prop_value=prop_value)
        ds_vic_2 = ds_wrapper_class(data_config_vic_2, skip_data=True)

        # Look at all saved meta-classifiers
        attack_model_path_dir = os.path.join(args.path, str(prop_value))
        if defense_config.victim_local_attack:
            attack_model_path_dir = os.path.join(
                attack_model_path_dir, "victim_local")

        for i, attack_model_path in enumerate(os.listdir(attack_model_path_dir)):
            # Load model
            attacker_model_path = os.path.join(
                attack_model_path_dir, attack_model_path)

            # Get old and new preds
            prefix = None
            if args.savepath:
                prefix = os.path.join(
                    args.savepath,
                    "unlearn",
                    data_config.prop,
                    str(data_config.value))
                prefix_1 = os.path.join(prefix,
                    str(data_config.value), str(i + 1))
                prefix_2 = os.path.join(prefix,
                    str(prop_value), str(i + 1))
                ensure_dir_exists(prefix_1)
                ensure_dir_exists(prefix_2)

            # 'Unlearn' and save new models
            preds_old_1, preds_new_1 = unlearn_models(
                ds_vic, attacker_model_path,
                save_prefix=prefix_1)
            preds_old_2, preds_new_2 = unlearn_models(
                ds_vic_2, attacker_model_path,
                save_prefix=prefix_2)

            # Construct ground truth
            ground_truth = np.concatenate(
                (np.zeros(preds_old_1.shape[0]), np.ones(preds_old_2.shape[0])))

            # Construct predictions
            preds_before = np.concatenate(
                (preds_old_1, preds_old_2))
            preds_after = np.concatenate(
                (preds_new_1, preds_new_2))

            # Compute useful statistics
            before_acc = np.mean(np.argmax(preds_before, 1) == ground_truth)
            after_acc = np.mean(np.argmax(preds_after, 1) == ground_truth)

            # Log raw predictions (for use later)
            logger.add_results(wb_attack_config.attack, prop_value,
                               before_acc, after_acc)

    # TODO: Assess drop in task performance for models

    # Save logger results
    logger.save()
