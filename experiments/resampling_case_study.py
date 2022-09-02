"""
    Look at performance of normal models, and models with under/over sampling.
    And identify simiilarities in predictions, loss values, error cases?
"""
from distribution_inference.config.core import TrainConfig
from simple_parsing import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dataclasses import replace
import os

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import DatasetConfig, FairnessEvalConfig
from distribution_inference.utils import flash_utils
from distribution_inference.attacks.blackbox.utils import get_preds
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fairness_config: FairnessEvalConfig = FairnessEvalConfig.load(
        args.load_config, drop_extra_fields=False)
    # Extract configuration information from config file
    train_config: TrainConfig = fairness_config.train_config
    model_data_config: DatasetConfig = train_config.data_config
    if train_config.misc_config is not None:
        # TODO: Figure out best place to have this logic in the module
        if train_config.misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(train_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(model_data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(model_data_config.name)()

    # Withoout-SHuffle model_config
    train_config_wo_shuffle = replace(train_config)
    train_config_wo_shuffle.misc_config = None
    
    # Create config objects for data-loading
    data_config: DatasetConfig = replace(model_data_config)
    ds_baseline = ds_wrapper_class(
        data_config, skip_data=False)
    train_0_config = replace(train_config_wo_shuffle)
    train_0_config.data_config.prop="0.5"
    # Check if models are graph-related
    are_graph_models = ds_baseline.is_graph_data
    if are_graph_models:
        # No concept of 'processed'
        data_ds, (_, test_idx) = ds_baseline.get_loaders(batch_size=fairness_config.batch_size)
        eval_loader = (data_ds, test_idx)
    else:
        _, eval_loader = ds_baseline.get_loaders(
            batch_size=fairness_config.batch_size)
    
    def to_preds(x):
        exp = np.exp(x)
        return exp / (1 + exp)

    def get_preds_for_model_config(mc: TrainConfig):
        # Create new DS object for models and data loading
        ds_model = ds_wrapper_class(
            mc.data_config,
            skip_data=True,
            label_noise=train_config.label_noise)

        # Load models
        models = ds_model.get_models(
            mc,
            n_models=fairness_config.num_models,
            on_cpu=fairness_config.on_cpu,
            shuffle=False,
            epochwise_version=mc.save_every_epoch,
            model_arch=mc.model_arch)

        # Get predictions for models
        preds, ground_truth = get_preds(
            eval_loader, models,
            preload=fairness_config.preload,
            multi_class=mc.multi_class)
    
        # Convert to probability values if logits
        if not models[0].is_sklearn_model:
            preds = to_preds(preds)
        
        return preds, ground_truth

    # Now that we have model predictions and GT values, compute fairness-related metrics
    preds_shuffle, gt_shuffle = get_preds_for_model_config(train_config)
    preds_noshuffle, gt_no_shuffle = get_preds_for_model_config(train_config_wo_shuffle)
    preds_0, gt_0 = get_preds_for_model_config(train_0_config)
    assert np.all(gt_no_shuffle == gt_shuffle == gt_0)

    # Compare agreement in predictions
    preds_shuffle = 1. * (preds_shuffle > 0.5)
    preds_noshuffle = 1. * (preds_noshuffle > 0.5)
    preds_0 = 1. * (preds_0 > 0.5)
    def find_ag(p1,p2,n1,n2):
        agreements = []
        for i in tqdm(range(p1.shape[0]), desc="Aggregating"):
            for j in range(p2.shape[0]):
                agreements.append(np.mean(p1[i] == p2[j]))
        print("Agreement between {} and {}:{}".format (n1,n2,np.mean(agreements)))
    find_ag(preds_shuffle,preds_noshuffle,"shuffle","no shuffle")
    find_ag(preds_shuffle,preds_0,"shuffle","D0")
    find_ag(preds_noshuffle,preds_0,"no shuffle","D0")
    # (n_models, n_samples)
    # Average across all models
    preds_shuffle = np.mean(preds_shuffle, axis=0)
    preds_noshuffle = np.mean(preds_noshuffle, axis=0)
    preds_0 = np.mean(preds_0,axis=0)
    def find_dif(p1,p2,n1,n2):

        # Compute average difference in predictions
        diff_mean = np.mean(np.abs(p1 - p2))
        diff_median = np.median(np.abs(p1 - p2))
        print("Average difference in predictions between {} and {}: {}".format(n1,n2,diff_mean))
        print("Median difference in predictions between {} and {}: {}".format(n1,n2,diff_median))
    find_dif(preds_shuffle,preds_noshuffle,"shuffle","no shuffle")
    find_dif(preds_shuffle,preds_0,"shuffle","D0")
    find_dif(preds_noshuffle,preds_0,"no shuffle","D0")