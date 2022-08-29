"""
    Load up models and data to measure fairness-related metrics.
    TODO: Implement. Currently just a copy-paste of BB attacks
"""
from distribution_inference.config.core import TrainConfig
from simple_parsing import ArgumentParser
import fairlearn.metrics as fm
from pathlib import Path
import numpy as np
from dataclasses import replace
import os
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import DatasetConfig, FairnessEvalConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import AttackResult
from distribution_inference.attacks.blackbox.utils import get_preds


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

    # Create config objects for data-loading
    data_config: DatasetConfig = replace(model_data_config)
    data_config.prop = fairness_config.prop
    data_config_lower = replace(data_config)
    data_config_lower.value = fairness_config.value_min
    data_config_upper = replace(data_config)
    data_config_upper.value = fairness_config.value_max
    ds_baseline_min = ds_wrapper_class(
        data_config_lower, skip_data=False)
    ds_baseline_max = ds_wrapper_class(
        data_config_upper, skip_data=False)

    # Create new DS object for models and data loading
    ds_model = ds_wrapper_class(
        model_data_config,
        skip_data=True,
        label_noise=train_config.label_noise)

    # Load models
    models = ds_model.get_models(
        train_config,
        n_models=fairness_config.num_models,
        on_cpu=fairness_config.on_cpu,
        shuffle=False,
        epochwise_version=train_config.save_every_epoch,
        model_arch=train_config.model_arch)
    
    # Check if models are graph-related
    are_graph_models = False
    if models[0].is_graph_model:
        are_graph_models = True
    
    def to_preds(x):
        exp = np.exp(x)
        return exp / (1 + exp)

    def generate_preds(ds):
        if are_graph_models:
            # No concept of 'processed'
            data_ds, (_, test_idx) = ds.get_loaders(batch_size=fairness_config.batch_size)
            eval_loader = (data_ds, test_idx)
        else:
            _, eval_loader = ds.get_loaders(batch_size=fairness_config.batch_size)

        # Get predictions for adversary models and data
        preds, ground_truth = get_preds(
            eval_loader, models, preload=fairness_config.preload,
            multi_class=train_config.multi_class)
    
        # Convert to probability values if logits
        if not models[0].is_sklearn_model:
            preds = to_preds(preds)
        
        return preds, ground_truth

    # Now that we have model predictions and GT values, compute fairness-related metrics
    preds_zero_att, gt_zero_att = generate_preds(ds_baseline_min)
    preds_one_att, gt_one_att = generate_preds(ds_baseline_max)

    # Shapres are (num_models,num_samples) and (num_samples)
    preds_combined = np.concatenate((preds_zero_att, preds_one_att), axis=1)
    gt_combined = np.concatenate((gt_zero_att, gt_one_att), axis=0)
    sensitive_features = np.concatenate((np.zeros(preds_zero_att.shape[1]), np.ones(preds_one_att.shape[1])), axis=0)

    # Function to collect fairness metrics per model
    def model_wise_eval(pred):
        pred_ = (pred > 0.5).astype(int)
        eq_odds_diff = fm.equalized_odds_difference(
            y_true=gt_combined,
            y_pred=pred_,
            sensitive_features=sensitive_features)
        dp_diff = fm.demographic_parity_difference(
            y_true=gt_combined,
            y_pred=pred_,
            sensitive_features=sensitive_features)

        return eq_odds_diff, dp_diff

    metrics = np.array([model_wise_eval(pred) for pred in preds_combined])
    print("Average equalized-odds difference: %.3f" % np.mean(metrics[:, 0]))
    print("Average demographic parity difference: %.3f" % np.mean(metrics[:, 1]))
