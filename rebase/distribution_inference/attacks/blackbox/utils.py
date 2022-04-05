from collections import OrderedDict
from typing import List, Tuple
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch as ch

from distribution_inference.attacks.blackbox.per_point import PerPointThresholdAttack
from distribution_inference.attacks.blackbox.standard import LossAndThresholdAttack
from distribution_inference.attacks.blackbox.core import PredictionsOnOneDistribution
from distribution_inference.datasets.base import CustomDatasetWrapper


ATTACK_MAPPING = {
    "threshold_perpoint": PerPointThresholdAttack,
    "loss_and_threshold": LossAndThresholdAttack,
}


def get_attack(attack_name: str):
    wrapper = ATTACK_MAPPING.get(attack_name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_name} not implemented")
    return wrapper


def calculate_accuracies(data, labels, use_logit: bool = True):
    """
        Function to compute model-wise average-accuracy on
            given data.
    """
    # Get predictions from each model (each model outputs logits)
    if use_logit:
        preds = (data >= 0).astype('int')
    else:
        preds = (data >= 0.5).astype('int')

    # Repeat ground-truth (across models)
    expanded_gt = np.repeat(np.expand_dims(labels, axis=1), preds.shape[1], axis=1)

    return np.average(1. * (preds == expanded_gt), axis=0)


def get_preds(loader, models: List[nn.Module]):
    """
        Get predictions for given models on given data
    """
    predictions = []
    ground_truth = []
    # Accumulate all data for given loader
    for data in loader:
        _, labels, _ = data
        ground_truth.append(labels.cpu().numpy())

    # Get predictions for each model
    for model in tqdm(models):
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        with ch.no_grad():
            predictions_on_model = []
            # Iterate through data-loader
            for data in loader:
                data_points, labels, _ = data
                data_points = data_points.cuda()
                # Get prediction
                prediction = model(data_points).detach()[:, 0]
                predictions_on_model.append(prediction.cpu())
        predictions_on_model = ch.cat(predictions_on_model)
        predictions.append(predictions_on_model)

    predictions = ch.stack(predictions, 0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    return predictions.numpy(), ground_truth


def get_preds_for_models(models: List[nn.Module],
                         ds_obj: CustomDatasetWrapper,
                         batch_size: int):
    # Get val data loader
    _, loader = ds_obj.get_loaders(batch_size=batch_size,
                                   eval_shuffle=False)
    # Get predictions for models on data
    preds, ground_truth = get_preds(loader, models)
    return preds, ground_truth


def _get_preds_for_vic_and_adv(models_vic: List[nn.Module],
                               models_adv: List[nn.Module],
                               loader):
    # Get predictions for victim models and data
    preds_vic, ground_truth = get_preds(loader, models_vic)
    # Get predictions for adversary models and data
    preds_adv, ground_truth_repeat = get_preds(loader, models_adv)
    assert np.all(ground_truth == ground_truth_repeat), "Val loader is probably shuffling data!"
    return preds_vic, preds_adv, ground_truth


def get_vic_adv_preds_on_distr(
        models_vic: Tuple[List[nn.Module], List[nn.Module]],
        models_adv: Tuple[List[nn.Module], List[nn.Module]],
        ds_obj: CustomDatasetWrapper,
        batch_size: int):
    # Get val data loader (should be same for all models, since get_loaders() gets new data for every call)
    _, loader = ds_obj.get_loaders(batch_size=batch_size)
    # Get predictions for first set of models
    preds_vic_1, preds_adv_1, ground_truth = _get_preds_for_vic_and_adv(
        models_vic[0], models_adv[0], loader)
    # Get predictions for second set of models
    preds_vic_2, preds_adv_2, _ = _get_preds_for_vic_and_adv(
        models_vic[1], models_adv[1], loader)
    adv_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_adv_1,
        preds_property_2=preds_adv_2
    )
    vic_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_vic_1,
        preds_property_2=preds_vic_2
    )
    return (adv_preds, vic_preds, ground_truth)


def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups):
    """ Compute the key metrics """
    from aif360.metrics import ClassificationMetric
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * \
        (classified_metric_pred.true_positive_rate() +
         classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = \
        classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = \
        classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = \
        classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    metrics["False discovery rate difference"] = \
        classified_metric_pred.false_discovery_rate_difference()
    metrics["False discovery rate ratio"] = \
        classified_metric_pred.false_discovery_rate_ratio()
    metrics["False omission rate difference"] = \
        classified_metric_pred.false_omission_rate_difference()
    metrics["False omission rate ratio"] = \
        classified_metric_pred.false_omission_rate_ratio()
    metrics["False negative rate difference"] = \
        classified_metric_pred.false_negative_rate_difference()
    metrics["False negative rate ratio"] = \
        classified_metric_pred.false_negative_rate_ratio()
    metrics["False positive rate difference"] = \
        classified_metric_pred.false_positive_rate_difference()
    metrics["False positive rate ratio"] = \
        classified_metric_pred.false_positive_rate_ratio()

    return metrics
