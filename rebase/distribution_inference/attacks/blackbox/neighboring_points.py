import numpy as np
from typing import Tuple
from typing import List
import gc
from distribution_inference.attacks.blackbox.core import PredictionsOnOneDistribution, PredictionsOnOneDistribution
from torch.distributions import normal
import torch as ch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from distribution_inference.datasets.base import CustomDatasetWrapper
from distribution_inference.attacks.blackbox.utils import get_preds


class AddGaussianNoise():
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        self.dis = normal.Normal(mean, std)

    def __call__(self, tensor):
        return tensor + self.dis.sample(tensor.size())

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class neighborDataset(Dataset):
    def __init__(self, num_neighbors, original, transform):
        self.num_neighbors = num_neighbors
        self.X = []
        for _ in range(num_neighbors):
            self.X.append(transform(original))
        self.X = ch.cat(self.X, 0)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]


def get_estimate(loader, models: List[nn.Module],
                 verbose: bool = True,
                 multi_class: bool = False,
                 latent: int = None,
                 not_using_logits: bool = False,
                 num_neighbor: int = 10,
                 mean: float = 0.0,
                 std: float = 0.1):
    assert not models[0].is_graph_model, "No support for graph model"
    noise = AddGaussianNoise(mean, std)
    predictions = []
    ground_truth = []
    thre = 0.5 if not_using_logits else 0
    # Accumulate all data for given loader
    for data in loader:
        if len(data) == 2:
            features, labels = data
        else:
            features, labels, _ = data
        ground_truth.append(labels.cpu().numpy())
    ground_truth = np.concatenate(ground_truth, axis=0)

    # Get predictions for each model
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Generating Predictions")
    for model in iterator:
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        with ch.no_grad():
            predictions_on_model = []

            for data in loader:
                data_points, labels, _ = data
                # Infer batch size
                batch_size_desired = len(data_points)
                
                # Create new loader that adds noise to data
                new_loader = DataLoader(dataset=neighborDataset(
                    num_neighbor, data_points, noise),
                    batch_size=batch_size_desired)
                p_collected = []
                for neighbor in new_loader:
                    # Get prediction
                    if latent != None:
                        prediction = model(
                                neighbor.cuda(), latent=latent).detach()
                    else:
                        prediction = model(neighbor.cuda()).detach()
                        if not multi_class:
                            prediction = prediction[:, 0]
                    p_collected.append(1. * (prediction.cpu().numpy() >= thre))
                # Tile predictions and average over appropriate means
                p_collected = np.array(p_collected).flatten().reshape(num_neighbor, -1)
                predictions_on_model.append(np.mean(p_collected, 0))
        predictions.append(np.concatenate(predictions_on_model, 0))
        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()
    predictions = np.stack(predictions, 0)
    gc.collect()
    ch.cuda.empty_cache()

    return predictions, ground_truth


def _get_preds_for_vic_and_adv(
        models_vic: List[nn.Module],
        models_adv: List[nn.Module],
        loader,
        mean: float,
        std : float,
        neighbors: int,
        epochwise_version: bool = False,
        preload: bool = False,
        multi_class: bool = False,):

    # Sklearn models do not support logits- take care of that
    use_prob_adv = models_adv[0].is_sklearn_model
    if epochwise_version:
        use_prob_vic = models_vic[0][0].is_sklearn_model
    else:
        use_prob_vic = models_vic[0].is_sklearn_model
    
    # preds_vic for this function will be probabilities always
    not_using_logits = True

    if type(loader) == tuple:
        #  Same data is processed differently for vic/adcv
        loader_vic, loader_adv = loader
    else:
        # Same loader
        loader_adv = loader
        loader_vic = loader

    # Get predictions for adversary models and data
    def to_preds(x):
        exp = np.exp(x)
        return exp / (1 + exp)
    preds_adv, ground_truth_repeat = get_preds(
        loader_adv, models_adv, preload=preload,
        multi_class=multi_class)
    if not_using_logits and not use_prob_adv:
        preds_adv = to_preds(preds_adv)

    # Get predictions for adversary models and data
    # preds_adv, ground_truth_repeat = get_estimate(
    #     loader_adv, models_adv, preload=preload,
    #     multi_class=multi_class, not_using_logits=not_using_logits)

    # Get predictions for victim models and data
    if epochwise_version:
        # Track predictions for each epoch
        preds_vic = []
        for models_inside_vic in tqdm(models_vic):
            preds_vic_inside, ground_truth = get_estimate(
                loader_vic, models_inside_vic,
                verbose=False, multi_class=multi_class,
                not_using_logits=use_prob_vic,
                num_neighbor=neighbors,
                mean = mean, std=std)

            # In epoch-wise mode, we need prediction results
            # across epochs, not models
            preds_vic.append(preds_vic_inside)
    else:
        preds_vic, ground_truth = get_estimate(
            loader_vic, models_vic,
            multi_class=multi_class,
            not_using_logits=use_prob_vic,
            num_neighbor=neighbors,
            mean = mean, std=std)
    assert np.all(ground_truth ==
                  ground_truth_repeat), "Val loader is shuffling data!"
    return preds_vic, preds_adv, ground_truth, not_using_logits


def get_vic_adv_preds_on_distr(
        models_vic: Tuple[List[nn.Module], List[nn.Module]],
        models_adv: Tuple[List[nn.Module], List[nn.Module]],
        ds_obj: CustomDatasetWrapper,
        batch_size: int,
        mean: float,
        std: float,
        neighbors: int,
        epochwise_version: bool = False,
        preload: bool = False,
        multi_class: bool = False,
        make_processed_version: bool = False):

    # Check if models are graph-related
    are_graph_models = False
    if epochwise_version:
        if models_vic[0][0][0].is_graph_model:
            are_graph_models = True
    else:
        if models_vic[0][0].is_graph_model:
            are_graph_models = True

    if are_graph_models:
        # No concept of 'processed'
        data_ds, (_, test_idx) = ds_obj.get_loaders(batch_size=batch_size)
        loader_vic = (data_ds, test_idx)
        loader_adv = loader_vic
    else:
        loader_for_shape, loader_vic = ds_obj.get_loaders(
            batch_size=batch_size, shuffle=False)
            # Given the way ground truth is collected, shuffling should be disabled
        adv_datum_shape = next(iter(loader_for_shape))[0].shape[1:]

        if make_processed_version:
            # Make version of DS for victim that processes data
            # before passing on
            adv_datum_shape = ds_obj.prepare_processed_data(loader_vic)
            loader_adv = ds_obj.get_processed_val_loader(batch_size=batch_size)
        else:
            # Get val data loader (should be same for all models, since get_loaders() gets new data for every call)
            loader_adv = loader_vic

        # TODO: Use preload logic here to speed things even more

    # Get predictions for first set of models
    preds_vic_1, preds_adv_1, ground_truth, not_using_logits = _get_preds_for_vic_and_adv(
        models_vic[0], models_adv[0],
        (loader_vic, loader_adv),
        mean = mean,
        std = std,
        neighbors = neighbors,
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class)
    # Get predictions for second set of models
    preds_vic_2, preds_adv_2, _, _ = _get_preds_for_vic_and_adv(
        models_vic[1], models_adv[1],
        (loader_vic, loader_adv),
        mean=mean,
        std=std,
        neighbors=neighbors,
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class)
    adv_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_adv_1,
        preds_property_2=preds_adv_2
    )
    vic_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_vic_1,
        preds_property_2=preds_vic_2
    )
    return adv_preds, vic_preds, ground_truth, not_using_logits
