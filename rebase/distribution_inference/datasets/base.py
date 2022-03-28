import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from typing import List
import warnings

from distribution_inference.utils import check_if_inside_cluster, warning_string, log
from distribution_inference.config import DatasetConfig, TrainConfig, WhiteBoxAttackConfig
from distribution_inference.attacks.whitebox.utils import get_weight_layers
import distribution_inference.datasets.utils as utils


class Constants:
    splits = ["victim", "adv"]
    if check_if_inside_cluster():
        base_data_directory = "/scratch/as9rw/datasets/"
        base_models_directory = "/scratch/as9rw/"
    else:
        base_data_directory = "/p/adversarialml/as9rw/datasets/"
        base_models_directory = "/p/adversarialml/as9rw/"


class DatasetInformation:
    def __init__(self, name: str, data_path: str,
                 models_path: str, properties: list,
                 values: dict, property_focus: dict):
        """
            data_path : path to dataset
            models_path: path to models
            properties: list of properties supported for experiments
            values(dict): list of values for each property
        """
        self.base_data_dir = os.path.join(Constants.base_data_directory, data_path)
        self.base_models_dir = os.path.join(Constants.base_models_directory, models_path)
        self.name = name
        self.properties = properties
        self.values = values
        self.property_focus = property_focus

    def get_model(self, cpu: bool = False) -> nn.Module:
        raise NotImplementedError(f"Implement method to model for {self.name} dataset")

    def get_model_for_dp(self, cpu: bool = False) -> nn.Module:
        raise NotImplementedError(f"DP Training not supported for {self.name} dataset")

    def generate_victim_adversary_splits(self, adv_ratio: float, test_ratio: float, num_tries: int):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        raise NotImplementedError("Dataset does not have a method to generate victim and adversary splits")


class CustomDataset(Dataset):
    def __init__(self, classify, prop, ratio, cwise_sample,
                 shuffle: bool = False, transform=None):
        self.num_samples = None

    def __len__(self):
        """
            self.num_samples should be populated in
            init to compute the number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
            Should return (datum, attribute, class-label)
        """
        raise NotImplementedError("Dataset does not implement __getitem__")


class CustomDatasetWrapper:
    def __init__(self, data_config: DatasetConfig):
        """
            self.ds_train and self.ds_val should be set to
            datasets to be used to train and evaluate.
        """
        self.prop = data_config.prop
        self.ratio = data_config.value
        self.split = data_config.split
        self.classify = data_config.classify
        self.augment = data_config.augment
        self.cwise_samples = data_config.cwise_samples
        self.drop_senstive_cols = data_config.drop_senstive_cols
        self.scale = data_config.scale
        self.squeeze = data_config.squeeze

        # Either set ds_train and ds_val here
        # Or set them inside get_loaders
        self.info_object = None

    def get_loaders(self, batch_size,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: float = 1,
                    num_workers: int = 0,
                    prefetch_factor: int = 2):

        train_loader = DataLoader(
            self.ds_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            prefetch_factor=prefetch_factor)

        test_loader = DataLoader(
            self.ds_val,
            batch_size=batch_size * val_factor,
            shuffle=eval_shuffle,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            prefetch_factor=prefetch_factor)

        return train_loader, test_loader

    def get_save_dir(self, train_config: TrainConfig) -> str:
        """
            Return path to directory where models will be saved,
            for a given configuration.
        """
        raise NotImplementedError("Function to fetch model save path not implemented")

    def get_save_path(self, train_config: TrainConfig, name: str) -> str:
        """
            Function to get prefix + name for saving
            the model.
        """
        prefix = self.get_save_dir(train_config)
        if name is None:
            return prefix
        return os.path.join(prefix, name)

    def load_model(self, path: str, on_cpu: bool = False) -> nn.Module:
        """Load model from a given path"""
        raise NotImplementedError("Function to load model not implemented")

    def _get_model_paths(self,
                         train_config: TrainConfig,
                         n_models: int = None,
                         shuffle: bool = True) -> List:
        # Get path to load models
        folder_path = self.get_save_dir(train_config)
        model_paths = os.listdir(folder_path)
        if shuffle:
            model_paths = np.random.permutation(model_paths)
        total_models = len(model_paths) if n_models is None else n_models
        log(f"Available models: {total_models}")
        return model_paths, folder_path, total_models

    def get_models(self,
                   train_config: TrainConfig,
                   n_models: int = None,
                   on_cpu: bool = False,
                   shuffle: bool = True) -> List[nn.Module]:
        """
            Load models
        """
        # Get path to load models
        model_paths, folder_path, total_models = self._get_model_paths(
            train_config, n_models, shuffle)
        i = 0
        models = []
        with tqdm(total=total_models, desc="Loading models") as pbar:
            for mpath in model_paths:
                # Break reading if requested number of models is reached
                if i >= n_models:
                    break
                # Skip any directories we may stumble upon
                if os.path.isdir(os.path.join(folder_path, mpath)):
                    continue
                model = self.load_model(os.path.join(
                    folder_path, mpath), on_cpu=on_cpu)
                models.append(model)
                i += 1
                pbar.update()
        if len(models) == 0:
            raise ValueError("No models found in the given path")
        if n_models is not None and len(models) != n_models:
            warnings.warn(warning_string(
                f"\nNumber of models loaded ({len(models)}) is less than requested ({n_models})"))
        return np.array(models, dtype='object')

    def get_model_features(self,
                           train_config: TrainConfig,
                           attack_config: WhiteBoxAttackConfig,
                           n_models: int = None,
                           on_cpu: bool = False,
                           shuffle: bool = True):
        """
            Extract features for a given model
        """
        # Get path to load models
        model_paths, folder_path, total_models = self._get_model_paths(
            train_config, n_models, shuffle)
        i = 0
        feature_vectors = []
        with tqdm(total=total_models, desc="Loading models") as pbar:
            for mpath in model_paths:
                # Break reading if requested number of models is reached
                if i >= n_models:
                    break
                # Skip any directories we may stumble upon
                if os.path.isdir(os.path.join(folder_path, mpath)):
                    continue

                # Load model
                model = self.load_model(os.path.join(
                    folder_path, mpath), on_cpu=on_cpu)

                # Extract model features
                # Get model params, shift to GPU
                dims, feature_vector = get_weight_layers(model, attack_config)
                feature_vectors.append(feature_vector)

                # Update progress
                i += 1
                pbar.update(1)

        if len(feature_vectors) == 0:
            raise ValueError("No models found in the given path")
        if n_models is not None and len(feature_vectors) != n_models:
            warnings.warn(warning_string(
                f"\nNumber of models loaded ({len(feature_vectors)}) is less than requested ({n_models})"))

        feature_vectors = np.array(feature_vectors, dtype='object')
        return dims, feature_vectors
