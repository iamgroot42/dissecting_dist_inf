import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn

from distribution_inference.utils import check_if_inside_cluster
from distribution_inference.config import DatasetConfig, TrainConfig
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
    def __init__(self, name: str, data_path: str, models_path: str, properties: list, values: dict, property_focus : dict):
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

    def get_model(self) -> nn.Module:
        raise NotImplementedError(f"Implement method to model for {self.name} dataset")

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

        # Either set ds_train and ds_val here
        # Or set them inside get_loaders

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

    def get_save_path(self, train_config: TrainConfig, name: str) -> str:
        """
            Function to get prefix + name for saving
            the model.
        """
        raise NotImplementedError("Function to fetch model save path not implemented")
