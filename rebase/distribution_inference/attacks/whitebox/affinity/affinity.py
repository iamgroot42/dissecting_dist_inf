from typing import List
import os
import warnings
import torch as ch
import torch.nn as nn
from tqdm import tqdm

from distribution_inference.attacks.whitebox.affinity.models import AffinityMetaClassifier
from distribution_inference.config import WhiteBoxAttackConfig, DatasetConfig, TrainConfig
from distribution_inference.utils import warning_string, get_save_path, ensure_dir_exists
from distribution_inference.models.core import BaseModel
from distribution_inference.training.core import train
from distribution_inference.attacks.whitebox.affinity.utils import get_seed_data


class AffinityAttack:
    def __init__(self,
                 num_dim: int,
                 num_layers: int,
                 config: WhiteBoxAttackConfig):
        super().__init__(config)
        self.num_dim = num_dim
        self.num_layers = num_layers
        self.use_logit = not self.config.affinity_config.only_latent
        self.num_retain = self.config.affinity_config.num_retain
        if self.num_retain > 1 or self.num_retain < 0:
            raise ValueError(
                f"num_retain must be in [0, 1] when provided as a float, got {self.num_retain}")

    def _prepare_model(self):
        self.model = AffinityMetaClassifier(
            self.num_dim,
            self.num_layers,
            self.config.affinity_config)
        if self.config.gpu:
            self.model = self.model.cuda()

    def make_affinity_features(self,
                               models: List[BaseModel],
                               loaders: List,
                               train_config: TrainConfig,
                               detach: bool = True,
                               num_samples_use: int = None):
        """
            1. Extract data from given dataloders
            2. Extract model features on give data for all models
            3. Compute pair-wise cosine similarity across all datapoints
        """
        # Step 1: extract all data
        seed_data = get_seed_data(
            loaders, train_config,
            num_samples_use=num_samples_use)
        all_features = []
        for model in tqdm(models, desc="Building affinity matrix"):
            # Steps 2 & 3: get all model features and affinity scores
            affinity_feature = self._make_affinity_feature(
                model, seed_data,
                detach=detach)
            all_features.append(affinity_feature)
        return ch.stack(all_features, 0)

    def _make_affinity_feature(self,
                               model: BaseModel,
                               data,
                               detach: bool = True):
        """
            Construct affinity matrix per layer based on affinity scores
            for a given model. Model them in a way that does not
            require graph-based models.
        """
        # Build affinity graph for given model and data
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Start with getting layer-wise model features
        model_features = model(data, get_all=True, detach_before_return=detach)
        layerwise_features = []
        for i, feature in enumerate(model_features):
            scores = []
            # Pair-wise iteration of all data
            for i in range(len(data)-1):
                others = feature[i+1:]
                scores += cos(ch.unsqueeze(feature[i], 0), others)
            layerwise_features.append(ch.stack(scores, 0))

        # If asked to use logits, convert them to probability scores
        # And then consider them as-it-is (instead of pair-wise comparison)
        if self.use_logit:
            logits = model_features[-1]
            probs = ch.sigmoid(logits)
            layerwise_features.append(probs)

        concatenated_features = ch.stack(layerwise_features, 0)
        return concatenated_features

    def execute_attack(self, train_data, test_data, train_config: TrainConfig):
        """
            Define and train meta-classifier
        """
        # Prepare model
        self._prepare_model()

        # Create laoaders out of all the given data
        def get_loader(data, shuffle):
            return ch.utils.data.DataLoader(
                ch.utils.data.TensorDataset(data[0], data[1]),
                batch_size=train_config.batch_size,
                shuffle=shuffle)

        train_loader = get_loader(train_data, True)
        test_loader = get_loader(test_data, False)

        # Train model
        # For this attack, we have features
        # as if in normal form
        # All we need to do is define loaders and call
        # normal training functions from training.core
        self.model, (test_loss, test_acc) = train(self.model,
                                                  (train_loader, test_loader),
                                                  train_config=train_config)
        self.trained_model = True
        return test_acc

    def save_model(self,
                   data_config: DatasetConfig,
                   attack_specific_info_string: str):
        """
            Save model to disk.
        """
        if not self.trained_model:
            warnings.warn(warning_string(
                "\nAttack being saved without training."))
        if self.config.regression_config:
            model_dir = "affinity/regression"
        else:
            model_dir = "affinity/classification"
        save_path = os.path.join(
            get_save_path(),
            model_dir,
            data_config.name,
            data_config.prop)
        if self.config.regression_config is not None:
            save_path = os.path.join(save_path, str(data_config.value))

        # Make sure folder exists
        ensure_dir_exists(save_path)

        model_save_path = os.path.join(
            save_path, f"{attack_specific_info_string}.ch")
        ch.save(self.model.state_dict(), model_save_path)
