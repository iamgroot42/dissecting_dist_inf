from typing import List
import os
import warnings
import torch as ch
import torch.nn as nn
from tqdm import tqdm

from distribution_inference.attacks.whitebox.core import Attack
from distribution_inference.attacks.whitebox.affinity.models import AffinityMetaClassifier
from distribution_inference.config import WhiteBoxAttackConfig, DatasetConfig, TrainConfig
from distribution_inference.utils import warning_string, get_save_path, ensure_dir_exists
from distribution_inference.models.core import BaseModel
from distribution_inference.training.core import train


class AffinityAttack(Attack):
    def __init__(self,
                 config: WhiteBoxAttackConfig):
        super().__init__(config)
        self.num_dim = None
        self.use_logit = not self.config.affinity_config.only_latent
        self.frac_retain_pairs = self.config.affinity_config.frac_retain_pairs
        self.num_layers = config.affinity_config.num_layers
        if type(self.num_layers) == int:
            self.num_layers = list(range(self.num_layers))
        if self.frac_retain_pairs > 1 or self.frac_retain_pairs < 0:
            raise ValueError(
                f"frac_retain_pairs must be in [0, 1] when provided as a float, got {self.frac_retain_pairs}")

    def _prepare_model(self):
        if self.num_dim is None:
            raise ValueError("num_dim must be set before calling _prepare_model")
        self.model = AffinityMetaClassifier(
            self.num_dim,
            self.num_layers,
            self.config.affinity_config)
        if self.config.gpu:
            self.model = self.model.cuda()

    def _collect_features(self,
                          model: BaseModel,
                          loader: ch.utils.data.DataLoader,
                          detach: bool = True):
        features = None
        for data in loader:
            if self.config.gpu:
                data = data.cuda()
            model_features = model(
                data, get_all=True,
                detach_before_return=detach)
            if features is None:
                features = model_features
            else:
                for i, mf in enumerate(model_features):
                    features[i].append(mf)
        return features

    def make_affinity_features(self,
                               models: List[BaseModel],
                               loader: ch.utils.data.DataLoader,
                               detach: bool = True):
        """
            1. Extract model features on give data for all models
            2. Compute pair-wise cosine similarity across all datapoints
        """
        all_features = []
        for model in tqdm(models, desc="Building affinity matrix"):
            # Steps 2 & 3: get all model features and affinity scores
            affinity_feature = self._make_affinity_feature(
                model, loader,
                detach=detach)
            all_features.append(affinity_feature)
        seed_data = ch.stack(all_features, 0)

        #TODO: Do something with frac_retain_pairs

        # Set num_dim
        self.num_dim = (len(seed_data) * (len(seed_data) - 1)) // 2

        return seed_data

    def _make_affinity_feature(self,
                               model: BaseModel,
                               loader: ch.utils.data.DataLoader,
                               detach: bool = True):
        """
            Construct affinity matrix per layer based on affinity scores
            for a given model. Model them in a way that does not
            require graph-based models.
        """
        # Build affinity graph for given model and data
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Start with getting layer-wise model features
        model_features = self._collect_features(model, loader, detach)
        layerwise_features = []
        for i, feature in enumerate(model_features):
            scores = []
            # Pair-wise iteration of all data
            for i in range(len(feature) - 1):
                others = feature[i+1:]
                scores += cos(ch.unsqueeze(feature[i], 0), others)
            layerwise_features.append(ch.stack(scores, 0))

        # If asked to use logits, convert them to probability scores
        # And then consider them as-it-is (instead of pair-wise comparison)
        if self.use_logit:
            logits = model_features[-1]
            probs = ch.sigmoid(logits).squeeze_(1)
            layerwise_features.append(probs)

        concatenated_features = ch.cat(layerwise_features, 0)
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
