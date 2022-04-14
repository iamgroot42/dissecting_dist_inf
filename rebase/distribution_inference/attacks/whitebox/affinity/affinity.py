from typing import List
import os
import warnings
import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from distribution_inference.attacks.whitebox.core import Attack, BasicDataset
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
        if self.frac_retain_pairs > 1 or self.frac_retain_pairs < 0:
            raise ValueError(
                f"frac_retain_pairs must be in [0, 1] when provided as a float, got {self.frac_retain_pairs}")
        self.retained_pairs = None

    def _prepare_model(self):
        if self.num_dim is None:
            raise ValueError(
                "num_dim must be set before calling _prepare_model")
        self.model = AffinityMetaClassifier(
            self.num_dim,
            self.num_layers,
            self.config.affinity_config,
            self.num_logit_features,)
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
                features = [[x] for x in model_features]
            else:
                for i, mf in enumerate(model_features):
                    features[i].append(mf)
        features = [ch.cat(x, 0) for x in features]
        return features

    def register_seed_data(self, seed_data_ds: BasicDataset):
        self.seed_data_ds = seed_data_ds

    def make_affinity_features(self,
                               models: List[BaseModel],
                               loader: ch.utils.data.DataLoader,
                               detach: bool = True):
        """
            Construct affinity matrix per layer based on affinity scores
            for a given model. Model them in a way that does not
            require graph-based models.
        """
        all_features = []
        for model in tqdm(models, desc="Building affinity matrix"):
            # Steps 2 & 3: get all model features and affinity scores
            affinity_feature, num_features, num_logit_features, num_layers = self._make_affinity_feature(
                model, loader,
                detach=detach)
            all_features.append(affinity_feature)

        seed_data = all_features

        # Retain only a fraction of all pairs
        if self.frac_retain_pairs < 1 and self.retained_pairs is None:
            # First-time (on train data)
            num_eff_layers = num_layers if self.use_logit else num_layers - 1
            # Collect STD values across models (per layer)
            std_values = []
            for i in range(num_eff_layers):
                std_values.append(
                    ch.std(ch.stack([x[i] for x in seed_data]), 0))
            # Sort std values in descending order (per layer) and store indices
            std_values = map(lambda x: x.sort(
                descending=True)[1], std_values)
            # Pick top frac_retain_pairs values indices per layer
            n_features_retain = int(self.frac_retain_pairs*num_features)
            self.retained_pairs = list(
                map(lambda x: x[:n_features_retain].cpu().numpy(), std_values))
            # Make a count array to keep track of top pairs per layer
            count_array = np.zeros((num_features,))
            for i in range(num_eff_layers):
                count_array[self.retained_pairs[i]] += 1
            # Ouf of these, pick the top self.frac_retain_pairs
            self.retained_pairs = np.sort(np.argsort(
                count_array)[::-1][:n_features_retain])

            # Consider only self.retained_pairs indices
            # For future cases, _make_affinity_feature() will handle
            # the case where self.retained_pairs is not None
            def selection_lambda(x):
                selected = [x[i][self.retained_pairs]
                            for i in range(num_eff_layers)]
                if self.use_logit:
                    selected.append(x[-1])
                return selected
            seed_data = list(map(selection_lambda, seed_data))

        # Set num_dim (returned value adjusted to handle retained_pairs)
        self.num_dim = num_features
        self.num_logit_features = num_logit_features
        self.num_layers = num_layers

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
            # If got to logit, break- will handle outside of loop
            if self.use_logit and i == len(model_features) - 1:
                break
            scores = []
            pairs_so_far = 0
            # Pair-wise iteration of all data
            for j in range(feature.shape[0]-1):
                others = feature[j+1:]
                if self.retained_pairs is not None:
                    # Optimization- if self.retained_pairs is not None, only
                    # consider those indices for computing cosine similarity
                    relevant_pairs = self.retained_pairs - pairs_so_far
                    relevant_pairs = relevant_pairs[relevant_pairs >= 0]
                    relevant_pairs = relevant_pairs[relevant_pairs < len(others)]
                    # relevant_pairs -= pairs_so_far
                    pairs_so_far += len(others)
                    others = others[relevant_pairs]
                if len(others) != 0:
                    scores += cos(ch.unsqueeze(feature[j], 0), others)
            stacked_scores = ch.stack(scores, 0)
            # if self.retained_pairs is not None:
            #     stacked_scores = stacked_scores[self.retained_pairs]
            layerwise_features.append(stacked_scores)

        num_layers = len(layerwise_features)
        # If asked to use logits, convert them to probability scores
        # And then consider them as-it-is (instead of pair-wise comparison)
        num_logit_features = 0
        if self.use_logit:
            logits = model_features[-1]
            probs = ch.sigmoid(logits).squeeze_(1)
            layerwise_features.append(probs)
            num_logit_features = probs.shape[0]

        num_features = layerwise_features[0].shape[0]
        return layerwise_features, num_features, num_logit_features, num_layers

    def execute_attack(self, train_data, test_data):
        """
            Define and train meta-classifier
        """
        # Prepare model
        self._prepare_model()

        # Make new train_config to train meta-classifier
        train_config = TrainConfig(
            data_config=None,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            verbose=True,
            weight_decay=self.config.weight_decay,
            get_best=True,
            expect_extra=False,
        )

        def collate_fn(data):
            features, labels = zip(*data)
            # Combine them per-layer
            x = [[] for _ in range(len(features[0]))]
            for feature in features:
                for i, layer_feature in enumerate(feature):
                    x[i].append(layer_feature)

            x = [ch.stack(x_, 0) for x_ in x]
            y = ch.tensor(labels).float()

            return x, y

        class BasicDataset(ch.utils.data.Dataset):
            def __init__(self, X, Y):
                self.X = X
                self.Y = Y
                assert len(self.X) == len(self.Y)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]

        # Create laoaders out of all the given data
        def get_loader(data, shuffle):
            ds = BasicDataset(data[0], data[1])
            return ch.utils.data.DataLoader(
                ds, batch_size=self.config.batch_size,
                collate_fn=collate_fn,
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
                                                  train_config=train_config,
                                                  input_is_list=True)
        self.trained_model = True
        return test_acc * 100

    def save_model(self,
                   data_config: DatasetConfig,
                   attack_specific_info_string: str):
        """
            Save model to disk.
        """
        if not self.trained_model:
            warnings.warn(warning_string(
                "\nAttack being saved without training.\n"))
        if self.config.regression_config:
            model_dir = "affinity/regression"
        else:
            model_dir = "affinity/classification"
        save_path = os.path.join(
            get_save_path(),
            model_dir,
            data_config.name,
            data_config.prop)
        if self.config.regression_config is None:
            save_path = os.path.join(save_path, str(data_config.value))

        # Make sure folder exists
        ensure_dir_exists(save_path)

        model_save_path = os.path.join(
            save_path, f"{attack_specific_info_string}.ch")
        ch.save({
            "model": self.model.state_dict(),
            "seed_data_ds": self.seed_data_ds,
            "retained_pairs": self.retained_pairs,
            "num_dim": self.num_dim,
            "num_logit_features": self.num_logit_features,
            "num_layers": self.num_layers
        }, model_save_path)

    def load_model(self, load_path: str):
        checkpoint = ch.load(load_path)
        self.model.load_state_dict(checkpoint["model"])
        self.seed_data_ds = checkpoint["seed_data_ds"]
        self.retained_pairs = checkpoint["retained_pairs"]
        self.num_dim = checkpoint["num_dim"]
        self.num_logit_features = checkpoint["num_logit_features"]
        self.num_layers = checkpoint["num_layers"]
