import torch as ch
import torch.nn as nn
from typing import List
from dataclasses import replace

import distribution_inference.attacks.whitebox.permutation as permutation
from distribution_inference.config import AffinityAttackConfig


class AffinityMetaClassifier(nn.Module):
    def __init__(self,
                 num_dim: int,
                 num_layers: int,
                 config: AffinityAttackConfig):
        super().__init__()
        self.num_dim = num_dim
        self.num_layers = num_layers
        self.only_latent = config.only_latent
        self.num_final = config.num_final
        self.final_act_size = self.num_final * self.num_layers
        self.models = []

        def make_small_model():
            return nn.Sequential(
                nn.Linear(self.num_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_final),
            )
        for _ in range(num_layers):
            self.models.append(make_small_model())
        self.models = nn.ModuleList(self.models)
        if not self.only_latent:
            self.final_layer = nn.Linear(self.num_final * self.num_layers, 1)

    def forward(self, x) -> ch.Tensor:
        # Get intermediate activations for each layer
        # Aggreage them to get a single feature vector
        all_acts = []
        for i, model in enumerate(self.models):
            all_acts.append(model(x[:, i]))
        all_accs = ch.cat(all_acts, 1)
        # Return pre-logit activations if requested
        if self.only_latent:
            return all_accs
        return self.final_layer(all_accs)


class WeightAndActMeta(nn.Module):
    """
        Combined meta-classifier that uses model weights as well as activation
        trends for property prediction.
    """
    def __init__(self, dims: List[int], num_dims: int,
                 num_layers: int, config: AffinityAttackConfig):
        super().__init__()
        self.dims = dims
        self.num_dims = num_dims
        self.num_layers = num_layers

        # Use version of config where only latent is used
        config = replace(config)
        config.use_latent = True

        # Affinity meta-classifier
        self.act_clf = AffinityMetaClassifier(
            num_dims, num_layers, config=config)
        # Weights meta-classifier
        self.weights_clf = permutation.PermInvModel(dims, only_latent=True)
        self.final_act_size = self.act_clf.final_act_size + \
            self.weights_clf.final_act_size
        # Layer to combine features from both
        self.combination_layer = nn.Linear(self.final_act_size, 1)

    def forward(self, w, x) -> ch.Tensor:
        # Output for weights
        weights = self.weights_clf(w)
        # Output for activations
        act = self.act_clf(x)
        # Combine them
        all_acts = ch.cat([act, weights], 1)
        return self.combination_layer(all_acts)
