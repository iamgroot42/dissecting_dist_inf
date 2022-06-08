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
                 config: AffinityAttackConfig,
                 num_logit: int = 0,
                 multi_class: bool = False):
        super().__init__()
        self.num_dim = num_dim
        self.num_layers = num_layers
        self.num_final = config.num_final
        self.multi_class = multi_class
        self.final_act_size = self.num_final * self.num_layers
        self.models = []
        self.num_logit = num_logit
        self.only_latent = config.only_latent
        self.layer_agnostic = config.layer_agnostic
        self.inner_dims = config.inner_dims
        self.num_rnn_layers = config.num_rnn_layers
        self.sequential_variant = config.sequential_variant
        self.shared_layerwise_params = config.shared_layerwise_params
        assert len(self.inner_dims) >= 1, "inner_dims must have at least 1 element"

        # Make inner model
        def make_small_model(dims):
            layers = [
                nn.Linear(dims, self.inner_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(self.inner_dims)):
                layers.append(
                    nn.Linear(self.inner_dims[i-1], self.inner_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.inner_dims[-1], self.num_final))
            return nn.Sequential(*layers)

        inside_dim = self.num_dim
        if self.shared_layerwise_params:
            # Shared model across all layers
            shared_model = make_small_model(inside_dim)
            for _ in range(num_layers):
                self.models.append(shared_model)
        else:
            # Make one model per feature layer
            for _ in range(num_layers):
                self.models.append(make_small_model(inside_dim))
        # If logits are also going to be provided, have a model for them as well
        if self.num_logit > 0 and not self.multi_class:
            self.models.append(make_small_model(self.num_logit))
        self.models = nn.ModuleList(self.models)

        num_eff_layers = self.num_layers
        num_eff_layers += 1 if (self.num_logit > 0 and not self.multi_class) else 0
        if not (self.layer_agnostic or self.sequential_variant):
            # Linear layer on top of concatenated embeddings
            self.final_layer = nn.Linear(self.num_final * num_eff_layers, 1)
        if self.sequential_variant:
            # Sequential model, process one embedding at a time
            self.recurrent_layer = nn.GRU(
                input_size=self.num_final,
                hidden_size=self.num_final,
                num_layers=self.num_rnn_layers,
                batch_first=False,
                bidirectional=False)
            self.final_layer = nn.Linear(self.num_final, 1)

    def forward(self, x, get_latent: bool = False) -> ch.Tensor:
        # Get intermediate activations for each layer
        # Aggregate them to get a single feature vector
        all_acts = []
        for i, model in enumerate(self.models):
            all_acts.append(model(x[i]))
        if self.sequential_variant:
            all_acts = ch.stack(all_acts)
        else:
            all_acts = ch.cat(all_acts, 1)
        # Return pre-logit activations if requested
        if self.only_latent:
            return all_acts
        # If agnostic to number of layers, average over given layer representations
        # and return
        if self.layer_agnostic:
            return ch.mean(all_acts, 1).unsqueeze(1)
        if self.sequential_variant:
            # Sequential model, process one embedding at a time
            # Initial hidden state defaults to 0s when not provided
            all_acts, _ = self.recurrent_layer(all_acts)
            # Reshape to be compatible with self.final_layer
            all_acts = all_acts[-1, :]
            all_acts = all_acts.contiguous()
        if get_latent:
            return all_acts
        return self.final_layer(all_acts)


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
