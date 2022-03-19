import torch as ch
import torch.nn as nn

import distribution_inference.attacks.whitebox.permutation as permutation


class AffinityMetaClassifier(nn.Module):
    def __init__(self, num_dim: int, numlayers: int,
                 num_final: int = 16, only_latent: bool = False):
        super().__init__()
        self.num_dim = num_dim
        self.numlayers = numlayers
        self.only_latent = only_latent
        self.num_final = num_final
        self.final_act_size = num_final * self.numlayers
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
        for _ in range(numlayers):
            self.models.append(make_small_model())
        self.models = nn.ModuleList(self.models)
        if not self.only_latent:
            self.final_layer = nn.Linear(self.num_final * self.numlayers, 1)

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
    def __init__(self, dims: List[int], num_dims: int, num_layers: int):
        super().__init__()
        self.dims = dims
        self.num_dims = num_dims
        self.num_layers = num_layers
        self.act_clf = AffinityMetaClassifier(
            num_dims, num_layers, only_latent=True)
        self.weights_clf = permutation.PermInvModel(dims, only_latent=True)
        self.final_act_size = self.act_clf.final_act_size + \
            self.weights_clf.final_act_size
        self.combination_layer = nn.Linear(self.final_act_size, 1)

    def forward(self, w, x) -> ch.Tensor:
        # Output for weights
        weights = self.weights_clf(w)
        # Output for activations
        act = self.act_clf(x)
        # Combine them
        all_acts = ch.cat([act, weights], 1)
        return self.combination_layer(all_acts)
