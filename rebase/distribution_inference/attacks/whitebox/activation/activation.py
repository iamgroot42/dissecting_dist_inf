import torch as ch
import torch.nn as nn


class ActivationMetaClassifier(nn.Module):
    def __init__(self, n_samples, dims, reduction_dims,
                 inside_dims=[64, 16],
                 n_classes=2, dropout=0.2):
        super(ActivationMetaClassifier, self).__init__()
        self.n_samples = n_samples
        self.dims = dims
        self.reduction_dims = reduction_dims
        self.dropout = dropout
        self.layers = []

        assert len(dims) == len(reduction_dims), "dims and reduction_dims must be same length"

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        def make_mini(y, last_dim):
            layers = [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(
                nn.Linear(inside_dims[len(inside_dims)-1], last_dim))
            layers.append(nn.ReLU())

            return nn.Sequential(*layers)

        # Reducing each activation into smaller representations
        for ld, dim in zip(self.reduction_dims, self.dims):
            self.layers.append(make_mini(dim, ld))

        self.layers = nn.ModuleList(self.layers)

        # Final layer to concatenate all representations across examples
        self.rho = nn.Sequential(
            nn.Linear(sum(self.reduction_dims) * self.n_samples, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
        )

    def forward(self, params):
        reps = []
        for param, layer in zip(params, self.layers):
            processed = layer(param.view(-1, param.shape[2]))
            # Store this layer's representation
            reps.append(processed)

        reps_c = ch.cat(reps, 1)
        reps_c = reps_c.view(-1, self.n_samples * sum(self.reduction_dims))

        logits = self.rho(reps_c)
        return logits
