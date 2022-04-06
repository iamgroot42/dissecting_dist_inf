import torch as ch
import torch.nn as nn
from typing import List, Tuple


class PermInvConvModel(nn.Module):
    def __init__(self, dims: Tuple,
                 inside_dims=[64, 8], n_classes=2,
                 dropout=0.5, only_latent=False,
                 scale_invariance=False):
        super(PermInvConvModel, self).__init__()
        self.dim_channels, self.dim_kernels = dims
        self.only_latent = only_latent
        self.scale_invariance = scale_invariance

        assert len(self.dim_channels) == len(
            self.dim_kernels), "Kernel size information missing!"

        self.dropout = dropout
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        # One network per kernel location
        def make_mini(y):
            layers = [
                nn.Dropout(self.dropout),
                nn.Linear(y, inside_dims[0]),
                nn.ReLU(),
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        # For each layer of kernels
        for i, dim in enumerate(self.dim_channels):
            # +1 for bias
            # prev_layer for previous layer
            if i > 0:
                prev_layer = inside_dims[-1] * dim

            # For each pixel in the kernel
            # Concatenated along pixels in kernel
            self.layers.append(
                make_mini(prev_layer + (1 + dim) * self.dim_kernels[i]))

        self.layers = nn.ModuleList(self.layers)

        # Experimental param: if scale invariance, also store overall scale multiplier
        dim_for_scale_invariance = 1 if self.scale_invariance else 0

        # Final network to combine them all
        # layer representations together
        if not self.only_latent:
            self.rho = nn.Linear(
                inside_dims[-1] * len(self.dim_channels) +
                dim_for_scale_invariance,
                n_classes)

    def forward(self, params):
        reps = []
        for_prev = None

        if self.scale_invariance:
            # Keep track of multiplier (with respect to smallest nonzero weight) across layers
            # For ease of computation, we will store in log scale
            scale_invariance_multiplier = ch.ones((params[0].shape[0]))
            # Shift to appropriate device
            scale_invariance_multiplier = scale_invariance_multiplier.to(
                params[0].device)

        for param, layer in zip(params, self.layers):
            # shape: (n_samples, n_pixels_in_kernel, channels_out, channels_in)
            prev_shape = param.shape

            # shape: (n_samples, channels_out, n_pixels_in_kernel, channels_in)
            param = param.transpose(1, 2)

            # shape: (n_samples, channels_out, n_pixels_in_kernel * channels_in)
            param = ch.flatten(param, 2)

            if self.scale_invariance:
                # TODO: Vectorize
                for i in range(param.shape[0]):
                    # Scaling mechanism- pick largest weight, scale weights
                    # such that largest weight becomes 1
                    scale_factor = ch.norm(param[i])
                    scale_invariance_multiplier[i] += ch.log(scale_factor)
                    # Scale parameter matrix (if it's not all zeros)
                    if scale_factor != 0:
                        param[i] /= scale_factor

            if for_prev is None:
                param_eff = param
            else:
                prev_rep = for_prev.repeat(1, param.shape[1], 1)
                param_eff = ch.cat((param, prev_rep), -1)

            # shape: (n_samples * channels_out, channels_in_eff)
            param_eff = param_eff.view(
                param_eff.shape[0] * param_eff.shape[1], -1)

            # shape: (n_samples * channels_out, inside_dims[-1])
            pp = layer(param_eff.reshape(-1, param_eff.shape[-1]))

            # shape: (n_samples, channels_out, inside_dims[-1])
            pp = pp.view(prev_shape[0], prev_shape[2], -1)

            # shape: (n_samples, inside_dims[-1])
            processed = ch.sum(pp, -2)

            # Store previous layer's representation
            # shape: (n_samples, channels_out * inside_dims[-1])
            for_prev = pp.view(pp.shape[0], -1)

            # shape: (n_samples, 1, channels_out * inside_dims[-1])
            for_prev = for_prev.unsqueeze(-2)

            # Store representation for this layer
            reps.append(processed)

        reps = ch.cat(reps, 1)

        # Add invariance multiplier
        if self.scale_invariance:
            scale_invariance_multiplier = ch.unsqueeze(
                scale_invariance_multiplier, 1)
            reps = ch.cat((reps, scale_invariance_multiplier), 1)

        if self.only_latent:
            return reps

        logits = self.rho(reps)
        return logits


class PermInvModel(nn.Module):
    def __init__(self, dims: List[int],
                 inside_dims: List[int] = [64, 8],
                 n_classes: int = 2,
                 dropout: float = 0.5,
                 only_latent: bool = False):
        super(PermInvModel, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.only_latent = only_latent
        self.final_act_size = inside_dims[-1] * len(dims)
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        def make_mini(y):
            layers = [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        for i, dim in enumerate(self.dims):
            # +1 for bias
            # prev_layer for previous layer
            # input dimension per neuron
            if i > 0:
                prev_layer = inside_dims[-1] * dim
            self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = nn.ModuleList(self.layers)

        if not self.only_latent:
            # Final network to combine them all together
            self.rho = nn.Linear(self.final_act_size, n_classes)

    def forward(self, params) -> ch.Tensor:
        reps = []
        prev_layer_reps = None
        is_batched = len(params[0].shape) > 2

        for param, layer in zip(params, self.layers):

            # Case where data is batched per layer
            if is_batched:
                if prev_layer_reps is None:
                    param_eff = param
                else:
                    prev_layer_reps = prev_layer_reps.repeat(
                        1, param.shape[1], 1)
                    param_eff = ch.cat((param, prev_layer_reps), -1)

                prev_shape = param_eff.shape
                processed = layer(param_eff.view(-1, param_eff.shape[-1]))
                processed = processed.view(
                    prev_shape[0], prev_shape[1], -1)

            else:
                if prev_layer_reps is None:
                    param_eff = param
                else:
                    prev_layer_reps = prev_layer_reps.repeat(param.shape[0], 1)
                    # Include previous layer representation
                    param_eff = ch.cat((param, prev_layer_reps), -1)
                processed = layer(param_eff)

            # Store this layer's representation
            reps.append(ch.sum(processed, -2))

            # Handle per-data/batched-data case together
            if is_batched:
                prev_layer_reps = processed.view(processed.shape[0], -1)
            else:
                prev_layer_reps = processed.view(-1)
            prev_layer_reps = ch.unsqueeze(prev_layer_reps, -2)

        if is_batched:
            reps_c = ch.cat(reps, 1)
        else:
            reps_c = ch.unsqueeze(ch.cat(reps), 0)

        if self.only_latent:
            return reps_c

        logits = self.rho(reps_c)
        return logits


class FullPermInvModel(nn.Module):
    def __init__(self,
                 dims: List[List],
                 inside_dims: List[int] = [64, 8],
                 n_classes: int = 2,
                 dropout: float = 0.5):
        super(FullPermInvModel, self).__init__()
        # Extract relevant dimensions from given dims
        dims_conv, dims_fc = dims
        self.dim_channels, self.dim_kernels, self.middle_dim = dims_conv
        self.dims_fc = dims_fc

        self.total_layers = len(self.dim_channels) + len(dims_fc)

        assert len(self.dim_channels) == len(
            self.dim_kernels), "Kernel size information missing!"

        self.dropout = dropout
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        # One network per kernel location
        def make_mini(y, add_drop=False):
            layers = []
            if add_drop:
                layers += [nn.Dropout(self.dropout)]
            layers += [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        # For each layer
        for i in range(self.total_layers):
            is_conv = i < len(self.dim_channels)

            if is_conv:
                dim = self.dim_channels[i]
            else:
                dim = self.dims_fc[i - len(self.dim_channels)]

            # +1 for bias
            # prev_layer for previous layer
            if i > 0:
                prev_layer = inside_dims[-1] * dim

            if is_conv:
                # Concatenated along pixels in kernel
                self.layers.append(
                    make_mini(prev_layer + (1 + dim) * self.dim_kernels[i],
                              add_drop=True))
            else:
                # FC layer
                if i == len(self.dim_channels):
                    prev_layer = inside_dims[-1] * self.middle_dim
                self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = nn.ModuleList(self.layers)

        # Final network to combine them all
        # layer representations together
        self.rho = nn.Linear(
            inside_dims[-1] * self.total_layers, n_classes)

    def forward(self, params: List[ch.Tensor]) -> ch.Tensor:
        reps = []
        for_prev = None
        i = 0

        for i, (param, layer) in enumerate(zip(params, self.layers)):
            is_conv = i < len(self.dim_channels)

            if is_conv:
                # Convolutional layer

                # shape: (n_samples, n_pixels_in_kernel, channels_out, channels_in)
                prev_shape = param.shape

                # shape: (n_samples, channels_out, n_pixels_in_kernel, channels_in)
                param = param.transpose(1, 2)

                # shape: (n_samples, channels_out, n_pixels_in_kernel * channels_in)
                param = ch.flatten(param, 2)

            # Concatenate previous layer representation, if available
            if for_prev is None:
                param_eff = param
            else:
                prev_rep = for_prev.repeat(1, param.shape[1], 1)
                param_eff = ch.cat((param, prev_rep), -1)

            if is_conv:
                # Convolutional layer

                # shape: (n_samples * channels_out, channels_in_eff)
                param_eff = param_eff.view(
                    param_eff.shape[0] * param_eff.shape[1], -1)

                # print(param_eff.reshape(-1, param_eff.shape[-1]).shape)

                # shape: (n_samples * channels_out, inside_dims[-1])
                pp = layer(param_eff.reshape(-1, param_eff.shape[-1]))

                # shape: (n_samples, channels_out, inside_dims[-1])
                pp = pp.view(prev_shape[0], prev_shape[2], -1)

            else:
                # FC layer
                prev_shape = param_eff.shape
                pp = layer(param_eff.view(-1, param_eff.shape[-1]))
                pp = pp.view(prev_shape[0], prev_shape[1], -1)

            processed = ch.sum(pp, -2)

            # Store previous layer's representation
            for_prev = pp.view(pp.shape[0], -1)
            for_prev = for_prev.unsqueeze(-2)

            # Store representation for this layer
            reps.append(processed)

        reps = ch.cat(reps, 1)
        logits = self.rho(reps)
        return logits
