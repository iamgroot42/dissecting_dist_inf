"""
    Large collection of utility functions and classes that are
    shared across all datasets. In the long run, we would have a common outer
    structure for all datasets, with dataset-specific configuration files.
"""
import torch as ch
import numpy as np
from os import environ
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR
from robustness.tools import folder
from robustness.tools.misc import log_statement

from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log(x):
    print(f"{bcolors.WARNING}%s{bcolors.ENDC}" % x)


class DataPaths:
    def __init__(self, name, data_path, stats_path):
        self.name = name
        self.data_path = data_path
        self.dataset = self.dataset_type(data_path)
        self.models = {'nat': None, 'l1': None,
                       'l2': None, 'temp': None, 'linf': None}
        self.model_prefix = {}
        self.stats_path = stats_path

    def get_dataset(self):
        return self.dataset

    def get_model(self, m_type, arch='resnet50', parallel=False):
        model_path = self.models.get(m_type, None)
        if not model_path:
            model_path = m_type
        else:
            model_path = self.model_prefix[arch] + self.models[m_type]
        model_kwargs = {
            'arch': arch,
            'dataset': self.dataset,
            'resume_path': model_path,
            'parallel': parallel
        }
        model, _ = make_and_restore_model(**model_kwargs)
        model.eval()
        return model

    def get_stats(self, m_type, arch='resnet50'):
        stats_path = os.path.join(self.stats_path, arch, m_type, "stats")
        return get_stats(stats_path)

    def get_deltas(self, m_type, arch='resnet50', numpy=False):
        ext = ".npy" if numpy else ".txt"
        deltas_path = os.path.join(
            self.stats_path, arch, m_type, "deltas" + ext)
        return get_sensitivities(deltas_path, numpy=numpy)


class BinaryCIFAR(DataPaths):
    def __init__(self, path):
        self.dataset_type = GenericBinary
        super(BinaryCIFAR, self).__init__('binary_cifar10', path, None)


class CIFAR10(DataPaths):
    def __init__(self, data_path=None):
        self.dataset_type = CIFAR
        datapath = "/p/adversarialml/as9rw/datasets/cifar10" if data_path is None else data_path
        super(CIFAR10, self).__init__('cifar10',
                                      datapath,
                                      "/p/adversarialml/as9rw/cifar10_stats/")
        self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_cifar10/"
        self.model_prefix['densenet169'] = "/p/adversarialml/as9rw/models_cifar10_densenet/"
        self.model_prefix['vgg19'] = "/p/adversarialml/as9rw/models_cifar10_vgg/"
        self.models['nat'] = "cifar_nat.pt"
        self.models['linf'] = "cifar_linf_8.pt"
        self.models['l2'] = "cifar_l2_0_5.pt"


class RobustCIFAR10(DataPaths):
    def __init__(self, datapath, stats_prefix):
        self.dataset_type = RobustCIFAR
        super(RobustCIFAR10, self).__init__('robustcifar10',
                                            datapath, stats_prefix)


class SVHN10(DataPaths):
    def __init__(self):
        self.dataset_type = SVHN
        super(SVHN10, self).__init__('svhn',
                                     "/p/adversarialml/as9rw/datasets/svhn",
                                     "/p/adversarialml/as9rw/svhn_stats/")
        self.model_prefix['vgg16'] = "/p/adversarialml/as9rw/models_svhn_vgg/"
        self.models['nat'] = "svhn_nat.pt"
        self.models['linf'] = "svhn_linf_4.pt"
        self.models['l2'] = "svhn_l2_0_5.pt"


class ImageNet1000(DataPaths):
    def __init__(self, data_path=None):
        self.dataset_type = ImageNet
        datapath = "/p/adversarialml/as9rw/datasets/imagenet/" if data_path is None else data_path
        super(ImageNet1000, self).__init__('imagenet1000',
                                           datapath,
                                           "/p/adversarialml/as9rw/imagenet_stats/")
        self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_imagenet/"
        self.models['nat'] = "imagenet_nat.pt"
        self.models['l2'] = "imagenet_l2_3_0.pt"
        self.models['linf'] = "imagenet_linf_4.pt"


def read_given_dataset(data_path):
    train_transform = transforms.Compose([])

    train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
    train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
    train_set = folder.TensorDataset(
        train_data, train_labels, transform=train_transform)

    X, Y = [], []
    for i in range(len(train_set)):
        X.append(train_set[i][0])
        Y.append(train_set[i][1].numpy())
    return (X, Y)


def scaled_values(val, mean, std, eps=1e-10):
    return (val - np.repeat(np.expand_dims(mean, 1), val.shape[1], axis=1)) / (np.expand_dims(std, 1) + eps)


def load_all_loader_data(data_loader):
    images, labels = [], []
    for (image, label) in data_loader:
        images.append(image)
        labels.append(label)
    images = ch.cat(images)
    labels = ch.cat(labels)
    return (images, labels)


def load_all_data(ds):
    batch_size = 512
    _, test_loader = ds.make_loaders(
        batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
    return load_all_loader_data(test_loader)


def get_sensitivities(path, numpy=False):
    log_statement("==> Loading Delta Values")
    # Directly load, if numpy array
    if numpy:
        return np.load(path)
    # Process, if text file
    features = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            values = np.array([float(x) for x in line.rstrip('\n').split(',')])
            features.append(values)
    return np.array(features)


def best_target_image(mat, which=0):
    sum_m = []
    for i in range(mat.shape[1]):
        mat_interest = mat[mat[:, i] != np.inf, i]
        sum_m.append(np.average(np.abs(mat_interest)))
    best = np.argsort(sum_m)
    return best[which]


def get_statistics(diff):
    l1_norms = ch.sum(ch.abs(diff), dim=1)
    l2_norms = ch.norm(diff, dim=1)
    linf_norms = ch.max(ch.abs(diff), dim=1)[0]
    return (l1_norms, l2_norms, linf_norms)


def get_stats(base_path):
    mean = np.load(os.path.join(base_path, "feature_mean.npy"))
    std = np.load(os.path.join(base_path, "feature_std.npy"))
    return mean, std


def get_logits_layer_name(arch):
    if "vgg" in arch:
        return "module.model.classifier.weight"
    elif "resnet" in arch:
        return "module.model.fc.weight"
    elif "densenet" in arch:
        return "module.model.linear.weight"
    return None


class SpecificLayerModel(ch.nn.Module):
    def __init__(self, model, layer_index):
        super(SpecificLayerModel, self).__init__()
        self.model = model
        self.layer_index = layer_index

    def forward(self, x):
        logits, _ = self.model(x, this_layer_input=self.layer_index)
        return logits


class MadryToNormal:
    def __init__(self, model, fake_relu=False):
        self.model = model
        self.fake_relu = fake_relu
        self.model.eval()

    def __call__(self, x):
        logits, _ = self.model(x, fake_relu=self.fake_relu)
        return logits

    def eval(self):
        return self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()


def classwise_pixelwise_stats(loader, num_classes=10, classwise=False):
    images, labels = load_all_loader_data(loader)
    if not classwise:
        return ch.mean(images, 0), ch.std(images, 0)
    means, stds = [], []
    for i in range(num_classes):
        specific_images = images[labels == i]
        means.append(ch.mean(specific_images, 0))
        stds.append(ch.std(specific_images, 0))
    return means, stds


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Input size: [batch, n_features]
        # Output size: [batch, 3, 32, 32]
        # Expects 48, 4, 4
        self.dnn = nn.Sequential(
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU())
        self.decoder = nn.Sequential(
            # [batch, 24, 8, 8]
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # [batch, 12, 16, 16]
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # [batch, 3, 32, 32]
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ = self.dnn(x)
        x_ = x_.view(x_.shape[0], 48, 4, 4)
        return self.decoder(x_)


class BasicDataset(ch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def compute_delta_values(logits, weights, actual_label=None):
    # Iterate through all possible classes, calculate flip probabilities
    actual_label = ch.argmax(logits)
    numerator = (logits[actual_label] - logits).unsqueeze(1)
    denominator = weights - weights[actual_label]
    numerator = numerator.repeat(1, denominator.shape[1])
    delta_values = ch.div(numerator, denominator)
    delta_values[actual_label] = np.inf
    return delta_values


def get_these_params(model, identifier):
    for name, param in model.state_dict().items():
        if name == identifier:
            return param
    return None


def flash_utils(args):
    log_statement("==> Arguments:")
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))


class MNISTFlatModel(nn.Module):
    def __init__(self):
        super(MNISTFlatModel, self).__init__()
        n_feat = 28 * 28
        self.dnn = nn.Sequential(
            nn.Linear(n_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10))

    def forward(self, x):
        x_ = x.view(x.shape[0], -1)
        return self.forward(x_)


def filter(df, condition, ratio, verbose=True):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            return pd.concat([df.iloc[qualify], df.iloc[nqi]])
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            return pd.concat([df.iloc[qi], df.iloc[notqualify]])
        return df.iloc[notqualify]


def get_cropped_faces(cropmodel, x):
    def renormalize(z): return (z * 0.5) + 0.5

    images = [Image.fromarray(
        (255 * np.transpose(renormalize(x_.numpy()), (1, 2, 0))).astype('uint8')) for x_ in x]
    crops = cropmodel(images)

    x_cropped = []
    indices = []
    for j, cr in enumerate(crops):
        if cr is not None:
            x_cropped.append(cr)
            indices.append(j)

    return ch.stack(x_cropped, 0), indices


# Function to extract model parameters
def get_weight_layers(m, normalize=False, transpose=True,
                      first_n=np.inf, start_n=0,
                      custom_layers=None,
                      conv=False, include_all=False,
                      prune_mask=[]):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Sort and store desired layers, if specified
    custom_layers = sorted(custom_layers) if custom_layers is not None else None

    track = 0
    for name, param in m.named_parameters():
        if "weight" in name:

            param_data = param.data.detach().cpu()

            # Apply pruning masks if provided
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose:
                param_data = param_data.T

            weights.append(param_data)
            if conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

        # Assume each layer has weight & bias
        i += 1

        if custom_layers is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

            # Break if all layers were processed
            if len(custom_layers) == j // 2:
                break

    if custom_layers is not None and len(custom_layers) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    if normalize:
        min_w = min([ch.min(x).item() for x in weights])
        max_w = max([ch.max(x).item() for x in weights])
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
        weights = [w / max_w for w in weights]

    cctd = []
    for w, b in zip(weights, biases):
        if conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = ch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    if conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd


class PermInvConvModel(nn.Module):
    def __init__(self, dim_channels, dim_kernels,
                 inside_dims=[64, 8], n_classes=2,
                 dropout=0.5, only_latent=False,
                 scale_invariance=False):
        super(PermInvConvModel, self).__init__()
        self.dim_channels = dim_channels
        self.dim_kernels = dim_kernels
        self.only_latent = only_latent
        self.scale_invariance = scale_invariance

        assert len(dim_channels) == len(
            dim_kernels), "Kernel size information missing!"

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
                make_mini(prev_layer + (1 + dim) * dim_kernels[i]))

        self.layers = nn.ModuleList(self.layers)

        # Experimental param: if scale invariance, also store overall scale multiplier
        dim_for_scale_invariance = 1 if self.scale_invariance else 0

        # Final network to combine them all
        # layer representations together
        if not self.only_latent:
            self.rho = nn.Linear(
                inside_dims[-1] * len(self.dim_channels) + dim_for_scale_invariance,
                n_classes)

    def forward(self, params):
        reps = []
        for_prev = None

        if self.scale_invariance:
            # Keep track of multiplier (with respect to smallest nonzero weight) across layers
            # For ease of computation, we will store in log scale
            scale_invariance_multiplier = ch.ones((params[0].shape[0]))
            # Shift to appropriate device
            scale_invariance_multiplier = scale_invariance_multiplier.to(params[0].device)

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
            scale_invariance_multiplier = ch.unsqueeze(scale_invariance_multiplier, 1)
            reps = ch.cat((reps, scale_invariance_multiplier), 1)

        if self.only_latent:
            return reps

        logits = self.rho(reps)
        return logits


class PermInvModel(nn.Module):
    def __init__(self, dims: List[int], inside_dims: List[int] = [64, 8],
                 n_classes: int = 2, dropout: float = 0.5,
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
    def __init__(self, dims, middle_dim, dim_channels, dim_kernels,
                 inside_dims=[64, 8], n_classes=2, dropout=0.5):
        super(FullPermInvModel, self).__init__()
        self.dim_channels = dim_channels
        self.dim_kernels = dim_kernels
        self.middle_dim = middle_dim
        self.dims = dims
        self.total_layers = len(dim_channels) + len(dims)

        assert len(dim_channels) == len(
            dim_kernels), "Kernel size information missing!"

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
                dim = self.dims[i - len(self.dim_channels)]

            # +1 for bias
            # prev_layer for previous layer
            if i > 0:
                prev_layer = inside_dims[-1] * dim

            if is_conv:
                # Concatenated along pixels in kernel
                self.layers.append(
                    make_mini(prev_layer + (1 + dim) * dim_kernels[i], add_drop=True))
            else:
                # FC layer
                if i == len(self.dim_channels):
                    prev_layer = inside_dims[-1] * middle_dim
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


class CombinedPermInvModel(nn.Module):
    def __init__(self, dims, dim_channels, dim_kernels,
                 inside_dims=[64, 8], n_classes=2, dropout=0.5):
        super(CombinedPermInvModel, self).__init__()
        # Model for convolutional layers
        self.conv_perm = PermInvConvModel(
            dim_channels, dim_kernels, inside_dims,
            n_classes, dropout, only_latent=True)
        # Model for FC layers
        self.fc_perm = PermInvModel(
            dims, inside_dims, n_classes,
            dropout, only_latent=True)

        self.n_conv_layers = len(dim_channels)
        self.n_fc_layers = len(dims)

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        n_layers = self.n_conv_layers + self.n_fc_layers
        self.rho = nn.Linear(inside_dims[-1] * n_layers, n_classes)

    def forward(self, x):
        # First n_conv_layers are for CONV model
        conv_latent = self.conv_perm(x[:self.n_conv_layers])
        # Layers after that are for FC model
        fc_latent = self.fc_perm(x[-self.n_fc_layers:])

        # Concatenate feature representations
        latent = ch.cat((fc_latent, conv_latent), -1)
        logits = self.rho(latent)
        return logits


class CustomBertModel(nn.Module):
    def __init__(self, base_model):
        super(CustomBertModel, self).__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        outputs = self.bert(**x)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_outputs(model, X, no_grad=False):

    with ch.set_grad_enabled(not no_grad):
        outputs = model(X)

    return outputs[:, 0]


def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups):
    """ Compute the key metrics """
    from aif360.metrics import ClassificationMetric
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * \
        (classified_metric_pred.true_positive_rate() +
         classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = \
        classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = \
        classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = \
        classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    metrics["False discovery rate difference"] = \
        classified_metric_pred.false_discovery_rate_difference()
    metrics["False discovery rate ratio"] = \
        classified_metric_pred.false_discovery_rate_ratio()
    metrics["False omission rate difference"] = \
        classified_metric_pred.false_omission_rate_difference()
    metrics["False omission rate ratio"] = \
        classified_metric_pred.false_omission_rate_ratio()
    metrics["False negative rate difference"] = \
        classified_metric_pred.false_negative_rate_difference()
    metrics["False negative rate ratio"] = \
        classified_metric_pred.false_negative_rate_ratio()
    metrics["False positive rate difference"] = \
        classified_metric_pred.false_positive_rate_difference()
    metrics["False positive rate ratio"] = \
        classified_metric_pred.false_positive_rate_ratio()

    return metrics


def get_z_value(metric_1, metric_2):
    assert len(metric_1) == len(metric_2), "Unequal sample sets!"
    n_samples = 2 * len(metric_1)
    m1, v1 = np.mean(metric_1), np.var(metric_1)
    m2, v2 = np.mean(metric_2), np.var(metric_2)

    mean_new = np.abs(m1 - m2)
    var_new = (v1 + v2) / n_samples

    Z = mean_new / np.sqrt(var_new)
    return Z


def get_threshold_acc(X, Y, threshold, rule=None):
    # Rule-1: everything above threshold is 1 class
    acc_1 = np.mean((X >= threshold) == Y)
    # Rule-2: everything below threshold is 1 class
    acc_2 = np.mean((X <= threshold) == Y)

    # If rule is specified, use that
    if rule == 1:
        return acc_1
    elif rule == 2:
        return acc_2

    # Otherwise, find and use the one that gives the best acc
    if acc_1 >= acc_2:
        return acc_1, 1
    return acc_2, 2


def find_threshold_acc(accs_1, accs_2, granularity=0.1):
    lower = min(np.min(accs_1), np.min(accs_2))
    upper = max(np.max(accs_1), np.max(accs_2))
    combined = np.concatenate((accs_1, accs_2))
    # Want to predict first set as 0s, second set as 1s
    classes = np.concatenate((np.zeros_like(accs_1), np.ones_like(accs_2)))
    best_acc = 0.0
    best_threshold = 0
    best_rule = None
    while lower <= upper:
        best_of_two, rule = get_threshold_acc(combined, classes, lower)
        if best_of_two > best_acc:
            best_threshold = lower
            best_acc = best_of_two
            best_rule = rule

        lower += granularity

    return best_acc, best_threshold, best_rule


def get_threshold_pred(X, Y, threshold, rule,
                       get_pred: bool = False,
                       confidence: bool = False):
    if X.shape[1] != Y.shape[0]:
        raise ValueError('Dimension mismatch between X and Y: %d and %d should match' % (X.shape[1], Y.shape[0]))
    if X.shape[0] != threshold.shape[0]:
        raise ValueError('Dimension mismatch between X and threshold: %d and %d should match' % (X.shape[0], threshold.shape[0]))
    res = []
    for i in range(X.shape[1]):
        prob = np.average((X[:, i] <= threshold) == rule)
        if confidence:
            res.append(prob)
        else:
            res.append(prob >= 0.5)
    res = np.array(res)
    if confidence:
        acc = np.mean((res >= 0.5) == Y)
    else:    
        acc = np.mean(res == Y)
    if get_pred:
        return res, acc
    return acc


def find_threshold_pred(pred_1, pred_2, granularity=0.005):
    if pred_1.shape[0] != pred_2.shape[0]:
        raise ValueError('dimension mismatch')
    thres, rules = [], []
    g = granularity
    for i in tqdm(range(pred_1.shape[0])):
        _, t, r = find_threshold_acc(pred_1[i], pred_2[i], g)
        while r is None:
            g = g/10
            _, t, r = find_threshold_acc(pred_1[i], pred_2[i], g)
        thres.append(t)
        rules.append(r-1)
    thres = np.array(thres)
    rules = np.array(rules)
    acc = get_threshold_pred(np.concatenate((pred_1, pred_2), axis=1), np.concatenate(
        (np.zeros(pred_1.shape[1]), np.ones(pred_2.shape[1]))), thres, rules)
    return acc, thres, rules

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# identity function
class basic(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # return input.clamp(min=0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# fake relu function
class fakerelu(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # return input.clamp(min=0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Fake-relu module wrapper
class FakeReluWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(FakeReluWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return fakerelu.apply(input)


# identity function module wrapper
class BasicWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(BasicWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return fakerelu.apply(input)


def get_n_effective(acc, r0, r1):
    if max(r0, r1) == 0:
        return np.inf

    if r0 == r1:
        return 0

    if acc == 1 or np.abs(r0 - r1) == 1:
        return np.inf

    num = np.log(1 - ((2 * acc - 1) ** 2))
    ratio_0 = min(r0, r1) / max(r0, r1)
    ratio_1 = (1 - max(r0, r1)) / (1 - min(r0, r1))
    den = np.log(max(ratio_0, ratio_1))
    return num / den


def bound(x, y, n):
    if max(x, y) == 0:
        return 0.5

    def bound_1():
        # Handle 0/0 form gracefully
        # if x == 0 and y == 0:
        #     return 0
        ratio = min(x, y) / max(x, y)
        return np.sqrt(1 - (ratio ** n))

    def bound_2():
        ratio = (1 - max(x, y)) / (1 - min(x, y))
        return np.sqrt(1 - (ratio ** n))

    l1 = bound_1()
    l2 = bound_2()
    pick = min(l1, l2) / 2
    return 0.5 + pick


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


def op_solution(x, y):
    """
        Return the optimal rotation to apply to x so that it aligns with y.
    """
    u, s, vh = np.linalg.svd(x.T @ y)
    optimal_x_to_y = u @ vh
    return optimal_x_to_y


def align_all_features(reference_point, features):
    """
        Perform layer-wise alignment of given features, using
        reference point. Return aligned features.
    """
    aligned_features = []
    for feature in tqdm(features, desc="Aligning features"):
        inside_feature = []
        for (ref_i, x_i) in zip(reference_point, feature):
            aligned_feature = x_i @ op_solution(x_i, ref_i)
            inside_feature.append(aligned_feature)
        aligned_features.append(inside_feature)
    return np.array(aligned_features, dtype=object)


def wrap_data_for_act_meta_clf(models_neg, models_pos,
                               data, get_activation_fn,
                               detach: bool = True):
    """
        Given models from two different distributions, get their
        activations on given data and activation-extraction function, and
        combine them into data-label format for a meta-classifier.
    """
    neg_w, neg_labels, _ = get_activation_fn(
        models_pos, data, 1, detach, verbose=False)
    pos_w, pos_labels, _ = get_activation_fn(
        models_neg, data, 0, detach, verbose=False)
    pp_x = prepare_batched_data(pos_w, verbose=False)
    np_x = prepare_batched_data(neg_w, verbose=False)
    X = [ch.cat((x, y), 0) for x, y in zip(pp_x, np_x)]
    Y = ch.cat((pos_labels, neg_labels))
    return X, Y.cuda().float()


def coordinate_descent(models_train, models_val,
                       models_test, dims, reduction_dims,
                       get_activation_fn,
                       n_samples, meta_train_args,
                       gen_optimal_fn, seed_data,
                       n_times: int = 10,
                       restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        dims: Dimensions of feature activations.
        reduction_dims: Dimensions for meta-classifier internal models.
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = ActivationMetaClassifier(
                    n_samples, dims,
                    reduction_dims=reduction_dims)
    metamodel = metamodel.cuda()

    best_clf, best_tacc = None, 0
    val_data = None
    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        X_tr, Y_tr = wrap_data_for_act_meta_clf(
            models_train[0], models_train[1], seed_data, get_activation_fn)
        X_te, Y_te = wrap_data_for_act_meta_clf(
            models_test[0], models_test[1], seed_data, get_activation_fn)
        if models_val is not None:
            val_data = wrap_data_for_act_meta_clf(
                models_val[0], models_val[1], seed_data, get_activation_fn)

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = ActivationMetaClassifier(
                n_samples, dims,
                reduction_dims=reduction_dims)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        metamodel.train()
        clf, tacc = train_meta_model(
                    metamodel,
                    (X_tr, Y_tr), (X_te, Y_te),
                    epochs=meta_train_args['epochs'],
                    binary=True, lr=1e-3,
                    regression=False,
                    batch_size=meta_train_args['batch_size'],
                    val_data=val_data, combined=True,
                    eval_every=10, gpu=True)
        all_accs.append(tacc)

        # Keep track of best model and latest model
        if tacc > best_tacc:
            best_tacc = tacc
            best_clf = clf

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data, get_activation_fn)

    # Return best and latest models
    return (best_tacc, best_clf), (tacc, clf), all_accs


def check_if_inside_cluster():
    """
        Check if current code is being run inside a cluster.
    """
    if environ.get('ISRIVANNA') == "1":
        return True
    return False


class AffinityMetaClassifier(nn.Module):
    def __init__(self, num_dim: int, numlayers: int,
                 num_final: int = 16, only_latent: bool = False):
        super(AffinityMetaClassifier, self).__init__()
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


def make_affinity_feature(model, data, use_logit=False, detach=True, verbose=True):
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
        # Old (before 2/4)
        # Skip logits if asked not to use (default)
        # if not use_logit and i == (len(model_features) - 1):
            # break
        scores = []
        # Pair-wise iteration of all data
        for i in range(len(data)-1):
            others = feature[i+1:]
            scores += cos(ch.unsqueeze(feature[i], 0), others)
        layerwise_features.append(ch.stack(scores, 0))

    # New (2/4)
    # If asked to use logits, convert them to probability scores
    # And then consider them as-it-is (instead of pair-wise comparison)
    if use_logit:
        logits = model_features[-1]
        probs = ch.sigmoid(logits)
        layerwise_features.append(probs)

    concatenated_features = ch.stack(layerwise_features, 0)
    return concatenated_features


def make_affinity_features(models, data, use_logit=False, detach=True, verbose=True):
    all_features = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Building affinity matrix")
    for model in iterator:
        all_features.append(
            make_affinity_feature(
                model, data, use_logit=use_logit, detach=detach, verbose=verbose)
        )
    return ch.stack(all_features, 0)


def coordinate_descent_new(models_train, models_val,
                           num_features, num_layers,
                           get_features,
                           meta_train_args,
                           gen_optimal_fn, seed_data,
                           n_times: int = 10,
                           restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        num_layers: Number of layers of models used for activations
        get_features: Function that takes (models, data) as input and returns features
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = AffinityMetaClassifier(num_features, num_layers)
    metamodel = metamodel.cuda()

    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        train_loader = get_features(
            models_train[0], models_train[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])
        val_loader = get_features(
            models_val[0], models_val[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = AffinityMetaClassifier(num_features, num_layers)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        _, val_acc = train(metamodel, (train_loader, val_loader),
                           epoch_num=meta_train_args['epochs'],
                           expect_extra=False,
                           verbose=False)
        all_accs.append(val_acc)

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data)

    # Return all accuracies
    return all_accs


class WeightAndActMeta(nn.Module):
    """
        Combined meta-classifier that uses model weights as well as activation
        trends for property prediction.
    """
    def __init__(self, dims: List[int], num_dims: int, num_layers: int):
        super(WeightAndActMeta, self).__init__()
        self.dims = dims
        self.num_dims = num_dims
        self.num_layers = num_layers
        self.act_clf = AffinityMetaClassifier(
            num_dims, num_layers, only_latent=True)
        self.weights_clf = PermInvModel(dims, only_latent=True)
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


def _perpoint_threshold_on_ratio(preds_1, preds_2, classes, threshold, rule):
    """
        Run perpoint threshold test (confidence)
        for a given "quartile" ratio
    """
    # Combine predictions into one vector
    combined = np.concatenate((preds_1, preds_2), axis=1)

    # Compute accuracy for given predictions, thresholds, and rules
    preds, acc = get_threshold_pred(
        combined, classes, threshold, rule, get_pred=True,
        confidence=True)

    return 100 * acc, preds
