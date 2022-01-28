import numpy as np
from tqdm import tqdm
import torch as ch
import torch.nn as nn
import os
from joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS


# BASE_MODELS_DIR = '/u/pdz6an/git/census/50_50_new'
BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_census/50_50_new"


class PortedMLPClassifier(nn.Module):
    def __init__(self):
        super(PortedMLPClassifier, self).__init__()
        layers = [
            nn.Linear(in_features=42, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: ch.Tensor, latent: int = None):
        if latent is None:
            return self.layers(x)

        if latent not in [0, 1, 2]:
            raise ValueError("Invald interal layer requested")

        # First three hidden layers correspond to outputs of
        # Model layers 1, 3, 5
        latent = (latent * 2) + 1

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == latent:
                return x


def port_mlp_to_ch(clf):
    """
        Extract weights from MLPClassifier and port
        to PyTorch model.
    """
    nn_model = PortedMLPClassifier()
    i = 0
    for (w, b) in zip(clf.coefs_, clf.intercepts_):
        w = ch.from_numpy(w.T).float()
        b = ch.from_numpy(b).float()
        nn_model.layers[i].weight = nn.Parameter(w)
        nn_model.layers[i].bias = nn.Parameter(b)
        i += 2  # Account for ReLU as well

    nn_model = nn_model.cuda()
    return nn_model


def layer_output(data, MLP, layer=0, get_all=False):
    """
        For a given model and some data, get output for each layer's activations < layer.
        If get_all is True, return all activations unconditionally.
    """
    L = data.copy()
    all = []
    for i in range(layer):
        L = ACTIVATIONS['relu'](
            np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
        if get_all:
            all.append(L)
    if get_all:
        return all
    return L


# Load models from directory, return feature representations
def get_model_representations(folder_path, label, first_n=np.inf, start_n=0):
    models_in_folder = os.listdir(folder_path)
    # np.random.shuffle(models_in_folder)
    w, labels = [], []
    for path in tqdm(models_in_folder):
        clf = load_model(os.path.join(folder_path, path))

        # Extract model parameters
        weights = [ch.from_numpy(x) for x in clf.coefs_]
        dims = [w.shape[0] for w in weights]
        biases = [ch.from_numpy(x) for x in clf.intercepts_]
        processed = [ch.cat((w, ch.unsqueeze(b, 0)), 0).float().T
                     for (w, b) in zip(weights, biases)]

        # Use parameters only from first N layers
        # and starting from start_n
        if first_n != np.inf:
            processed = processed[start_n:first_n]
            dims = dims[start_n:first_n]

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    w = np.array(w, dtype=object)
    labels = ch.from_numpy(labels)

    return w, labels, dims


def get_model(max_iter=40,
              hidden_layer_sizes=(32, 16, 8),):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter)
    return clf


def get_models(folder_path, n_models=1000, shuffle=True):
    paths = os.listdir(folder_path)
    if shuffle:
        paths = np.random.permutation(paths)
    paths = paths[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, property, split)
    return os.path.join(BASE_MODELS_DIR,  property, split, value)
