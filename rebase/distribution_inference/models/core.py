import torch as ch
import torch.nn as nn
import numpy as np
from typing import List, Union
from torchvision.models import densenet121
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from distribution_inference.models.utils import BasicWrapper, FakeReluWrapper


class BaseModel(nn.Module):
    def __init__(self,
                 is_conv: bool = False,
                 transpose_features: bool = True,
                 is_sklearn_model: bool = False,
                 is_graph_model: bool = False):
        super().__init__()
        self.is_conv = is_conv
        self.transpose_features = transpose_features
        self.is_sklearn_model = is_sklearn_model
        self.is_graph_model = is_graph_model
    
    def forward(self, x: Union[np.ndarray, ch.Tensor]) -> Union[np.ndarray, ch.Tensor]:
        converted = False
        # Convert from tensor to numpy
        if type(x) == ch.Tensor:
            device = x.device
            x = x.detach().cpu().numpy()
            converted = True
        # Get predictions
        if self.is_sklearn_model:
            preds = self.model.predict_proba(x)
            # If binary, show same behavior as PyTorch models
            if preds.shape[1] == 2:
                preds = preds[:, [1]]
        else:
            preds = self.model(x)
        # Return predictions
        if converted:
            # Convert to tensor
            preds = ch.from_numpy(preds).float()
            preds = preds.to(device)
        return preds

    def fit(self, x, y):
        self.model.fit(x, y)
        return self.acc(x, y)

    def score(self, x, y):
        return log_loss(y, self.forward(x))

    def acc(self, x, y):
        return self.model.score(x, y)


class SVMClassifier(BaseModel):
    def __init__(self,
                 C=1.0,
                 kernel='rbf',
                 degree=3):
        super().__init__(is_sklearn_model=True)
        self.model = SVC(probability=True,
                         C=C,
                         kernel=kernel,
                         degree=degree)


class RandomForest(BaseModel):
    def __init__(self,
                 max_depth: int = None,
                 n_estimators: int = 100,
                 n_jobs: int = -1,
                 min_samples_leaf: int = 1):
        super().__init__(is_sklearn_model=True)
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            min_samples_leaf=min_samples_leaf)


class LRClassifier(BaseModel):
    def __init__(self,
                 C: float = 1.0,
                 penalty: str = 'l2',
                 solver: str = 'lbfgs',
                 multi_class: str = 'multinomial'):
        super().__init__(is_sklearn_model=True)
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            multi_class=multi_class)


class KNeighborsClassifier(BaseModel):
    def __init__(self,
                n_neighbors: int = 5,
                leaf_size: int=30,
                p :int=2,
                n_jobs: int = -1):
        super().__init__(is_sklearn_model=True)
        self.model = KN(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs)


class GaussianProcessClassifier(BaseModel):
    def __init__(self,
                n_restarts_optimizer: int = 0,
                max_iter_predict: int = 100,
                n_jobs:int = -1):
        super().__init__(is_sklearn_model=True)
        self.model = GPC(
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            n_jobs=n_jobs)

class KNeighborsClassifier(BaseModel):
    def __init__(self,
                n_neighbors: int = 5,
                leaf_size:int=30,
                p:int=2,
                n_jobs:int = -1
                ):
        super().__init__(is_sklearn_model=True)
        self.model = KN(n_neighbors=n_neighbors,leaf_size=leaf_size,p=p,n_jobs=n_jobs)


class GaussianProcessClassifier(BaseModel):
    def __init__(self,
                n_restarts_optimizer: int = 0,
                max_iter_predict:int=100,
                n_jobs:int = -1
                ):
        super().__init__(is_sklearn_model=True)
        self.model = GPC(n_restarts_optimizer=n_restarts_optimizer,
        max_iter_predict=max_iter_predict,n_jobs=n_jobs)


class MultinomialNB(BaseModel):
    def __init__(self,
                alpha:float = 1.0
                ):
        super().__init__(is_sklearn_model=True)
        self.model = MNB(alpha=alpha)


class InceptionModel(BaseModel):
    def __init__(self,
                 num_classes: int = 1,
                 fake_relu: bool = False,
                 latent_focus: int = None) -> None:
        super().__init__(is_conv=True)
        self.model = densenet121(num_classes=num_classes) #, aux_logits=False)

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        return self.model(x)


class MyAlexNet(BaseModel):
    def __init__(self,
                 num_classes: int = 1,
                 fake_relu: bool = False,
                 latent_focus: int = None) -> None:
        super().__init__(is_conv=True)
        # expected input shape: 218,178
        if fake_relu:
            act_fn = BasicWrapper
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus

        layers = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        # Don't need EACH layer (too much memory) for now,
        # can get away with skipping most layers
        # self.valid_for_all_conv = [2, 5, 7, 9, 12]
        self.valid_for_all_conv = [5, 9]

        clf_layers = [
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        ]
        self.valid_for_all_fc = [1, 3, 4]

        mapping = {0: 1, 1: 4, 2: 7, 3: 9, 4: 11, 5: 1, 6: 3}
        if self.latent_focus is not None:
            if self.latent_focus < 5:
                layers[mapping[self.latent_focus]] = act_fn(inplace=True)
            else:
                clf_layers[mapping[self.latent_focus]] = act_fn(inplace=True)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x: ch.Tensor,
                latent: int = None,
                detach_before_return: bool = False,
                get_all: bool = False,
                layers_to_target_conv: List[int] = None,
                layers_to_target_fc: List[int] = None,) -> ch.Tensor:

        # Override list of layers if given
        valid_conv = layers_to_target_conv if layers_to_target_conv else self.valid_for_all_conv
        valid_fc = layers_to_target_fc if layers_to_target_fc else self.valid_for_all_fc

        if latent is None:
            all_latents = []

            # Function to collect latents (for given layers) from given model
            def _collect_latents(model, wanted):
                nonlocal x
                for i, layer in enumerate(model):
                    x = layer(x)
                    if get_all and i in wanted:
                        x_flat = x.view(x.size(0), -1)
                        if detach_before_return:
                            all_latents.append(x_flat.detach())
                        else:
                            all_latents.append(x_flat)

            _collect_latents(self.features, valid_conv)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            _collect_latents(self.classifier, valid_fc)

            if get_all:
                return all_latents
            if detach_before_return:
                return x.detach()
            return x

        if latent not in list(range(7)):
            raise ValueError("Invald interal layer requested")

        # Pick activations just before previous layers
        # Since any intermediate functions that pool activations
        # Introduce invariance to further layers, so should be
        # Clubbed according to pooling

        if latent < 4:
            # Latent from Conv part of model
            mapping = {0: 2, 1: 5, 2: 7, 3: 9}
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == mapping[latent]:
                    return x.view(x.shape[0], -1)

        elif latent == 4:
            x = self.features(x)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            return x
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            for i, layer in enumerate(self.classifier):
                x = layer(x)
                if i == 2 * (latent - 5) + 1:
                    return x


class MLPTwoLayer(BaseModel):
    def __init__(self, n_inp: int, num_classes: int = 1, dims: List[int] = [64, 16]):
        super().__init__(is_conv=False)
        self.layers = nn.Sequential(
            nn.Linear(n_inp, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], num_classes),
        )
        self.valid_for_all_fc = [1, 3, 4]

    def forward(self, x,
                detach_before_return: bool = False,
                get_all: bool = False,
                layers_to_target_conv: List[int] = None,
                layers_to_target_fc: List[int] = None,
                latent:int=None):

        # Override list of layers if given
        valid_fc = layers_to_target_fc if layers_to_target_fc else self.valid_for_all_fc
        latent_mapping = {0: 1, 1:3}
        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if latent is not None and i == latent_mapping[latent]:
                if detach_before_return:
                    return x.detach()
                else:
                    return x
            if get_all and i in valid_fc:
                if detach_before_return:
                    all_latents.append(x.detach())
                else:
                    all_latents.append(x)

        if get_all:
            return all_latents
        if detach_before_return:
            return x.detach()
        return x


class MLPThreeLayer(BaseModel):
    def __init__(self, n_inp: int, num_classes: int = 1, dims: List[int] = [64, 32, 16]):
        super().__init__(is_conv=False)
        self.layers = nn.Sequential(
            nn.Linear(n_inp, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], num_classes),
        )
        self.valid_for_all_fc = [1, 3, 5, 6]

    def forward(self, x,
                detach_before_return: bool = False,
                get_all: bool = False,
                layers_to_target_conv: List[int] = None,
                layers_to_target_fc: List[int] = None,
                latent:int=None):

        # Override list of layers if given
        valid_fc = layers_to_target_fc if layers_to_target_fc else self.valid_for_all_fc
        latent_mapping = {0: 1, 1:3, 2:5}
        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if latent is not None and i == latent_mapping[latent]:
                if detach_before_return:
                    return x.detach()
                else:
                    return x
            if get_all and i in valid_fc:
                if detach_before_return:
                    all_latents.append(x.detach())
                else:
                    all_latents.append(x)

        if get_all:
            return all_latents
        if detach_before_return:
            return x.detach()
        return x


class BoneModel(BaseModel):
    def __init__(self,
                 n_inp: int = 1024,
                 fake_relu: bool = False,
                 latent_focus: int = None):
        super().__init__(is_conv=False)
        # if latent_focus is not None:
        #     if latent_focus not in [0, 1]:
        #         raise ValueError("Invalid interal layer requested")

        # if fake_relu:
        #     act_fn = BasicWrapper
        # else:
        #     act_fn = nn.ReLU

        self.layers = nn.Sequential(
            nn.Linear(n_inp, 128),
            FakeReluWrapper(inplace=True),
            nn.Linear(128, 64),
            FakeReluWrapper(inplace=True),
            nn.Linear(64, 1)
        )
        self.valid_for_all_fc = [1, 3, 4]

    def forward(self, x: ch.Tensor,
                detach_before_return: bool = False,
                get_all: bool = False,
                latent: int = None,
                layers_to_target_conv: List[int] = None,
                layers_to_target_fc: List[int] = None,) -> ch.Tensor:

        # Override list of layers if given
        valid_fc = layers_to_target_fc if layers_to_target_fc else self.valid_for_all_fc

        latent_mapping = {0: 1, 1:3} # layer wanted: actual layer in model

        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Get specific latent output
            if latent is not None and i == latent_mapping[latent]:
                if detach_before_return:
                    return x.detach()
                else:
                    return x

            if get_all and i in valid_fc:
                if detach_before_return:
                    all_latents.append(x.detach())
                else:
                    all_latents.append(x)

        if get_all:
            return all_latents
        if detach_before_return:
            return x.detach()
        return x


class MLPFourLayer(BaseModel):
    def __init__(self, n_inp: int, num_classes: int = 1):
        super().__init__(is_conv=False)
        self.layers = nn.Sequential(
            nn.Linear(n_inp, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
        self.valid_for_all_fc = [1, 3, 4]

    def forward(self, x,
                detach_before_return: bool = False,
                get_all: bool = False,
                layers_to_target_conv: List[int] = None,
                layers_to_target_fc: List[int] = None,):

        # Override list of layers if given
        valid_fc = layers_to_target_fc if layers_to_target_fc else self.valid_for_all_fc

        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if get_all and i in valid_fc:
                if detach_before_return:
                    all_latents.append(x.detach())
                else:
                    all_latents.append(x)

        if get_all:
            return all_latents
        if detach_before_return:
            return x.detach()
        return x


class MLPFiveLayer(BaseModel):
    def __init__(self, n_inp: int, num_classes: int = 1, dims: List[int] = [1024, 512, 128, 64]):
        super().__init__(is_conv=False)
        self.layers = nn.Sequential(
            nn.Linear(n_inp, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
            nn.Linear(dims[3], num_classes),
        )
        self.valid_for_all_fc = [1, 3, 5,7,9]

    def forward(self, x,
                detach_before_return: bool = False,
                get_all: bool = False,
                layers_to_target_conv: List[int] = None,
                layers_to_target_fc: List[int] = None,):

        # Override list of layers if given
        valid_fc = layers_to_target_fc if layers_to_target_fc else self.valid_for_all_fc

        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if get_all and i in valid_fc:
                if detach_before_return:
                    all_latents.append(x.detach())
                else:
                    all_latents.append(x)

        if get_all:
            return all_latents
        if detach_before_return:
            return x.detach()
        return x


class DenseNet(BaseModel):
    def __init__(self,
                 n_inp: int = 1024,
                 fake_relu: bool = False,
                 latent_focus: int = None):
        # TODO: Implement latent focus
        # TODO: Implement fake_relu
        super().__init__(is_conv=True)

        # Densenet
        self.model = densenet121(pretrained=True)
        self.model.classifier = nn.Linear(n_inp, 1)

        # TODO: Implement fake_relu

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        # TODO: Implement latent functionality
        return self.model(x)


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

    def forward(self, x: ch.Tensor,
                latent: int = None,
                get_all: bool = False,
                detach_before_return: bool = True,
                on_cpu: bool = False):
        """
        Args:
            x: Input tensor of shape (batch_size, 42)
            latent: If not None, return only the latent representation. Else, get requested latent layer's output
            get_all: If True, return all activations
            detach_before_return: If True, detach the latent representation before returning it
            on_cpu: If True, return the latent representation on CPU
        """
        if latent is None and not get_all:
            return self.layers(x)

        if latent not in [0, 1, 2] and not get_all:
            raise ValueError("Invald interal layer requested")

        if latent is not None:
            # First three hidden layers correspond to outputs of
            # Model layers 1, 3, 5
            latent = (latent * 2) + 1
        valid_for_all = [1, 3, 5, 6]

        latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Append activations for all layers (post-activation only)
            if get_all and i in valid_for_all:
                if detach_before_return:
                    if on_cpu:
                        latents.append(x.detach().cpu())
                    else:
                        latents.append(x.detach())
                else:
                    if on_cpu:
                        latents.append(x.cpu())
                    else:
                        latents.append(x)
            if i == latent:
                if on_cpu:
                    return x.cpu()
                else:
                    return x

        return latents


class GCN(BaseModel):
    def __init__(self, n_hidden, n_layers,
                 dropout,  num_classes,
                 num_features):
        super().__init__(is_conv=True,
                         is_graph_model=True,
                         transpose_features=False)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(num_features, n_hidden, activation=F.relu))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=F.relu))

        # output layer
        self.layers.append(GraphConv(n_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features, latent=None):
        if latent is not None:
            if latent < 0 or latent > len(self.layers):
                raise ValueError("Invald interal layer requested")

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
            if i == latent:
                return h
        return h


class AnyLayerMLP(BaseModel):
    def __init__(self, n_inp: int, n_classes: int, depth: int = 1):
        super().__init__(is_conv=False)
        self.mapping = {
            1: [16, 8, 4],
        }
        if depth not in self.mapping:
            raise ValueError(f"Depth {depth} not supported")
        desired_dims = self.mapping[depth]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_inp, desired_dims[0]))
        for i in range(1, len(desired_dims)):
            self.layers.append(nn.Linear(desired_dims[i - 1], desired_dims[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(desired_dims[-1], n_classes))
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x: ch.Tensor) -> ch.Tensor:
        return self.model(x)