import torch as ch
import torch.nn as nn
from torchvision.models import densenet121

from distribution_inference.models.utils import BasicWrapper, FakeReluWrapper


class BaseModel(nn.Module):
    def __init__(self,
                 is_conv: bool = False,
                 transpose_features: bool = True):
        self.is_conv = is_conv
        self.transpose_features = transpose_features
        super(BaseModel, self).__init__()


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
            FakeReluWrapper(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            FakeReluWrapper(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            FakeReluWrapper(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            FakeReluWrapper(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            FakeReluWrapper(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        # Don't need EACH layer (too much memory) for now,
        # can get away with skipping most layers
        # self.valid_for_all_conv = [2, 5, 7, 9, 12]
        self.valid_for_all_conv = [5, 9]

        clf_layers = [
            nn.Linear(64 * 6 * 6, 64),
            FakeReluWrapper(inplace=True),
            nn.Linear(64, 32),
            FakeReluWrapper(inplace=True),
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
                get_all: bool = False) -> ch.Tensor:

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

            _collect_latents(self.features, self.valid_for_all_conv)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            _collect_latents(self.classifier, self.valid_for_all_fc)

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
    def __init__(self, n_inp: int, num_classes: int = 1):
        super().__init__(is_conv=False)
        self.layers = nn.Sequential(
            nn.Linear(n_inp, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
        self.valid_for_all = [1, 3, 4]

    def forward(self, x,
                detach_before_return: bool = False,
                get_all: bool = False):
        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if get_all and i in self.valid_for_all:
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
        self.valid_for_all = [1, 3, 4]

    def forward(self, x: ch.Tensor,
                detach_before_return: bool = False,
                get_all: bool = False) -> ch.Tensor:
        all_latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if get_all and i in self.valid_for_all:
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
