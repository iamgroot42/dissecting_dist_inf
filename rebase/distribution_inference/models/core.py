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

        # expected input shape: 218,178
        if fake_relu:
            act_fn = BasicWrapper
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus

        super().__init__(is_conv=True)
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

        clf_layers = [
            nn.Linear(64 * 6 * 6, 64),
            FakeReluWrapper(inplace=True),
            nn.Linear(64, 32),
            FakeReluWrapper(inplace=True),
            nn.Linear(32, num_classes),
        ]

        mapping = {0: 1, 1: 4, 2: 7, 3: 9, 4: 11, 5: 1, 6: 3}
        if self.latent_focus is not None:
            if self.latent_focus < 5:
                layers[mapping[self.latent_focus]] = act_fn(inplace=True)
            else:
                clf_layers[mapping[self.latent_focus]] = act_fn(inplace=True)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:

        if latent is None:
            x = self.features(x)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            x = self.classifier(x)
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

    def forward(self, x):
        x = self.layers(x)
        return x
