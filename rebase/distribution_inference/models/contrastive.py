
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List
from torch.nn import Parameter

from distribution_inference.models.core import BaseModel


class _ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int,
                out_features: int,
                s: float = 30.0,
                m: float = 0.50,
                easy_margin: bool = False):
        super(_ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(ch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None, train: bool = False,):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ch.sqrt((1.0 - ch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = ch.where(cosine > 0, phi, cosine)
        else:
            phi = ch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        if train:
            one_hot = ch.zeros(cosine.size(), device=input.get_device())
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = ch.where(one_hot == 1, phi, cosine)
        else:
            output = cosine
        output *= self.s

        return output


def _conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = _conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = _conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = _SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class _SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(_SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block,
                layers: List[int],
                n_out: int = 512,
                use_se: bool = True):
        self.inplanes = 64
        self.n_out = n_out
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, self.n_out)
        self.bn5 = nn.BatchNorm1d(self.n_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                      downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return x


class GenericArcFace(BaseModel):
    def __init__(self,
                 model: nn.Module, 
                 n_out: int = 512,
                 n_people: int = None):
        super().__init__(is_conv=True, is_contrastive_model=True)
        self.model = model
        self.n_people = n_people
        if self.n_people is not None:
            self.metric_fc = _ArcMarginProduct(
                n_out, self.n_people, s=30, m=0.5)

    def forward(self, x: ch.Tensor,
                only_embedding: bool=False,
                latent: int = None,
                get_both: bool = False) -> ch.Tensor:
        if type(x) is tuple:
            if self.n_people is None:
                raise ValueError(
                    "n_people must be provided when loading model for generating predictions")
            # Class label also provided for contrastive learning
            x, y = x
        else:
            if self.training:
                raise ValueError("Training mode requires class labels")
            y = None
        
        feature = self.model(x)
        if only_embedding:
            return feature

        output = self.metric_fc(feature, label=y, train=self.training)
        if get_both:
            return feature, output

        return output


class ArcFaceResnet(GenericArcFace):
    def __init__(self,
                 n_out: int = 512,
                 use_se: bool = True,
                 n_people: int = None):
        model = ResNetFace(
            _IRBlock, [2, 2, 2, 2], n_out=n_out, use_se=use_se)
        super().__init__(model, n_out=n_out, n_people=n_people)


class RelationNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, multiplier: int = 5):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size*2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.multiplier = multiplier
        # The 5x5 below comes from the Conv, not n-way or k-way
        # Adjust according to input image size:
        # (32: 64*1*1) (84: 64*3*3) (96: 64*5*5) (112: 64*6*6) (160: 64*9*9) (224: 64*13*13)
        if multiplier > 1:
            self.fc1 = nn.Linear(input_size*multiplier*multiplier, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)
        else:
            self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        if self.multiplier > 1:
            out = F.relu(self.fc1(out))
            out = ch.sigmoid(self.fc2(out))
        else:
            out = ch.sigmoid(self.fc1(out))
        return out


class SCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=0),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2)),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2)),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU())
        )
        self.num_layers = 4
    
    def forward(self, x, latent: int = None):
        if latent is not None:
            if latent < 0 or latent > self.num_layers - 1:
                raise ValueError("Invalid layer index")
        out = x
        j = 0
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == latent:
                return out

        return out


class SCNNDeeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=0),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2)),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2)),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2)),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2))
        )
    
    def forward(self, x, latent: int = None):
        if latent is not None:
            if latent < 0 or latent > 3:
                raise ValueError("Invalid layer index")
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == latent:
                return out

        return out


class GenericFaceAudit(BaseModel):
    def __init__(self,
                 model: nn.Module,
                 hidden_size: int,
                 feat_dim: int,
                 multiplier: int = 5):
        super().__init__(is_conv=True, is_contrastive_model=True, is_relation_based=True)
        self.features = model
        self.feat_dim = feat_dim
        self.classifier = None
        self.relation_network = RelationNetwork(
            input_size=self.feat_dim, hidden_size=hidden_size,\
                multiplier=multiplier)

    def forward(self, x: ch.Tensor,
                embedding_mode: bool,
                latent: int = None) -> ch.Tensor:
        """
            If embedding_mode is True, use self.features
            Else, use self.relation_network
        """        
        if embedding_mode:
            feature = self.features(x, latent=latent)
            return feature
        else:
            out = self.relation_network(x)
            return out


class SCNNFaceAudit(GenericFaceAudit):
    def __init__(self,
                 feat_dim: int = 64):
        model = SCNN()
        super().__init__(model, feat_dim = feat_dim, hidden_size=100)


class SCNNDeeperFaceAudit(GenericFaceAudit):
    def __init__(self,
                 feat_dim: int = 64):
        model = SCNNDeeper()
        super().__init__(model, feat_dim = feat_dim, hidden_size=100, multiplier=1)
