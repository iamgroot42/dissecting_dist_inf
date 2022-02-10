import torch as ch
import torch.nn as nn
from torchvision import models
import numpy as np
from tqdm import tqdm
import torch.nn.utils.prune as prune
from torchvision.models import densenet121
from utils import get_weight_layers, ensure_dir_exists, BasicWrapper, FakeReluWrapper
import os
import argparse
import os
import sys
import torch
import logging

# from torch.optim.lr_scheduler import StepLR, MultiStepLR

import utils
from data_utils import BoneWrapper, get_df, get_features
from model_utils import load_model, BASE_MODELS_DIR

from nni.compression.pytorch.utils.counter import count_flops_params

import nni


from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    FPGMPruner,
    TaylorFOWeightFilterPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)


logging.getLogger("nni.algorithms.compression.pytorch.pruning.iterative_pruner").setLevel(
    logging.WARNING)  # Supress info
logging.getLogger("nni.compression.pytorch.compressor").setLevel(
    logging.WARNING)  # Supress info

BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_boneage/"


class BoneModel(nn.Module):
    def __init__(self,
                 n_inp: int,
                 fake_relu: bool = False,
                 latent_focus: int = None):
        if latent_focus is not None:
            if latent_focus not in [0, 1]:
                raise ValueError("Invalid interal layer requested")

        if fake_relu:
            act_fn = BasicWrapper
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus
        super(BoneModel, self).__init__()
        layers = [
            nn.Linear(n_inp, 128),
            FakeReluWrapper(inplace=True),
            nn.Linear(128, 64),
            FakeReluWrapper(inplace=True),
            nn.Linear(64, 1)
        ]

        mapping = {0: 1, 1: 3}
        if self.latent_focus is not None:
            layers[mapping[self.latent_focus]] = act_fn(inplace=True)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        if latent is None:
            return self.layers(x)

        if latent not in [0, 1]:
            raise ValueError("Invald interal layer requested")

        # First, second hidden layers correspond to outputs of
        # Model layers 1, 3
        latent = (latent * 2) + 1

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == latent:
                return x


class BoneFullModel(nn.Module):
    def __init__(self,
                 fake_relu: bool = False,
                 latent_focus: int = None):
        # TODO: Implement latent focus
        # TODO: Implement fake_relu
        super(BoneFullModel, self).__init__()

        # Densenet
        self.model = densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, 1)

        # TODO: Implement fake_relu

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        # TODO: Implement latent functionality
        return self.model(x)


# Save model in specified directory
def save_model(model, split, prop_and_name, full_model=False):
    if full_model:
        subfolder_prefix = os.path.join(split, "full")
    else:
        subfolder_prefix = split

    # Make sure directory exists
    ensure_dir_exists(os.path.join(BASE_MODELS_DIR, subfolder_prefix))

    ch.save(model.state_dict(), os.path.join(
        BASE_MODELS_DIR, subfolder_prefix, prop_and_name))


# Load model from given directory
def load_model(path: str, fake_relu: bool = False,
               latent_focus: int = None, cpu: bool = False):
    model = BoneModel(1024, fake_relu=fake_relu, latent_focus=latent_focus)
    if cpu:
        model.load_state_dict(ch.load(path))
    else:
        model.load_state_dict(ch.load(path))

    model.eval()
    return model


# Get model path, given perameters
def get_model_folder_path(split, ratio):
    return os.path.join(BASE_MODELS_DIR, split, ratio)

#For pruning


def get_data(dataset, data_dir, batch_size, test_batch_size, ratio):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'boneage':
        # Ready data
        def filter(x): return x["gender"] == 1

        df_train, df_val = get_df("adv")
        features = get_features("adv")

        # Get data with ratio
        df_1 = utils.heuristic(
            df_train, filter, float(ratio), 200,
            class_imbalance=1.0, n_tries=300)

        df_2 = utils.heuristic(
            df_val, filter, float(ratio), 700,
            class_imbalance=1.0, n_tries=300)

        ds_1 = BoneWrapper(df_1, df_2, features=features)

        train_loader, test_loader = ds_1.get_loaders(batch_size, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()

    return train_loader, test_loader, criterion

#For pruning


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()
            target = target.unsqueeze(1)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = (output > 0).float()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    #print('Test Loss: {}  Accuracy: {}%\n'.format(
    #    test_loss, acc))
    return acc

#For pruning


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target = target.float()
        target = target.unsqueeze(1)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % 100 == 0: #Set log-interval argument to 100
        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #    epoch, batch_idx * len(data), len(train_loader.dataset),
        #    100. * batch_idx / len(train_loader), loss.item()))


# Function to extract model weights for all models in given directory
def get_model_features(model_dir, sparsity, fine_tune_epochs, ratio, max_read=None, first_n=np.inf, start_n=0, prune_ratio=None):
    experiment_data_dir = './experiment_data'
    dataset = 'boneage'
    batch_size = 128
    test_batch_size = 200
    pruner_str = 'agp'
    data_dir = './data/'

    #For pruning
    str2pruner = {
        'level': LevelPruner,  # Linear support
        'l1filter': L1FilterPruner,  # Conv-layers only
        'l2filter': L2FilterPruner,  # Conv-layers only
        'slim': SlimPruner,  # Conv-layers only
        'agp': AGPPruner,  # Linear support only with level
        'fpgm': FPGMPruner,  # Conv-layers only
        'mean_activation': ActivationMeanRankFilterPruner,  # Conv-layers only
        'apoz': ActivationAPoZRankFilterPruner,  # Conv-layers only
        'taylorfo': TaylorFOWeightFilterPruner  # Conv-layers only

        # TODO: Look into SimulatedAnnealing, LotteryTicketPruner, AutoCompress
    }

    vecs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(experiment_data_dir, exist_ok=True)

    # prepare model and data
    train_loader, test_loader, criterion = get_data(
        dataset, data_dir, batch_size, test_batch_size, ratio=ratio)

    iterator = os.listdir(model_dir)[:1]
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        model = load_model(os.path.join(model_dir, mpath))

        ############################################################
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # Shift model to device
        model.to(device)

        # Get dummy input (for computing FLOPS etc)
        dummy_input = torch.randn([test_batch_size, 1024]).to(device)
        flops, params, _ = count_flops_params(model, dummy_input)
        #print(f"FLOPs: {flops}, params: {params}")

        #print(f'start {pruner_str} pruning...')

        def trainer(model, optimizer, criterion, epoch):
            return train(model, device, train_loader, criterion, optimizer, epoch=epoch)

        pruner_cls = str2pruner[pruner_str]

        kw_args = {}
        config_list = [{
            'sparsity': sparsity,
            'op_types': ['Linear']
        }]

        if pruner_str == 'level':
            config_list = [{
                'sparsity': sparsity,
                'op_types': ['default']
            }]

        else:
            if pruner_str not in ('l1filter', 'l2filter', 'fpgm'):
                # set only work for training aware pruners
                kw_args['trainer'] = trainer
                kw_args['optimizer'] = optimizer
                kw_args['criterion'] = criterion

            if pruner_str in ('mean_activation', 'apoz', 'taylorfo'):
                kw_args['sparsifying_training_batches'] = 1

            if pruner_str == 'slim':
                kw_args['sparsifying_training_epochs'] = 1

            if pruner_str == 'agp':
                kw_args['pruning_algorithm'] = 'level'
                kw_args['num_iterations'] = 10
                kw_args['epochs_per_iteration'] = 1

        pruner = pruner_cls(model, config_list, **kw_args)

        # Pruner.compress() returns the masked model
        model = pruner.compress()
        pruner.get_pruned_weights()

        #print("Pruning complete")

        if True:  # Replaced args.test_only with True
            test(model, device, criterion, test_loader)

        #print('start finetuning...')

        # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        all_accs = []
        true_accs = []
        best_top1 = 0
        save_path = os.path.join(experiment_data_dir, f'finetuned.pth')
        for epoch in range(fine_tune_epochs):
            #print('# Epoch {} #'.format(epoch))
            train(model, device, train_loader, criterion, optimizer, epoch)
            #torch.set_printoptions(profile="full")
            top1 = test(model, device, criterion, test_loader)
            all_accs.append(top1)
            if top1 > best_top1:
                best_top1 = top1
                torch.save(model.state_dict(), save_path)

        flops, params, results = count_flops_params(model, dummy_input)
        #print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_top1: .2f}')
        print(all_accs)
        #for name, para in model.named_parameters():
        #    print('{}: {}'.format(name, para.shape))
        #    print('Sum: {}'.format(torch.sum(torch.square(para))))
        #print(model.state_dict())

        '''
        prune_mask = []
        # Prune weight layers, if requested
        if prune_ratio is not None:
            for layer in model.layers:
                if type(layer) == nn.Linear:
                    # Keep track of weight pruning mask
                    prune.l1_unstructured(
                        layer, name='weight', amount=prune_ratio)
                    prune_mask.append(layer.weight_mask.data.detach().cpu())'''

        # Get model params, shift to GPU
        dims, fvec = get_weight_layers(
            model, first_n=first_n, start_n=start_n)
        fvec = [x.cuda() for x in fvec]

        vecs.append(fvec)

    return dims, vecs


def get_pre_processor():
    # Load model
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Get rid of existing classification layer
    # Extract only features
    model.classifier = nn.Identity()
    return model


# Check with this model number exists
def check_if_exists(model_id, split, full_model=False):
    if full_model:
        model_check_path = os.path.join(
            BASE_MODELS_DIR, split, "full")
    else:
        model_check_path = os.path.join(BASE_MODELS_DIR, split)
    for model_name in os.listdir(model_check_path):
        if ("%d_" % model_id) in model_name:
            return True
    return False
