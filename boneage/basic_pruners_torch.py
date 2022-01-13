# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported basic pruning algorithms.
In this example, we show the end-to-end pruning process: pre-training -> pruning -> fine-tuning.
Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speed up is required.
You can also try auto_pruners_torch.py to see the usage of some automatic pruning algorithms.
'''

import argparse
import os
import sys
import torch
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


str2pruner = {
    'level': LevelPruner, # Linear support
    'l1filter': L1FilterPruner,  # Conv-layers only
    'l2filter': L2FilterPruner, # Conv-layers only
    'slim': SlimPruner,  # Conv-layers only
    'agp': AGPPruner,  # Linear support only with level
    'fpgm': FPGMPruner,  # Conv-layers only
    'mean_activation': ActivationMeanRankFilterPruner, # Conv-layers only
    'apoz': ActivationAPoZRankFilterPruner,  # Conv-layers only
    'taylorfo': TaylorFOWeightFilterPruner  # Conv-layers only

    # TODO: Look into SimulatedAnnealing, LotteryTicketPruner, AutoCompress
}


def get_dummy_input(args, device):
    if args.dataset == 'boneage':
        dummy_input = torch.randn([args.test_batch_size, 1024]).to(device)
    return dummy_input


def get_data(dataset, data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'boneage':
        # Ready data
        def filter(x): return x["gender"] == 1

        df_train, df_val = get_df("adv")
        features = get_features("adv")

        # Get data with ratio
        df_1 = utils.heuristic(
            df_train, filter, float(args.ratio), 200,
            class_imbalance=1.0, n_tries=300)

        df_2 = utils.heuristic(
            df_val, filter, float(args.ratio), 700,
            class_imbalance=1.0, n_tries=300)

        ds_1 = BoneWrapper(df_1, df_2, features=features)

        train_loader, test_loader = ds_1.get_loaders(args.batch_size, shuffle=False)
        criterion = torch.nn.BCEWithLogitsLoss()

    return train_loader, test_loader, criterion


def get_model_optimizer_scheduler(path):
    # Load model
    model = load_model(path, cpu=False)
    model.train()
    scheduler = None

    # setup new optimizer for pruning
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = None

    return model, optimizer, scheduler


def train(args, model, device, train_loader, criterion, optimizer, epoch):
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
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(args, model, device, criterion, test_loader):
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

    print('Test Loss: {}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)

    # prepare model and data
    train_loader, test_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)

    basepath = os.path.join(BASE_MODELS_DIR, "adv/%s/" % args.ratio)
    some_file_path = os.listdir(basepath)[42]
    model, optimizer, _ = get_model_optimizer_scheduler(os.path.join(basepath, some_file_path))

    # Shift model to device
    model.to(device)

    # Get dummy input (for computing FLOPS etc)
    dummy_input = get_dummy_input(args, device)
    flops, params, _ = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")

    print(f'start {args.pruner} pruning...')

    def trainer(model, optimizer, criterion, epoch):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch)

    pruner_cls = str2pruner[args.pruner]

    kw_args = {}
    config_list = [{
        'sparsity': args.sparsity,
        'op_types': ['Linear']
    }]

    if args.pruner == 'level':
        config_list = [{
            'sparsity': args.sparsity,
            'op_types': ['default']
        }]

    else:
        if args.global_sort:
            print('Enable the global_sort mode')
            # only taylor pruner supports global sort mode currently
            kw_args['global_sort'] = True
        if args.dependency_aware:
            dummy_input = get_dummy_input(args, device)
            print('Enable the dependency_aware mode')
            # note that, not all pruners support the dependency_aware mode
            kw_args['dependency_aware'] = True
            kw_args['dummy_input'] = dummy_input
        if args.pruner not in ('l1filter', 'l2filter', 'fpgm'):
            # set only work for training aware pruners
            kw_args['trainer'] = trainer
            kw_args['optimizer'] = optimizer
            kw_args['criterion'] = criterion

        if args.pruner in ('mean_activation', 'apoz', 'taylorfo'):
            kw_args['sparsifying_training_batches'] = 1

        if args.pruner == 'slim':
            kw_args['sparsifying_training_epochs'] = 1

        if args.pruner == 'agp':
            kw_args['pruning_algorithm'] = 'level'
            kw_args['num_iterations'] = 10
            kw_args['epochs_per_iteration'] = 1

    pruner = pruner_cls(model, config_list, **kw_args)

    # Pruner.compress() returns the masked model
    model = pruner.compress()
    pruner.get_pruned_weights()

    print("Pruning complete")

    if args.test_only:
        test(args, model, device, criterion, test_loader)

    if args.speed_up:
        # Unwrap all modules to normal state
        pruner._unwrap_model()
        m_speedup = ModelSpeedup(model, dummy_input, mask_path, device)
        m_speedup.speedup_model()

    print('start finetuning...')

    # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    best_top1 = 0
    save_path = os.path.join(args.experiment_data_dir, f'finetuned.pth')
    for epoch in range(args.fine_tune_epochs):
        print('# Epoch {} #'.format(epoch))
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        top1 = test(args, model, device, criterion, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), save_path)

    flops, params, results = count_flops_params(model, dummy_input)
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_top1: .2f}')

    if args.nni:
        nni.report_final_result(best_top1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='boneage',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='bonemodel',
                        choices=['lenet', 'vgg16', 'vgg19', 'resnet18','bonemodel'],
                        help='model to use')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=200,
                        help='input batch size for testing')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='get perf on test data')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run on mulitple gpus')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='ratio of dataset')

    # pruner
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--dependency-aware', action='store_true', default=False,
                        help='toggle dependency aware mode')
    parser.add_argument('--global-sort', action='store_true', default=False,
                        help='toggle global sort mode')
    parser.add_argument('--pruner', type=str, default='l1filter',
                        choices=['level', 'l1filter', 'l2filter', 'slim', 'agp',
                                 'fpgm', 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use')

    # speed-up
    parser.add_argument('--speed-up', action='store_true', default=False,
                        help='Whether to speed-up the pruned model')

    # fine-tuning
    parser.add_argument('--fine-tune-epochs', type=int, default=0,
                        help='epochs to fine tune')

    parser.add_argument('--nni', action='store_true', default=False,
                        help="whether to tune the pruners using NNi tuners")

    args = parser.parse_args()

    if args.nni:
        params = nni.get_next_parameter()
        print(params)
        args.sparsity = params['sparsity']
        args.pruner = params['pruner']
        args.model = params['model']

    main(args)
