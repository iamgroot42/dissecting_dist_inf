
from tqdm import tqdm
import torch as ch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from distribution_inference.training.utils import AverageMeter, generate_adversarial_input
from distribution_inference.config import TrainConfig, AdvTrainingConfig
from distribution_inference.training.dp import train as train_with_dp
from distribution_inference.utils import warning_string


def train(model, loaders, train_config: TrainConfig):
    if train_config.misc_config and train_config.misc_config.dp_config:
        # If DP training, call appropriate function
        return train_with_dp(model, loaders, train_config)
    else:
        # If DP training, call appropriate function
        return train_without_dp(model, loaders, train_config)


def train_epoch(train_loader, model, criterion, optimizer, epoch,
                verbose: bool = True,
                adv_config: AdvTrainingConfig = None,
                expect_extra: bool = True):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    if verbose:
        iterator = tqdm(train_loader)
    for tuple in iterator:
        if expect_extra:
            data, labels, _ = tuple
        else:
            data, labels = tuple
        data, labels = data.cuda(), labels.cuda()
        N = data.size(0)

        if adv_config is None:
            # Clear accumulated gradients
            optimizer.zero_grad()
            outputs = model(data)[:, 0]
        else:
            # Adversarial inputs
            adv_x = generate_adversarial_input(model, data, adv_config)
            # Important to zero grad after above call, else model gradients
            # get accumulated over attack too
            optimizer.zero_grad()
            outputs = model(adv_x)[:, 0]

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        prediction = (outputs >= 0)
        train_acc.update(prediction.eq(
            labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        if verbose:
            iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (
                epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion,
                   verbose: bool = True,
                   adv_config: AdvTrainingConfig = None,
                   expect_extra: bool = True):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    adv_val_loss = AverageMeter()
    adv_val_acc = AverageMeter()

    with ch.set_grad_enabled(adv_config is not None):
        for tuple in val_loader:
            if expect_extra:
                data, labels, _ = tuple
            else:
                data, labels = tuple
            data, labels = data.cuda(), labels.cuda()
            N = data.size(0)

            outputs = model(data)[:, 0]
            prediction = (outputs >= 0)

            if adv_config is not None:
                adv_x = generate_adversarial_input(model, data, adv_config)
                outputs_adv = model(adv_x)[:, 0]
                prediction_adv = (outputs_adv >= 0)

                adv_val_acc.update(prediction_adv.eq(
                    labels.view_as(prediction_adv)).sum().item()/N)

                adv_val_loss.update(
                    criterion(outputs_adv, labels.float()).item())

            val_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels.float()).item())

    if verbose:
        if adv_config is None:
            print('[Validation], Loss: %.5f, Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg))
        else:
            print('[Validation], Loss: %.5f, Accuracy: %.4f | Adv-Loss: %.5f, Adv-Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg,
                   adv_val_loss.avg, adv_val_acc.avg))
        print()

    if adv_config is None:
        return val_loss.avg, val_acc.avg
    return (val_loss.avg, adv_val_loss.avg), (val_acc.avg, adv_val_acc.avg)


def train_without_dp(model, loaders, train_config: TrainConfig):
    # Get data loaders
    train_loader, val_loader = loaders

    # Define optimizer, loss function
    optimizer = ch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay)
    criterion = nn.BCEWithLogitsLoss().cuda()

    iterator = range(1, train_config.epochs + 1)
    if not train_config.verbose:
        iterator = tqdm(iterator)

    adv_config = None
    if train_config.misc_config is not None:
        adv_config = train_config.misc_config.adv_config

        # Special case for CelebA
        # Given the way scaling is done, eps (passed as argument) should be
        # 2^(1/p) for L_p norm
        if train_config.data_config.name == "celeba":
            adv_config.epsilon *= 2
            print(warning_string("Special Behavior: Doubling epsilon for Celeb-A"))

    # If eps-iter is not set, use default rule
    if adv_config is not None and adv_config.epsilon_iter is None:
        adv_config.epsilon_iter = 2.5 * adv_config.epsilon / adv_config.iters

    best_model, best_loss = None, np.inf
    for epoch in iterator:
        _, tacc = train_epoch(train_loader, model,
                              criterion, optimizer, epoch,
                              verbose=train_config.verbose,
                              adv_config=adv_config,
                              expect_extra=train_config.expect_extra)

        vloss, vacc = validate_epoch(
            val_loader, model, criterion,
            verbose=train_config.verbose,
            adv_config=adv_config,
            expect_extra=train_config.expect_extra)
        if not train_config.verbose:
            if adv_config is None:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f |" % (100 * tacc, 100 * vacc))
            else:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f | adv_val_acc: %.2f" % (100 * tacc, 100 * vacc[0], 100 * vacc[1]))

        vloss_compare = vloss
        if adv_config is not None:
            vloss_compare = vloss[0]

        if train_config.get_best and vloss_compare < best_loss:
            best_loss = vloss_compare
            best_model = deepcopy(model)

    # Special case for CelebA
    # Return epsilon back to normal
    if train_config.misc_config is not None and train_config.data_config.name == "celeba":
        adv_config.epsilon /= 2

    if train_config.get_best:
        return best_model, (vloss, vacc)
    return vloss, vacc
