
from tqdm import tqdm
import torch as ch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent

from distribution_inference.training.utils import AverageMeter


def train_epoch(train_loader, model, criterion, optimizer, epoch,
                verbose=True, adv_train=False, expect_extra=True):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    if verbose:
        iterator = tqdm(train_loader)
    for data in iterator:
        if expect_extra:
            images, labels, _ = data
        else:
            images, labels = data
        images, labels = images.cuda(), labels.cuda()
        N = images.size(0)

        if adv_train is False:
            # Clear accumulated gradients
            optimizer.zero_grad()
            outputs = model(images)[:, 0]
        else:
            # Adversarial inputs
            adv_x = projected_gradient_descent(
                model, images, eps=adv_train['eps'],
                eps_iter=adv_train['eps_iter'],
                nb_iter=adv_train['nb_iter'],
                norm=adv_train['norm'],
                clip_min=adv_train['clip_min'],
                clip_max=adv_train['clip_max'],
                random_restarts=adv_train['random_restarts'],
                binary_sigmoid=True)
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


def validate_epoch(val_loader, model, criterion, verbose=True, adv_train=False, expect_extra=True):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    adv_val_loss = AverageMeter()
    adv_val_acc = AverageMeter()

    with ch.set_grad_enabled(adv_train is not False):
        for data in val_loader:
            if expect_extra:
                images, labels, _ = data
            else:
                images, labels = data
            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)

            outputs = model(images)[:, 0]
            prediction = (outputs >= 0)

            if adv_train is not False:
                adv_x = projected_gradient_descent(
                    model, images, eps=adv_train['eps'],
                    eps_iter=adv_train['eps_iter'],
                    nb_iter=adv_train['nb_iter'],
                    norm=adv_train['norm'],
                    clip_min=adv_train['clip_min'],
                    clip_max=adv_train['clip_max'],
                    random_restarts=adv_train['random_restarts'],
                    binary_sigmoid=True)
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
        if adv_train is False:
            print('[Validation], Loss: %.5f, Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg))
        else:
            print('[Validation], Loss: %.5f, Accuracy: %.4f | Adv-Loss: %.5f, Adv-Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg,
                   adv_val_loss.avg, adv_val_acc.avg))
        print()

    if adv_train is False:
        return val_loss.avg, val_acc.avg
    return (val_loss.avg, adv_val_loss.avg), (val_acc.avg, adv_val_acc.avg)


def train(model, loaders, lr=1e-3, epoch_num=10,
          weight_decay=0, verbose=True, get_best=False,
          adv_train=False, expect_extra=True):
    # Get data loaders
    train_loader, val_loader = loaders

    # Define optimizer, loss function
    optimizer = ch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().cuda()

    iterator = range(1, epoch_num+1)
    if not verbose:
        iterator = tqdm(iterator)

    best_model, best_loss = None, np.inf
    for epoch in iterator:
        _, tacc = train_epoch(train_loader, model,
                              criterion, optimizer, epoch,
                              verbose=verbose, adv_train=adv_train,
                              expect_extra=expect_extra)

        vloss, vacc = validate_epoch(
            val_loader, model, criterion, verbose=verbose,
            adv_train=adv_train,
            expect_extra=expect_extra)
        if not verbose:
            if adv_train is False:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))
            else:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f | adv_val_acc: %.2f" % (tacc, vacc[0], vacc[1]))

        vloss_compare = vloss
        if adv_train is not False:
            vloss_compare = vloss[0]

        if get_best and vloss_compare < best_loss:
            best_loss = vloss_compare
            best_model = deepcopy(model)

    if get_best:
        return best_model, (vloss, vacc)
    return vloss, vacc
