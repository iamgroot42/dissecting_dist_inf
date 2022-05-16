
from tqdm import tqdm
import torch as ch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import os

from distribution_inference.training.utils import AverageMeter, generate_adversarial_input, save_model
from distribution_inference.config import TrainConfig, AdvTrainingConfig
from distribution_inference.training.dp import train as train_with_dp
from distribution_inference.utils import warning_string


def train(model, loaders, train_config: TrainConfig,
          input_is_list: bool = False,
          extra_options: dict = None):
    if train_config.misc_config and train_config.misc_config.dp_config:
        # If DP training, call appropriate function
        return train_with_dp(model, loaders, train_config, input_is_list, extra_options)
    else:
        # If DP training, call appropriate function
        return train_without_dp(model, loaders, train_config, input_is_list, extra_options)


def train_epoch(train_loader, model, criterion, optimizer, epoch,
                verbose: bool = True,
                adv_config: AdvTrainingConfig = None,
                expect_extra: bool = True,
                input_is_list: bool = False,
                regression: bool = False,
                multi_class: bool = False):
    model.train()
    train_loss = AverageMeter()
    if not regression:
        train_acc = AverageMeter()
    iterator = train_loader
    if verbose:
        iterator = tqdm(train_loader)
    for tuple in iterator:
        # Extract data
        if expect_extra:
            data, labels, _ = tuple
        else:
            data, labels = tuple

        # Support for using same code for AMC
        if input_is_list:
            data = [x.cuda() for x in data]
        else:
            data = data.cuda()
        labels = labels.cuda()
        N = labels.size(0)

        if adv_config is None:
            # Clear accumulated gradients
            optimizer.zero_grad()
            outputs = model(data)
            if not multi_class:
                outputs = outputs[:, 0]
        else:
            # Generate adversarial inputs
            adv_x = generate_adversarial_input(model, data, adv_config)
            # Important to zero grad after above call, else model gradients
            # get accumulated over attack too
            optimizer.zero_grad()
            outputs = model(adv_x)
            if not multi_class:
                outputs = outputs[:, 0]

        loss = criterion(outputs, labels.long() if multi_class else labels.float())
        loss.backward()
        optimizer.step()
        if not regression:
            if multi_class:
                prediction = outputs.argmax(dim=1)
            else:
                prediction = (outputs >= 0)
            train_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        if verbose:
            if regression:
                iterator.set_description('[Train] Epoch %d, Loss: %.5f' % (
                    epoch, train_loss.avg))
            else:
                iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (
                    epoch, train_loss.avg, train_acc.avg))
    if regression:
        return train_loss.avg, None
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion,
                   verbose: bool = True,
                   adv_config: AdvTrainingConfig = None,
                   expect_extra: bool = True,
                   input_is_list: bool = False,
                   regression: bool = False,
                   get_preds: bool = False,
                   multi_class: bool = False):
    model.eval()
    val_loss = AverageMeter()
    adv_val_loss = AverageMeter()
    if not regression:
        val_acc = AverageMeter()
        adv_val_acc = AverageMeter()

    collected_preds = []
    with ch.set_grad_enabled(adv_config is not None):
        for tuple in val_loader:
            if expect_extra:
                data, labels, _ = tuple
            else:
                data, labels = tuple
            if input_is_list:
                data = [x.cuda() for x in data]
            else:
                data = data.cuda()
            labels = labels.cuda()
            N = labels.size(0)

            # Get model outputs
            outputs = model(data)
            if not multi_class:
                outputs = outputs[:, 0]

            if get_preds:
                collected_preds.append(outputs.detach().cpu().numpy())
            if not regression:
                if multi_class:
                    prediction = outputs.argmax(dim=1)
                else:
                    prediction = (outputs >= 0)
                val_acc.update(prediction.eq(
                    labels.view_as(prediction)).sum().item()/N)
            val_loss_specific = criterion(
                outputs, labels.long() if multi_class else labels.float())
            val_loss.update(val_loss_specific.item())

            if adv_config is not None:
                adv_x = generate_adversarial_input(model, data, adv_config)
                # Get model outputs
                outputs_adv = model(adv_x)
                if not multi_class:
                    outputs = outputs[:, 0]

                if not regression:
                    if multi_class:
                        prediction_adv = outputs_adv.argmax(dim=1)
                    else:
                        prediction_adv = (outputs_adv >= 0)
                    adv_val_acc.update(prediction_adv.eq(
                        labels.view_as(prediction_adv)).sum().item()/N)
                adv_loss_specific = criterion(
                    outputs_adv, labels.long() if multi_class else labels.float())
                adv_val_loss.update(adv_loss_specific.item())

    if verbose:
        if adv_config is None:
            if regression:
                print('[Validation], Loss: %.5f,' % val_loss.avg)
            else:
                print('[Validation], Loss: %.5f, Accuracy: %.4f' %
                      (val_loss.avg, val_acc.avg))
        else:
            if regression:
                print('[Validation], Loss: %.5f | Adv-Loss: %.5f' %
                      (val_loss.avg,  adv_val_loss.avg))
            else:
                print('[Validation], Loss: %.5f, Accuracy: %.4f | Adv-Loss: %.5f, Adv-Accuracy: %.4f' %
                      (val_loss.avg, val_acc.avg, adv_val_loss.avg, adv_val_acc.avg))
        print()

    if get_preds:
        collected_preds = np.concatenate(collected_preds, axis=0)

    if adv_config is None:
        if regression:
            if get_preds:
                return val_loss.avg, None, collected_preds
            else:
                return val_loss.avg, None
        if get_preds:
            return val_loss.avg, val_acc.avg, collected_preds
        else:
            return val_loss.avg, val_acc.avg
    if regression:
        if get_preds:
            return (val_loss.avg, adv_val_loss.avg), (None, None), collected_preds
        else:
            return (val_loss.avg, adv_val_loss.avg), (None, None)
    if get_preds:
        return (val_loss.avg, adv_val_loss.avg), (val_acc.avg, adv_val_acc.avg), collected_preds    
    return (val_loss.avg, adv_val_loss.avg), (val_acc.avg, adv_val_acc.avg)


def train_without_dp(model, loaders, train_config: TrainConfig,
                     input_is_list: bool = False,
                     extra_options: dict = None):
    # Get data loaders
    if len(loaders) == 2:
        train_loader, test_loader = loaders
        val_loader = None
        if train_config.get_best:
            print(warning_string("\nUsing test-data to pick best-performing model\n"))
    else:
        train_loader, test_loader, val_loader = loaders

    # Define optimizer, loss function
    optimizer = ch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay)
    if train_config.regression:
        criterion = nn.MSELoss()
    elif train_config.multi_class:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    # LR Scheduler
    if train_config.lr_scheduler is not None:
        scheduler_config = train_config.lr_scheduler
        if len(loaders) == 2:
            print(warning_string("\nUsing LR scheduler on test data\n"))
        lr_scheduler = ch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            cooldown=scheduler_config.cooldown,
            verbose=scheduler_config.verbose)
    else:
        lr_scheduler = None

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
        tloss, tacc = train_epoch(train_loader, model,
                                  criterion, optimizer, epoch,
                                  verbose=train_config.verbose,
                                  adv_config=adv_config,
                                  expect_extra=train_config.expect_extra,
                                  input_is_list=input_is_list,
                                  regression=train_config.regression,
                                  multi_class=train_config.multi_class)

        # Get metrics on val data, if available
        if val_loader is not None:
            use_loader_for_metric_log = val_loader
        else:
            use_loader_for_metric_log = test_loader
        vloss, vacc = validate_epoch(use_loader_for_metric_log,
                                     model, criterion,
                                     verbose=train_config.verbose,
                                     adv_config=adv_config,
                                     expect_extra=train_config.expect_extra,
                                     input_is_list=input_is_list,
                                     regression=train_config.regression,
                                     multi_class=train_config.multi_class)

        # LR Scheduler, if requested
        if lr_scheduler is not None:
            lr_scheduler.step(vloss)

        # Log appropriate metrics
        if not train_config.verbose:
            if adv_config is None:
                if train_config.regression:
                    iterator.set_description(
                        "train_loss: %.2f | val_loss: %.2f |" % (tloss, vloss))
                else:
                    iterator.set_description(
                        "train_acc: %.2f | val_acc: %.2f | train_loss: %.3f | val_loss: %.3f" % (100 * tacc, 100 * vacc, tloss, vloss))
            else:
                if train_config.regression:
                    iterator.set_description(
                        "train_loss: %.2f | val_loss: %.2f | adv_val_loss: %.2f" % (tloss, vloss[0], vloss[1]))
                else:
                    iterator.set_description(
                        "train_acc: %.2f | val_acc: %.2f | adv_val_acc: %.2f" % (100 * tacc, 100 * vacc[0], 100 * vacc[1]))

        vloss_compare = vloss
        if adv_config is not None:
            vloss_compare = vloss[0]

        if train_config.get_best and vloss_compare < best_loss:
            best_loss = vloss_compare
            best_model = deepcopy(model)

        if train_config.save_every_epoch:
            # If adv training, suffix is a bit different
            if train_config.misc_config and train_config.misc_config.adv_config:
                if train_config.regression:
                    suffix = "_%.4f_adv_%.4f.ch" % (vloss[0], vloss[1])
                else:
                    suffix = "_%.2f_adv_%.2f.ch" % (vacc[0], vacc[1])
            else:
                if train_config.regression:
                    suffix = "_tr%.4f_te%.4f.ch" % (tloss, vloss)
                else:
                    suffix = "_tr%.2f_te%.2f.ch" % (tacc, vacc)

            # Get model "name" and function to save model
            model_num = extra_options.get("curren_model_num")
            save_path_fn = extra_options.get("save_path_fn")

            # Save model in current epoch state
            file_name = os.path.join(str(epoch), str(
                model_num + train_config.offset) + suffix)
            save_path = save_path_fn(train_config, file_name)
            # Make sure this directory exists
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            save_model(model, save_path)

    # Special case for CelebA
    # Return epsilon back to normal
    if train_config.misc_config is not None and train_config.data_config.name == "celeba":
        adv_config.epsilon /= 2

    # Use test-loader to compute final test metrics
    if val_loader is not None:
        test_loss, test_acc = validate_epoch(
                test_loader,
                model, criterion,
                verbose=False,
                adv_config=adv_config,
                expect_extra=train_config.expect_extra,
                input_is_list=input_is_list,
                regression=train_config.regression,
                multi_class=train_config.multi_class)
    else:
        test_loss, test_acc = vloss, vacc

    if train_config.get_best:
        return best_model, (test_loss, test_acc)
    return test_loss, test_acc
