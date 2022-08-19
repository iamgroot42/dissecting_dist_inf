from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
import torch as ch
import torch.nn as nn
from tqdm import tqdm
from distribution_inference.utils import warning_string
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train_epoch, validate_epoch
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import TrainingResult
import os
import torch.nn as nn
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv


def train(model, loaders, loader2, train_config: TrainConfig,
          input_is_list: bool = False,
          extra_options: dict = None):
    if extra_options != None and "more_metrics" in extra_options.keys():
        more_metrics = extra_options["more_metrics"]
    else:
        more_metrics = False
    # Wrap with dataparallel if requested
    if train_config.parallel:
        model = ch.nn.DataParallel(model)

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

    tlosses, tacces, R_crosses = [], [], []
    vlosses, vacces = [], []
    for epoch in iterator:
        tloss, tacc = train_epoch(train_loader, model,
                                  criterion, optimizer, epoch,
                                  verbose=train_config.verbose,
                                  adv_config=adv_config,
                                  expect_extra=train_config.expect_extra,
                                  input_is_list=input_is_list,
                                  regression=train_config.regression,
                                  multi_class=train_config.multi_class)
        tlosses.append(tloss)
        tacces.append(tacc)

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
        vlosses.append(vloss)
        vacces.append(vacc)
        R_crosse, _ = validate_epoch(
            loader2,
            model, criterion,
            verbose=False,
            adv_config=None,
            expect_extra=train_config.expect_extra,
            input_is_list=False,
            regression=train_config.regression,
            multi_class=train_config.multi_class)
        R_crosses.append(R_crosse)
        # LR Scheduler, if requested
        if lr_scheduler is not None:
            lr_scheduler.step(vloss)

    R_crosses = list(np.array(R_crosses)-np.array(vlosses))
    # Now that training is over, remove dataparallel wrapper
    if train_config.parallel:
        model = model.module

    return vlosses, vacces, tlosses, tacces, R_crosses


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file", type=Path)
    parser.add_argument(
        "--ratios", nargs='+', help="ratios", type=float)
    parser.add_argument(
        "--split", type=str)
    parser.add_argument(
        "--exp", type=str)
    parser.add_argument('--gpu',
                        default=None, help="device number")
    parser.add_argument('--offset',
                        default=0, type=int)
    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_config = TrainConfig.load(args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    dp_config = None
    train_config: TrainConfig = train_config
    data_config: DatasetConfig = train_config.data_config
    misc_config: MiscTrainConfig = train_config.misc_config
    print(args.ratios)
    assert type(args.ratios) == list, "{} is {}".format(
        args.ratios, type(args.ratios))
    if args.split:
        data_config.split = args.split
        train_config.data_config.split = args.split
    if misc_config is not None:
        dp_config: DPTrainingConfig = misc_config.dp_config

        # TODO: Figure out best place to have this logic in the module
        if misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(train_config)
    exp = args.exp if args.exp else "not saving"
    # Define logger
    offset = args.offset if args.offset else train_config.offset
    exp_name = "_".join([train_config.data_config.split, train_config.data_config.prop,
                        train_config.model_arch, exp, str(offset)])
    logger = TrainingResult(exp_name, train_config)
    for ratio in args.ratios:
        data_config.value = ratio
        train_config.data_config.value = ratio
        # Get dataset wrapper
        ds_wrapper_class = get_dataset_wrapper(data_config.name)

        # Get dataset info object
        ds_info = get_dataset_information(
            data_config.name)(train_config.save_every_epoch)

        # Create new DS object
        ds = ds_wrapper_class(data_config, epoch=train_config.save_every_epoch)
        """
        # train_ds, val_ds = ds.load_data()
        # print(len(train_ds))
        # print(len(val_ds))
        # exit(0)
        
        train_ds, val_ds = ds.load_data()
        y=[]
        for t in val_ds:
            y.append(t[1])
        print("loaded")
        y = np.array(y)
        print(max(np.mean(y==1), 1 - np.mean(y==1)))
        """
        # Train models
        for i in range(1, train_config.num_models + 1):
            # Skip training model if it already exists
            if not train_config.save_every_epoch:
                save_path = ds.get_save_path(train_config, None)
                if ds.check_if_exists(save_path, str(i + offset)):
                    print(
                        f"Model {i + offset} already exists. Skipping training.")
                    continue

            print("Training classifier %d / %d" % (i, train_config.num_models))

            # Get data loaders
            train_loader, val_loader = ds.get_loaders(
                batch_size=train_config.batch_size)

            # Get model
            if dp_config is None:
                model = ds_info.get_model(model_arch=train_config.model_arch)
            else:
                model = ds_info.get_model_for_dp()

            # Train model
            _, data_config_vic = get_dfs_for_victim_and_adv(
                data_config, prop_value=0.5)
            ds_vic_2 = ds_wrapper_class(
                data_config_vic,
                label_noise=train_config.label_noise,
                epoch=True)
            _, loader2 = ds_vic_2.get_loaders(
                batch_size=train_config.batch_size)
            vlosses, vacces, tlosses, tacces, R_crosses = train(model, (train_loader, val_loader), loader2,
                                                                train_config=train_config,
                                                                extra_options={
                "curren_model_num": i + offset,
                "save_path_fn": ds.get_save_path})

            extras = {"train_loss": tlosses, "train_acc": tacces, "loss_dif": list(np.array(
                vlosses)-np.array(tlosses)), "acc_dif": list(np.array(tacces)-np.array(vacces))}
            logger.add_result(data_config.value, vlosses,
                              vacces, extras, R_cross=list(R_crosses))

    # Save logger
    if args.exp:
        logger.save()
