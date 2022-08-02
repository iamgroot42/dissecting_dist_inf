"""
This script generates loss, acc, and R cross
"""
from distribution_inference.training.core import validate_epoch
from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import TrainingResult
import os
import torch.nn as nn
from distribution_inference.attacks.utils import get_dfs_for_victim_and_adv
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file", type=Path)
    parser.add_argument(
        "--en", help="experiment name",
        type=str, required=True)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument(
        "--ratios", nargs='+', help="ratios", type=float)
    parser.add_argument("--D0", default=0.5, type=float)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_config: TrainConfig = TrainConfig.load(
        args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    dp_config = None
    data_config: DatasetConfig = train_config.data_config
    misc_config: MiscTrainConfig = train_config.misc_config
    if misc_config is not None:
        dp_config: DPTrainingConfig = misc_config.dp_config

        # TODO: Figure out best place to have this logic in the module
        if misc_config.adv_config:
            # Scale epsilon by 255 if requested
            if train_config.misc_config.adv_config.scale_by_255:
                train_config.misc_config.adv_config.epsilon /= 255

    # Print out arguments
    flash_utils(train_config)
    data_config.value = args.D0
    train_config.data_config.value = args.D0
    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(
        data_config.name)(True)
    # Define logger
    exp_name = "_".join([train_config.data_config.split, train_config.data_config.prop,
                        train_config.model_arch, args.en, str(train_config.offset)])
    logger = TrainingResult(exp_name, train_config)
    ds_vic_1 = ds_wrapper_class(
        data_config,
        label_noise=train_config.label_noise)
    _, loader1 = ds_vic_1.get_loaders(batch_size=train_config.batch_size)

    if train_config.regression:
        criterion = nn.MSELoss()
    elif train_config.multi_class:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for ratio in args.ratios:
        _, data_config_vic = get_dfs_for_victim_and_adv(
            data_config, prop_value=ratio)
        ds_vic_2 = ds_wrapper_class(
            data_config_vic,
            label_noise=train_config.label_noise,
            epoch=True)
        _, loader2 = ds_vic_2.get_loaders(batch_size=train_config.batch_size)

        # Get model
        models_vic = ds_vic_2.get_models(
            train_config,
            n_models=train_config.num_models,
            on_cpu=False,
            shuffle=False,
            epochwise_version=True,
            model_arch=train_config.model_arch,
            custom_models_path=None)

        for i in tqdm(range(1, train_config.num_models + 1)):
            vloss, vacc, R_cross = [], [], []

            for e in range(train_config.epochs):
                model = models_vic[e][i-1]
                vlosse, vacce = validate_epoch(
                    loader2,
                    model, criterion,
                    verbose=True,
                    adv_config=None,
                    expect_extra=train_config.expect_extra,
                    input_is_list=False,
                    regression=train_config.regression,
                    multi_class=train_config.multi_class)

                R_crosse, _ = validate_epoch(
                    loader1,
                    model, criterion,
                    verbose=False,
                    adv_config=None,
                    expect_extra=train_config.expect_extra,
                    input_is_list=False,
                    regression=train_config.regression,
                    multi_class=train_config.multi_class)

                vloss.append(vlosse)
                vacc.append(vacce)
                R_cross.append(R_crosse)

            R_cross = np.array(R_cross) - np.array(vloss)
            logger.add_result(ratio, vloss, vacc, R_cross = list(R_cross))

    # Save logger
    logger.save()
