"""
    Evalate task accuracy for a given set of models.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import TrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.training.core import validate_epoch


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file",
        type=Path, required=True)
    # parser.add_argument(
    #     "--en", help="experiment name",
    #     type=str, required=True)
    parser.add_argument(
        "--victim_path", help="path to victim'smodels directory",
        type=str, default=None)
    args = parser.parse_args()
    # Attempt to extract as much information from config file as you can
    train_config: TrainConfig = TrainConfig.load(
        args.load_config, drop_extra_fields=False)
    data_config = train_config.data_config

    # Print out arguments
    flash_utils(train_config)

    # Define logger
    # logger = AttackResult(args.en, attack_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create DS object
    ds = ds_wrapper_class(data_config)
    # Get data loaders
    _, val_loader = ds.get_loaders(
        batch_size=train_config.batch_size)

    # Load models
    models = ds.get_models(
        train_config,
        n_models=train_config.num_models,
        on_cpu=True,
        shuffle=False,
        model_arch=train_config.model_arch,
        custom_models_path=args.victim_path)

    # Define criterion
    if train_config.regression:
        criterion = nn.MSELoss()
    elif train_config.multi_class:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    losses, accs = [], []
    # TODO: Maintain running average and display in iterator
    for model in tqdm(models):
        # Shift model to GPU
        model.cuda()
        # Record loss and accuracy
        vloss, vacc = validate_epoch(val_loader, model, criterion,
                                     verbose=train_config.verbose,
                                     adv_config=None,
                                     expect_extra=train_config.expect_extra,
                                     regression=train_config.regression,
                                     multi_class=train_config.multi_class)
        losses.append(vloss)
        accs.append(vacc)

    losses = np.array(losses)
    accs = np.array(accs)
    print(np.mean(losses), np.std(losses))
    print(np.mean(accs), np.std(accs))

    # Save logger results
    # logger.save()
