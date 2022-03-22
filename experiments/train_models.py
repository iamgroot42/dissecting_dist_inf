from simple_parsing import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import os

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train
from distribution_inference.training.utils import save_model
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.utils import flash_utils


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--load_config", help="Specify config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = None
    if args.load_config is not None:
        config = TrainConfig.load(args.load_config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(TrainConfig, dest="train_config", default=config)
    args = parser.parse_args(remaining_argv)

    # Extract configuration information from config file
    train_config: TrainConfig = args.train_config
    data_config: DatasetConfig = train_config.data_config

    # Print out arguments
    flash_utils(train_config)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object
    ds = ds_wrapper_class(data_config)

    # Train models
    for i in range(1, train_config.num_models + 1):
        print("Training classifier %d / %d" % (i, train_config.num_models))

        # Get data loaders
        train_loader, val_loader = ds.get_loaders(
            batch_size=train_config.batch_size)

        # Get model
        model = ds_info.get_model()

        # Train model
        model, (vloss, vacc) = train(model, (train_loader, val_loader),
                                     train_config=train_config)

        # Get path to save model
        file_name = str(i + train_config.offset) + ("_%.2f" % vacc)
        save_path = ds.get_save_path(train_config, file_name)

        # Save model
        save_model(model, save_path)
