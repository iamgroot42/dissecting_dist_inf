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
    parser.add_argument("--config_file", help="Specify config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    if args.config_file is not None:
        config = TrainConfig.load(args.config_file, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(TrainConfig, dest="train_config", default=config)
    args = parser.parse_args(remaining_argv)

    # Extract configuration information from config file
    train_config: TrainConfig = args.train_config
    data_config: DatasetConfig = train_config.data_config

    # Print out arguments
    flash_utils(train_config)
    exit(0)

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)()

    # Create new DS object
    ds = ds_wrapper_class(data_config)

    # Define iterator (for training models)
    iterator = range(1, train_config.num_models + 1)
    if not train_config.verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        if train_config.verbose:
            print("Training classifier %d" % i)

        # Get data loaders
        train_loader, val_loader = ds.get_loaders(batch_size=train_config.batch_size, squeeze=True)

        # Get model
        model = ds_info.get_model()

        # Train model
        model, (vloss, vacc) = train(model, (train_loader, val_loader),
                                    lr=train_config.learning_rate,
                                    epoch_num=train_config.epochs,
                                    weight_decay=train_config.weight_decay,
                                    verbose=train_config.verbose,
                                    get_best=train_config.get_best)

        # Get path to save model
        file_name = str(i + train_config.offset) + ("_%.2f" % vacc)
        save_path = ds.get_save_path(train_config, file_name)

        # Save model
        print(save_path)
        exit(0)
        # model_utils.save_model(clf, save_path)
