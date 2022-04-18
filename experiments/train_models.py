from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from simple_parsing import ArgumentParser
from pathlib import Path

from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train
from distribution_inference.training.utils import save_model
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig
from distribution_inference.utils import flash_utils


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file", type=Path, required=True)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    train_config = TrainConfig.load(args.load_config, drop_extra_fields=False)

    # Extract configuration information from config file
    dp_config = None
    train_config: TrainConfig = train_config
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

    # Get dataset wrapper
    ds_wrapper_class = get_dataset_wrapper(data_config.name)

    # Get dataset info object
    ds_info = get_dataset_information(data_config.name)(train_config.save_every_epoch)

    # Create new DS object
    ds = ds_wrapper_class(data_config)

    # Train models
    for i in range(1, train_config.num_models + 1):
        print("Training classifier %d / %d" % (i, train_config.num_models))

        # Get data loaders
        train_loader, val_loader = ds.get_loaders(
            batch_size=train_config.batch_size)

        # Get model
        if dp_config is None:
            model = ds_info.get_model()
        else:
            model = ds_info.get_model_for_dp()

        # Train model
        model, (vloss, vacc) = train(model, (train_loader, val_loader),
                                     train_config=train_config,
                                     extra_options={
                                        "curren_model_num": i + train_config.offset,
                                        "save_path_fn": ds.get_save_path})

        # If saving only the final model
        if not train_config.save_every_epoch:
            # If adv training, suffix is a bit different
            if misc_config and misc_config.adv_config:
                suffix = "_%.2f_adv_%.2f.ch" % (vacc[0], vacc[1])
            else:
                suffix = "_%.2f.ch" % vacc

            # Get path to save model
            file_name = str(i + train_config.offset) + suffix
            save_path = ds.get_save_path(train_config, file_name)

            # Save model
            save_model(model, save_path)
