# Handle multiple workers
from traceback import print_list
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from distribution_inference.defenses.active.augment import AugmentDefense
from simple_parsing import ArgumentParser
from pathlib import Path
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train
from distribution_inference.training.utils import save_model
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import TrainingResult
from distribution_inference.defenses.active.shuffle import ShuffleDefense


EXTRA = False  #True
if __name__ == "__main__":
    print("started")
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = TrainConfig.load(args.load_config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(TrainConfig, dest="train_config", default=config)
    args = parser.parse_args(remaining_argv)
    train_config = args.train_config

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
    ds_info = get_dataset_information(
        data_config.name)(train_config.save_every_epoch)
    exp_name = "_".join([config.data_config.split,
                        config.data_config.prop,
                        config.model_arch if config.model_arch else ds_info.default_model,
                        str(config.data_config.value),
                        str(config.offset)])
    # Define logger
    
    logger = TrainingResult(exp_name, train_config)
   
    # If ShuffleDefense, get non-shuffled train loader, process, then get actual ones
    shuffle_defense = None
    if train_config.misc_config is not None:
        shuffle_defense_config = train_config.misc_config.shuffle_defense_config
        if shuffle_defense_config and not train_config.expect_extra:
            raise ValueError(
                "Need access to property labels for shuffle defense. Set expect_extra to True")

        if shuffle_defense_config is not None:
            if shuffle_defense_config.augment:
                shuffle_defense = AugmentDefense(shuffle_defense_config)
            else:
                shuffle_defense = ShuffleDefense(shuffle_defense_config)

    # Create new DS object
    ds = ds_wrapper_class(data_config,
                         epoch=train_config.save_every_epoch,
                         shuffle_defense=shuffle_defense,
                         label_noise=train_config.label_noise)

    # train_ds, val_ds = ds.load_data()
    # print(len(train_ds))
    # print(len(val_ds))
    # exit(0)
    # train_ds, val_ds = ds.load_data()
    # y = []
    # for t in val_ds:
        # y.append(t[1])
    # print("loaded")
    # y = np.array(y)
    # print(max(np.mean(y == 1), 1 - np.mean(y == 1)))

    # Train models
    for i in range(1, train_config.num_models + 1):
        # Skip training model if it already exists
        """"
        if not train_config.save_every_epoch:
            save_path = ds.get_save_path(train_config, None)
            if ds.check_if_exists(save_path, str(i + train_config.offset)):
                print(
                    f"Model {i + train_config.offset} already exists. Skipping training.")
                continue
        """
        print("Training classifier %d / %d" % (i, train_config.num_models))

        # Get data loaders
        train_loader, val_loader = ds.get_loaders(
            batch_size=train_config.batch_size)
        plist = []
        for t in train_loader:
            _,_,prop_l = t
            for k in prop_l:
                plist.append(k)
        print(np.mean(plist))
        # Get model
        if dp_config is None:
            model = ds_info.get_model(model_arch=train_config.model_arch)
        else:
            model = ds_info.get_model_for_dp()

        # Train model
        if EXTRA:
            model, (vloss, vacc, extras) = train(model, (train_loader, val_loader),
                                                 train_config=train_config,
                                                 extra_options={
                "curren_model_num": i + train_config.offset,
                "save_path_fn": ds.get_save_path,
                "more_metrics": EXTRA},
                shuffle_defense=shuffle_defense)
            logger.add_result(data_config.value, vloss, vacc, extras)
        else:
            model, (vloss, vacc) = train(model, (train_loader, val_loader),
                                         train_config=train_config,
                                         extra_options={
                "curren_model_num": i + train_config.offset,
                "save_path_fn": ds.get_save_path},
                shuffle_defense=shuffle_defense)
            logger.add_result(data_config.value, vloss, vacc)

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
            #save_model(model, save_path)

            # Save logger
            logger.save()
