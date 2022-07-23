from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train
from distribution_inference.training.utils import save_model
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig
from distribution_inference.utils import flash_utils
from distribution_inference.logging.core import TrainingResult
import os
EXTRA=False
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_config", help="Specify config file", type=Path)
    parser.add_argument(
        "--ratios", nargs='+',help="ratios", type=float)
    parser.add_argument(
        "--split", type=str)
    parser.add_argument(
        "--exp", type=str)
    parser.add_argument('--gpu',
                        default='0,1,2,3', help="device number")
    parser.add_argument('--offset',
                        default=0,type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_config = TrainConfig.load(args.load_config, drop_extra_fields=False)
    
    # Extract configuration information from config file
    dp_config = None
    train_config: TrainConfig = train_config
    data_config: DatasetConfig = train_config.data_config
    misc_config: MiscTrainConfig = train_config.misc_config
    print(args.ratios)
    assert type(args.ratios)==list, "{} is {}".format(args.ratios,type(args.ratios))
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
                        train_config.model_arch, exp,str(offset)])
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
        ds = ds_wrapper_class(data_config,epoch=train_config.save_every_epoch)
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
            if EXTRA:
                model, (vloss, vacc,extras) = train(model, (train_loader, val_loader),
                                        train_config=train_config,
                                        extra_options={
                "curren_model_num": i + offset,
                "save_path_fn": ds.get_save_path,
                "more_metrics":EXTRA})
                logger.add_result(data_config.value, vloss, vacc,extras)
                print("Precision:{},Recall:{},F1:{}".format(extras["precision"],extras["recall"],extras["F1"]))
            else:
                model, (vloss, vacc) = train(model, (train_loader, val_loader),
                                        train_config=train_config,
                                        extra_options={
                "curren_model_num": i + offset,
                "save_path_fn": ds.get_save_path})
                logger.add_result(data_config.value, vloss, vacc)
            
        

            # If saving only the final model
            if not train_config.save_every_epoch:
                # If adv training, suffix is a bit different
                if misc_config and misc_config.adv_config:
                    suffix = "_%.2f_adv_%.2f.ch" % (vacc[0], vacc[1])
                else:
                    suffix = "_%.2f.ch" % vacc

                # Get path to save model
                file_name = str(i + offset) + suffix
                save_path = ds.get_save_path(train_config, file_name)

                # Save model
                save_model(model, save_path)

    # Save logger
    if args.exp:
        logger.save()