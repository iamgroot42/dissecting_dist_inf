from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from distribution_inference.config.core import DPTrainingConfig, MiscTrainConfig
from simple_parsing import ArgumentParser
from pathlib import Path
import numpy as np
from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.training.core import train
from distribution_inference.training.utils import save_model
from distribution_inference.config import TrainConfig, DatasetConfig, MiscTrainConfig
from distribution_inference.utils import flash_utils
from sympy import true

def get_model(max_iter=40,
              hidden_layer_sizes=(1024,512,128,64),):
    """
        Create new MLPClassifier model
    """
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter)
    return clf

if __name__ == "__main__":
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
    ds_info = get_dataset_information(data_config.name)(train_config.save_every_epoch)

    # Create new DS object
    ds = ds_wrapper_class(data_config)
# Get data loaders
    (x_tr, y_tr), (x_te, y_te), _ = ds.load_data(None)
        # Get model
    model = get_model()
    N_TRAIN_SAMPLES = x_tr.shape[0]
    N_BATCH = train_config.batch_size
    N_CLASSES = np.unique(y_tr)
    # Train models
    for epoch in range(train_config.epochs):
        # Train model
        random_perm = np.random.permutation(N_TRAIN_SAMPLES)
        mini_batch_index = 0
        while True:
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            model.partial_fit(x_tr[indices],y_tr[indices],classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break
        print("loss :"+str(model.loss_))
        print("Train acc: "+str(model.score(x_tr,y_tr)))
        print("Validation acc: "+str(model.score(x_te,y_te)))

        