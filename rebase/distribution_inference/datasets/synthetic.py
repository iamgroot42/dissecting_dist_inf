import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import numpy as np
import torch as ch
from torch.utils.data import TensorDataset
import torch.nn as nn
import copy
from typing import List, Literal

from distribution_inference.config import SyntheticDatasetConfig, TrainConfig, DatasetConfig
from distribution_inference.models.core import AnyLayerMLP, RandomForest, LRClassifier
from distribution_inference.training.utils import load_model
import distribution_inference.datasets.base as base


# Mapping between numbers and corresponding configs
# Useful to point to relevant folder to load data/models from
# and keep track of different kinds of experiments
# TODO: Read JSON from file and parse
CONFIG_MAPPING = {
    "1": SyntheticDatasetConfig(
        dimensionality=10,
        adv_noise_to_mean=1.0,
        noise_cov = 0.98,
        noise_model = 1.0,
        mean_range = 5.0,
        dist_diff_mean = 1.0,
        dist_diff_std = 1.0,
        layer = [15,5], 
        cov = 3.0,
        n_samples_vic = 5000,
        n_samples_adv=1000,
        diff_posteriors=True
  )
}

class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Use 'properties' to refer to specific configs
        super().__init__(name="Synthetic Data",
                         data_path="synthetic",
                         models_path="models/synthetic",
                         properties=list(CONFIG_MAPPING.keys()),
                         values=None,
                         supported_models=[f"mlp_{d}" for d in range(1, 5)],
                         default_model="mlp_1",
                         epoch_wise=epoch_wise)

    def get_model(self, n_inp: int, n_classes: int, cpu: bool = False, model_arch: str = None) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model
        if model_arch.startswith("mlp_"):
            depth = int(model_arch.split("mlp_")[1])
            model = AnyLayerMLP(n_inp=n_inp, n_classes=n_classes, depth=depth)
        elif model_arch == "random_forest":
            model = RandomForest(min_samples_leaf=5, n_jobs=4, n_estimators=10)
        elif model_arch == "lr":
            model = LRClassifier()
        else:
            raise NotImplementedError("Model architecture not supported")

        if not model.is_sklearn_model and not cpu:
            model = model.cuda()
        return model

    def get_model_for_dp(self, n_inp:int, n_classes:int, cpu: bool = False, model_arch: str = None) -> nn.Module:
        if model_arch == "mlp_1":
            model = AnyLayerMLP(n_inp=n_inp, n_classes=n_classes, depth=1)
        else:
            raise NotImplementedError("Model architecture not supported")

        if not cpu:
            model = model.cuda()
        return model


class GroundTruthModel(nn.Module):
    """
        Model f(x) used to generate ground-truth predictions.
    """
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: List[int]):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        layers = nn.ModuleList()
        for hdim in hidden_dim:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            current_dim = hdim
        layers.append(nn.Linear(current_dim, self.n_classes))
        self.model = nn.Sequential(*layers)
        self._weights_init()

    def _weights_init(self):
        for layer in self.model:
            if type(layer) == nn.Linear:
                layer.weight.data = ch.randn(layer.weight.size())
                layer.bias.data = ch.randn(layer.bias.size())

    def forward(self, x):
        return self.model(x)


class Distribution:
    """
        Multivariate Gaussian, with ground-truth labels generated
        using given y_model.
    """

    def __init__(self, mean, cov, y_model):
        self.mean = mean
        self.cov = cov
        self.y_model = y_model
    
    def get_params(self):
        return {
            "mean": self.mean,
            "cov": self.cov,
            "y_model": self.y_model.state_dict()
        }
    
    def load_params(self, param_dict):
        self.mean = param_dict["mean"]
        self.cov = param_dict["cov"]
        self.y_model.load_state_dict(param_dict["y_model"])

    def sample(self, n):
        x = np.random.multivariate_normal(self.mean, self.cov, n).astype(np.float32)
        y = self.y_model(ch.from_numpy(x).float())
        y = ch.argmax(y.detach(), dim=1).numpy()
        # y = y.reshape(y.shape[0], 1)
        print(np.mean(y))
        return x, y


class VicAdvDistribution:
    """
        Model victim's and adversary's views for a given distribution.
        Adversary's parameter estimates may be a bit off (domain shift), 
        and the labels might be too (will be added later).
    """

    def __init__(self, config: SyntheticDatasetConfig, mean, cov, y_model: GroundTruthModel):
        self.vic_distr = Distribution(mean, cov, y_model)

        # Create perturbed mean/cov for adv
        mean_adv = mean + np.random.rand(config.dimensionality) * config.adv_noise_to_mean
        cov_adv = np.diagflat(
            cov.diagonal() + np.random.rand(config.dimensionality) * config.noise_cov)

        # TODO: y_model variation different for the adversary
        self.adv_distr = Distribution(mean_adv, cov_adv, y_model)
    
    def get_params(self):
        return {
            "adv": self.adv_distr.get_params(),
            "victim": self.vic_distr.get_params()
        }
    
    def load_params(self, param_dict):
        self.adv_distr.load_params(param_dict["adv"])
        self.vic_distr.load_params(param_dict["victim"])

    def sample(self, n: int, split: Literal["victim", "adv"]):
        if split == "victim":
            return self.vic_distr.sample(n)
        elif split == "adv":
            return self.adv_distr.sample(n)
        else:
            raise ValueError("Required split not found")


class SyntheticDataset:
    """
        Main class for generating synthetic data from given distributions.
    """
    def __init__(self, config: SyntheticDatasetConfig, drop_senstive_cols: bool):
        self.config = config
        self.dimensionality = config.dimensionality
        self.adv_noise_to_mean = config.adv_noise_to_mean
        self.noise_cov = config.noise_cov
        self.noise_model = config.noise_model
        self.mean_range = config.mean_range
        self.layer = config.layer
        self.diff_posteriors = config.diff_posteriors
        self.dist_diff_mean = config.dist_diff_mean
        self.dist_diff_std = config.dist_diff_std
        self.drop_senstive_cols = drop_senstive_cols
        self.cov = config.cov
        self.n_samples_adv = config.n_samples_adv
        self.n_samples_vic = config.n_samples_vic

        # Start with some parameters for D0
        mean = np.random.uniform(-config.mean_range, config.mean_range, config.dimensionality)
        mean_copy = mean + np.random.uniform(-config.dist_diff_mean, config.dist_diff_mean, config.dimensionality)
        # set variance range between 1 and cov
        cov_list = np.random.uniform(1, config.cov, config.dimensionality)

        # Create D1 with some offset between D0 and D1
        cov = np.diagflat(cov_list)
        # not sure if can let it go negative
        cov_copy = np.diagflat(cov_list + np.random.uniform(0, config.dist_diff_mean))

        # Create y_model for both distrs
        model_D0 = GroundTruthModel(config.dimensionality, config.num_classes, config.layer)
        model_D1 = copy.deepcopy(model_D0)
        if not self.diff_posteriors:
            model_D1 = self.__get_perturbed_model_copy(model_D1)

        # Create D0 for vic/adv
        self.distr_0 = VicAdvDistribution(config, mean, cov, model_D0)
        # Create D1 for vic/adv
        self.distr_1 = VicAdvDistribution(
            config, mean_copy, cov_copy, model_D1)
    
    def save_params(self, path):
        """
            Save parameters related to current distribution in provided path
        """
        param_dict = {
            "d0": self.distr_0.get_params(),
            "d1": self.distr_1.get_params()
        }
        ch.save(param_dict, path)

    def load_params(self, path):
        """
            Load up exact distribution parameters from provided path
        """
        param_dict = ch.load(path)
        self.distr_0.load_params(param_dict["d0"])
        self.distr_1.load_params(param_dict["d1"])
        
    def __get_perturbed_model_copy(self, model):
        model_p = copy.deepcopy(model)
        with ch.no_grad():
            for param in model_p.parameters():
                param.add_(ch.randn(param.size()) * self.config.dist_diff_mean)
        return model_p

    def get_data(self, alpha: float, split: Literal["victim", "adv"], label_noise: float = 0.0):
        """
          Generate n_samples from the distribution (1-alpha)D0 + (alpha)D1.
          Achieves so by sampling (1-alpha) ratio of samples from D0, and 
          remaining from D1. Uses vic/adv split depending on 'split'
        """
        n_samples = self.n_samples_adv if split == "adv" else self.n_samples_vic
        n_samples_one = int(n_samples * alpha)
        n_samples_zero = n_samples - n_samples_one

        x_0, y_0 = self.distr_0.sample(n_samples_zero, split)  # D_0
        x_1, y_1 = self.distr_1.sample(n_samples_one, split)  # D_1

        x = np.concatenate((x_0, x_1))
        y = np.concatenate((y_0, y_1))
        p_labels = np.concatenate(
            (np.zeros(n_samples_zero), np.ones(n_samples_one)))
        
        # TODO: Implement label-noise later
        
        if not self.drop_senstive_cols:
            # Add p_labels as an extra dimension onto x
            x = np.concatenate((x, p_labels.reshape(-1, 1)), axis=1)
        
        x = ch.from_numpy(x)
        y = ch.from_numpy(y)
        p_labels = ch.from_numpy(p_labels)
        return x, y, p_labels


class SyntheticWrapper(base.CustomDatasetWrapper):
    def __init__(self,
                 data_config: DatasetConfig,
                 skip_data: bool = False,
                 epoch: bool = False,
                 label_noise: float = 0,
                 shuffle_defense: ShuffleDefense = None):
        super().__init__(data_config,
                         skip_data=skip_data,
                         label_noise=label_noise,
                         shuffle_defense=shuffle_defense)
        self.distribution_config = CONFIG_MAPPING.get(self.prop, None)
        if self.distribution_config is None:
            raise ValueError("Requested config not available")

        self.dimensionality = self.distribution_config.dimensionality
        self.n_classes = self.distribution_config.num_classes

        if not skip_data:
            self.ds = SyntheticDataset(self.distribution_config,
                                       drop_senstive_cols=self.drop_senstive_cols)
        self.info_object = DatasetInformation(epoch_wise=epoch)

    def load_data(self):
        x, y, p = self.ds.get_data(split=self.split,
                                alpha=self.ratio,
                                label_noise=self.label_noise)
        # Split into train and val data
        val_ratio = 0.2
        val_indices = np.random.choice(x.shape[0], int(x.shape[0] * val_ratio), replace=False)
        train_indices = np.array(list(set(range(x.shape[0])) - set(val_indices)))
        x_train, y_train, p_train = x[train_indices], y[train_indices], p[train_indices]
        x_val, y_val, p_val = x[val_indices], y[val_indices], p[val_indices]
        return (x_train, y_train, p_train), (x_val, y_val, p_val)

    def get_loaders(self,batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,):
        train_data, val_data = self.load_data()
        self.ds_train = TensorDataset(*train_data)
        self.ds_val = TensorDataset(*val_data)
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,)

    def load_model(self, path: str, on_cpu: bool = False, model_arch: str = None) -> nn.Module:
        info_object = self.info_object
        model = info_object.get_model(n_inp=self.dimensionality,
                                      n_classes=self.n_classes,
                                      cpu=on_cpu,
                                      model_arch=model_arch)
        return load_model(model, path, on_cpu=on_cpu)

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        base_models_dir = os.path.join(base_models_dir, self.prop)

        dp_config = None
        shuffle_defense_config = None
        if train_config.misc_config is not None:
            dp_config = train_config.misc_config.dp_config
            shuffle_defense_config = train_config.misc_config.shuffle_defense_config

        # Standard logic
        if model_arch == "None":
            model_arch = self.info_object.default_model
        if model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")

        base_models_dir = os.path.join(base_models_dir, model_arch)

        if dp_config is None:
            if shuffle_defense_config is None:
                base_path = os.path.join(base_models_dir, "normal")
            else:
                if self.ratio == shuffle_defense_config.desired_value:
                    # When ratio of models loaded is same as target ratio of defense,
                    # simply load 'normal' model of that ratio
                    base_path = os.path.join(base_models_dir, "normal")
                else:
                    base_path = os.path.join(base_models_dir, "shuffle_defense",
                                             "%s" % shuffle_defense_config.sample_type,
                                             "%.2f" % shuffle_defense_config.desired_value)
        else:
            base_path = os.path.join(
                base_models_dir, "DP_%.2f" % dp_config.epsilon)
        if self.label_noise:
            base_path = os.path.join(
                base_models_dir, "label_noise:{}".format(train_config.label_noise))

        save_path = os.path.join(base_path, self.split)

        if self.ratio is not None:
            save_path = os.path.join(save_path, str(self.ratio))

        if self.drop_senstive_cols:
            save_path = os.path.join(save_path, "drop")

        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path
