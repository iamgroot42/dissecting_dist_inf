from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from simple_parsing.helpers import Serializable, field


@dataclass
class AdvTrainingConfig(Serializable):
    """
        Hyper-parameters for adversarial training.
    """
    epsilon: float
    """Bound on total perturbation norm"""
    iters: int
    """Number of iterations to run PGD for"""
    epsilon_iter: Optional[float] = None
    """Bound on perturbation per iteration"""
    clip_min: float = None
    """Minimum value to clip to"""
    clip_max: float = None
    """Maximum value to clip to"""
    random_restarts: int = 1
    """Number of random restarts to run PGD for"""
    norm: float = np.inf
    """Norm for perturbation budget"""
    scale_by_255: bool = False
    """Scale given epsilon by 255?"""


@dataclass
class DPTrainingConfig(Serializable):
    """
        Hyper-parameters for DP training.
    """
    epsilon: float
    """Epsilon (privacy budget) to use when training"""
    delta: float
    """Delta value (probability of leakage) while DP training. Should be less than inverse of train dataset"""
    physical_batch_size: int
    """Physical batch size (scales in square in memory) when training"""
    max_grad_norm: float
    """Maximum gradient norm to clip to"""


@dataclass
class DatasetConfig(Serializable):
    """
        Dataset-specific configuration values.
    """
    name: str
    """Name of dataset"""
    prop: str
    """Property to filter on"""
    value: float
    """Value of property to filter on"""
    split: str = field(choices=["victim", "adv"])
    """Split of dataset to use (victim or adv)"""
    drop_senstive_cols: Optional[bool] = False
    """Whether to drop sensitive columns"""
    scale: Optional[float] = 1.0
    """Scale subsample size by this value"""
    augment: Optional[bool] = False
    """Use data augmentation?"""
    classify: Optional[str] = None
    """Which classification task to use"""
    cwise_samples: Optional[dict] = None
    """Mapping between property/task/split and number of samples"""
    squeeze: Optional[bool] = False
    """Whether to squeeze label data (because of extra dimension)"""


@dataclass
class MiscTrainConfig(Serializable):
    adv_config: Optional[AdvTrainingConfig] = None
    """Configuration to be used for adversarial training"""
    dp_config: Optional[DPTrainingConfig] = None
    """Configuration to be used for DP training"""


@dataclass
class TrainConfig(Serializable):
    """
        Configuration values for training models.
    """
    data_config: DatasetConfig
    """Configuration for dataset"""
    epochs: int
    """Number of epochs to train for"""
    learning_rate: float
    """Learning rate for optimizer"""
    batch_size: int
    """Batch size for training"""

    misc_config: Optional[MiscTrainConfig] = None
    """Extra configuration for model training env"""

    verbose: Optional[bool] = False
    """Whether to print out per-classifier stats"""
    num_models: int = 1
    """Number of models to train"""
    offset: Optional[int] = 0
    """Offset to start counting from when saving models"""
    weight_decay: Optional[float] = 0
    """L2 regularization weight"""
    get_best: Optional[bool] = True
    """Whether to get the best performing model (based on validation data)"""
    cpu: Optional[bool] = False
    """Whether to train on CPU or GPU"""
    expect_extra: Optional[bool] = True
    """Expect dataloaders to have 3-value tuples instead of two"""
    use_best: Optional[bool] = True
    """Use model with best validation loss"""


@dataclass
class BlackBoxAttackConfig(Serializable):
    """
        Configuration values for black-box attacks.
    """
    attack_type: List[str]
    """Which attacks to compute performance for"""
    ratios: Optional[List[float]] = field(default_factory=lambda: [1.0])
    """List of ratios (percentiles) to try"""
    granularity: float = 0.005
    """Graunularity while finding threshold candidates"""
    batch_size: int = 256
    """Batch size to use for loaders when generating predictions"""
    num_adv_models: int = 50
    """Number of models adversary uses per distribution (for estimating statistics)"""


@dataclass
class PermutationAttackConfig(Serializable):
    """
        Configuration values for permutation-invariant networks
    """
    focus: Optional[str] = "all"
    """Which kind of meta-classifier to use"""
    scale_invariance: Optional[bool] = False
    """Whether to use scale-invariant meta-classifier"""


@dataclass
class RegressionConfig(Serializable):
    """
        Configuration for regression-based attacks
    """
    values_to_train: List
    """Values of property to use while training"""


@dataclass
class WhiteBoxAttackConfig(Serializable):
    """
        Configuration values for white-box attacks.
    """
    attack: str
    """Which attack to use"""

    # Valid for training
    epochs: int
    """Number of epochs to train meta-classifiers for"""
    batch_size: int
    """Batch size for training meta-classifiers"""
    learning_rate: Optional[float] = 1e-3
    """Learning rate for meta-classifiers"""
    weight_decay: Optional[float] = 0.01
    """Weight-decay while training meta-classifiers"""
    train_sample: Optional[int] = 800
    """Number of models to train meta-classifiers on (per run)"""
    val_sample: Optional[int] = 0
    """Number of models to validate meta-classifiers on (per run)"""
    save: Optional[bool] = False
    """Save meta-classifiers?"""
    regression_config: Optional[RegressionConfig] = None
    """Whether to use regression meta-classifier"""
    eval_every: Optional[int] = 10
    """Print evaluation metrics on test data every X epochs"""
    binary: Optional[bool] = True
    """Use BCE loss with binary classification"""
    gpu: Optional[bool] = True
    """Whether to train on GPU or CPU"""
    shuffle: Optional[bool] = True
    """Shuffle train data in each epoch?"""

    # Valid for MLPs
    custom_layers_fc: Optional[List[int]] = None
    """Indices of layers to extract features from (in specific) for FC"""
    start_n_fc: Optional[int] = 0
    """Layer index to start with (while extracting parameters) for FC"""
    first_n_fc: Optional[int] = None
    """Layer index (from start) until which to extract parameters for FC"""

    # Valid only for CNNs
    custom_layers_conv: Optional[List[int]] = None
    """Indices of layers to extract features from (in specific) for FC"""
    start_n_conv: Optional[int] = 0
    """Layer index to start with (while extracting parameters) for conv layers"""
    first_n_conv: Optional[int] = None
    """Layer index (from start) until which to extract parameters for conv layers"""

    # Valid for specific attacks
    permutation_config: Optional[PermutationAttackConfig] = None
    """Configuration for permutation-invariant attacks"""


@dataclass
class AttackConfig(Serializable):
    """
        Configuration values for attacks in general.
    """
    train_config: TrainConfig
    """Configuration used when training models"""
    values: List
    """List of values (on property specified) to launch attack against"""
    white_box: Optional[WhiteBoxAttackConfig] = None
    """Configuration for white-box attacks"""
    black_box: Optional[BlackBoxAttackConfig] = None
    """Configuration for black-box attacks"""
    tries: int = 1
    """Number of times to try each attack experiment"""
    num_victim_models: Optional[int] = 1000
    """Number of victim models (per distribution) to test on"""
    on_cpu: Optional[bool] = False
    """Keep models read on CPU?"""
    adv_diff_misc_config: Optional[bool] = False
    """If true, indicates adv models having different mist training config"""
    adv_misc_config: Optional[MiscTrainConfig] = None
    """If given, specifies extra training params (adv, DP, etc) for adv models"""
    num_total_adv_models: Optional[int] = 1000
    """Total number of adversarial models to load"""
