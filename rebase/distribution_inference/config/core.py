from dataclasses import dataclass
from typing import Optional, List
from simple_parsing.helpers import Serializable, field


@dataclass
class DPTraining(Serializable):
    """
        Hyper-parameters for DP training.
    """
    epsilon: float
    """Epsilon (privacy budget) to use when training"""
    delta: float
    """Delta value (probability of leakage) while DP training. Should be less than inverse of train dataset"""
    physical_batch_size: int
    """Physical batch size (scales in square in memory) when training"""


@dataclass
class AdvTraining(Serializable):
    """
        Hyper-parameters for adversarial training.
    """
    epsilon: float
    """Bound on total perturbation norm"""
    epsilon_iter: float
    """Bound on perturbation per iteration"""
    iters: int
    """Number of iterations to run PGD for"""


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
    dp_config: Optional[DPTraining] = None
    """Configuration to be used for training with DP"""
    adv_config: Optional[AdvTraining] = None
    """Configuration to be used for adversarial training"""
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
    adv_train: Optional[AdvTraining] = None
    """Config for adversarial training"""
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
class WhiteBoxAttackConfig(Serializable):
    """
        Configuration values for white-box attacks.
    """
    # Valid for training
    epochs: int
    """Number of epochs to train meta-classifiers for"""
    batch_size: int
    """Batch size for training meta-classifiers"""
    learning_rate: Optional[float] = 1e-3
    """Learning rate for meta-classifiers"""
    train_sample: Optional[int] = 800
    """Number of models to train meta-classifiers on (per run)"""
    val_sample: Optional[int] = 0
    """Number of models to validate meta-classifiers on (per run)"""
    save: Optional[bool] = False
    """Save meta-classifiers?"""

    # Valid only for MLPs
    start_n: Optional[int] = 0
    """Layer index to start with (while extracting parameters)"""
    first_n: Optional[int] = None
    """Layer index (from start) until which to extract parameters"""

    # Valid only for CNNs
    start_n_conv: Optional[int] = 0
    """Layer index to start with (while extracting parameters) for conv layers"""
    first_n_conv: Optional[int] = None
    """Layer index (from start) until which to extract parameters for conv layers"""
    start_n_fc: Optional[int] = 0
    """Layer index to start with (while extracting parameters) for fc layers"""
    first_n_fc: Optional[int] = None
    """Layer index (from start) until which to extract parameters for fc layers"""

    # Loading models other than those trained with standard training
    use_adv_for_adv: Optional[bool] = False
    """Whether to use adversarially-trained models for training meta-classifier(s)"""
    use_adv_for_victim: Optional[bool] = False
    """Whether to use adversarially-trained models for training victim model(s)"""

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
    black_box: BlackBoxAttackConfig
    """Configuration for black-box attacks"""
    value: List
    """List of values (on property specified) to launch attack against"""
    tries: int = 1
    """Number of times to try each attack experiment"""
    white_box: Optional[WhiteBoxAttackConfig] = None
    """Configuration for white-box attacks"""
    num_victim_models: Optional[int] = 1000
    """Number of victim models (per distribution) to test on"""
    on_cpu: Optional[bool] = False
    """Keep models read on CPU?"""
