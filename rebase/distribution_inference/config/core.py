from dataclasses import dataclass
from lib2to3.pgen2.token import OP
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
class ShuffleDefenseConfig(Serializable):
    """
        Config to randomly sample during training
        to maintain a desired ratio.
        For now, implemented at batch-level.
    """
    desired_value: float
    """Desired ratio to be achieved when shuffling"""
    sample_type: str = field(choices=["over", "under"])
    """Over-sample or under-sample?"""
    data_level: bool = False
    """Perform under/over sampling at data level (true), or batch level (false)?"""
    augment: bool = False
    """Augment data during over-sampling?"""
    use_mixup: bool = False
    """Use mixup during augmentation?"""


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
    cwise_samples: Optional[int] = None
    """Number of samples for train & test to use (override existing values)"""
    squeeze: Optional[bool] = False
    """Whether to squeeze label data (because of extra dimension)"""
    processed_variant: Optional[bool] = True
    """Use processed version of data (relevant for BoneAge,CelebA)?"""
    prune: Optional[float] = 0
    """Prune graph by removing nodes? (only valid for arXiv dataset)"""


@dataclass
class MiscTrainConfig(Serializable):
    """
        Miscellaneous training configurations.
    """
    adv_config: Optional[AdvTrainingConfig] = None
    """Configuration to be used for adversarial training"""
    dp_config: Optional[DPTrainingConfig] = None
    """Configuration to be used for DP training"""
    shuffle_defense_config: Optional[ShuffleDefenseConfig] = None
    """Configuration to be usef for shuffle-based defense"""


@dataclass
class LRScheduler(Serializable):
    """
        Hyper-parameters for learning-rate scheduler
    """
    patience: Optional[int] = 5
    """Number of epochs to wait before reducing learning rate"""
    factor: Optional[float] = 0.1
    """Factor to reduce learning rate by"""
    cooldown: Optional[int] = 0
    """Number of epochs to wait before reducing learning rate"""
    mode: Optional[str] = "min"
    """Mode to use for learning rate scheduler"""
    verbose: Optional[bool] = True
    """Whether to print out per-classifier stats"""


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
    lr_scheduler: Optional[LRScheduler] = None
    """Use learning-rate scheduler?"""
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
    save_every_epoch: Optional[bool] = False
    """Save model after every epoch?"""
    extra_info: Optional[dict] = None
    """Optional dictionary to store misc information for dataset-specific args"""
    regression: Optional[bool] = False
    """Training for regression (MSE)?"""
    multi_class: Optional[bool] = False
    """Training for multi-class classification?"""
    label_noise: Optional[float] = 0
    """Randomly flip a proportion of labels"""
    model_arch: Optional[str] = None
    """Model architecture to use (defaults to dataset-specific)"""
    parallel: Optional[bool] = False
    """Use multiple GPUs for model training?"""

@dataclass
class GenerativeAttackConfig(Serializable):
    steps: int
    """Number of steps to run generation for"""
    step_size: float
    """Step size for generation"""
    latent_focus: int
    """Latent dimension to focus on"""
    model_ratio: float
    n_samples: Optional[int] = 500
    """Number of samples to generate"""
    use_normal: Optional[bool] = True
    start_natural: Optional[bool] = False
    """Whether to start with natural images"""
    constrained: Optional[bool] = False
    """Whether to use constrained generation"""
    use_best: Optional[bool] = False
    """Whether to use the best attack image, or the most recent one"""
    clamp: Optional[bool] = False


@dataclass
class BlackBoxAttackConfig(Serializable):
    """
        Configuration values for black-box attacks.
    """
    attack_type: List[str]
    """Which attacks to compute performance for"""
    ratios: Optional[List[float]] = field(default_factory=lambda: [1.0])
    """List of ratios (percentiles) to try"""
    batch_size: int = 256
    """Batch size to use for loaders when generating predictions"""
    num_adv_models: int = 50
    """Number of models adversary uses per distribution (for estimating statistics)"""
    granularity: Optional[float] = 0.005
    """Graunularity while finding threshold candidates"""
    preload: Optional[bool] = False
    """Pre-load data while launching attack (faster, if memory available)?"""
    multi: Optional[int] = None
    """Multi model setting (1), number of victim models"""
    multi2: Optional[int] = None
    """Multi model setting (2), number of victim models"""
    multi_class: Optional[bool] = False
    """Are the model logits > 1 dimension?"""
    save: Optional[bool] = False
    """Save predictions?"""
    tune_final_threshold: Optional[bool] = False
    """Tune final classification threshold, instead of a blind 0.5?"""

    Start_epoch: Optional[int] = 1
    "Start epoch to consider for single-update attack"
    End_epoch: Optional[int] = 20
    "End epoch to consider for single-update attack"
    
    relative_threshold: Optional[bool] = False
    """Thresholds are relative to mean accuracy/logits"""
    loss_variant: Optional[bool] = False
    """Where applicable (PPTT), ues loss values instead of logits"""
    random_order: Optional[bool] = False
    """Order points randomly instead of optimal ordering"""

    kl_frac: Optional[float] = 0.8
    """Frac of pairs to use (if KL test)"""
    kl_voting: Optional[bool] = False
    """Use comparison instead of differences"""
    generative_attack: Optional[GenerativeAttackConfig]=None
    """Use generative attack?"""
    order_name: Optional[str] = None
    """Type of ordering to use"""
    geo_mean:Optional[bool] = False


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
class AffinityAttackConfig(Serializable):
    """
        Configuration for affinity-based meta-classifier
    """
    num_final: int = 16
    """Number of activations in final mini-model (per layer)"""
    only_latent: bool = False
    """Ignore logits (output) layer"""
    random_edge_selection: bool = False
    """Use random selection of pairs (not trivial heuristic)"""
    frac_retain_pairs: float = 1.0
    """What fraction of pairs to use when training classifier"""
    better_retain_pair: bool = False
    """Compute STD across different models, instead of over all models"""
    optimal_data_identity: bool = False
    """Heuristic to identify most useful samples"""
    model_sample_for_optimal_data_identity: int = 50
    """Number of models to sample to identify optimal points"""
    num_samples_use: int = None
    """How many examples to compute pair-wise similarities for"""
    layer_agnostic: Optional[bool] = False
    """Whether to use layer-agnostic version of meta-classifier"""
    inner_dims: Optional[List[int]] = field(default_factory=lambda: [1024, 64])
    """Dimensions of inner layers"""
    shared_layerwise_params: Optional[bool] = False
    """Use same layer-wise model for all layers?"""
    sequential_variant: Optional[bool] = False
    """Use sequential model instead of linear layer on concatenated embeddings?"""
    num_rnn_layers: Optional[int] = 2
    """Number of layers in RNN"""
    layers_to_target_conv: Optional[List[int]] = None
    """Which conv layers of the model to target while extracting features?"""
    layers_to_target_fc: Optional[List[int]] = None
    """Which conv layers of the model to target while extracting features?"""
    perpoint_based_selection: Optional[int] = 0
    """If > 0, use same selection logic used by Per-Point Threshold Test with these many models"""
    flip_selection_logic: Optional[bool] = False
    """Flip ordering generated by heuristic (for flag-activated heuristics)?"""


@dataclass
class RegressionConfig(Serializable):
    """
        Configuration for regression-based attacks
    """
    additional_values_to_test: Optional[List] = None
    """Values of property to use while testing in addition to ratios used to train"""


@dataclass
class ComparisonAttackConfig(Serializable):
    Start_epoch: int
    """Epoch to use for 'before'"""
    End_epoch: int
    """Epoch to use for 'after'"""
    num_models: int
    """Number of models to use for attack"""

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
    train_sample: Optional[int] = 750
    """Number of models to train meta-classifiers on (per run)"""
    val_sample: Optional[int] = 50
    """Number of models to validate meta-classifiers on (per run)"""
    save: Optional[bool] = False
    """Save meta-classifiers?"""
    load: Optional[str] = None
    """Path to load meta-classifiers from"""
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
    multi_class: Optional[bool] = False
    """Are the model logits > 1 dimension?"""

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
    affinity_config: Optional[AffinityAttackConfig] = None
    """Configuration for affinity-based attacks"""
    comparison_config: Optional[ComparisonAttackConfig] = None

@dataclass
class FairnessEvalConfig(Serializable):
    """
        Configuration values for fairness-based model evaluation
    """
    train_config: TrainConfig
    """Configuration used when training models"""
    prop: str
    """Which attribute to use as 'group' in analyses"""
    value_min: float = 0.0
    """Distribution from which data is drawn for attr=0"""
    value_max: float = 1.0
    """Distribution from which data is drawn for attr=1"""
    batch_size: int = 256
    """Batch size to use for loaders when generating predictions"""
    num_models: Optional[int] = 250
    """Number of victim models (per distribution) to test on"""
    on_cpu: Optional[bool] = False
    """Keep models read on CPU?"""
    preload: Optional[bool] = False
    """Pre-load data while launching attack (faster, if memory available)?"""

@dataclass
class AttackConfig(Serializable):
    """
        Configuration values for attacks in general.
    """
    train_config: TrainConfig
    """Configuration used when training models"""
    values: List
    """List of values (on property specified) to launch attack against. In regression, this the list of values to train on"""
    black_box:  Optional[BlackBoxAttackConfig] = None
    """Configuration for black-box attacks"""
    white_box: Optional[WhiteBoxAttackConfig] = None
    """Configuration for white-box attacks"""
    tries: int = 1
    """Number of times to try each attack experiment"""
    num_victim_models: Optional[int] = 1000
    """Number of victim models (per distribution) to test on"""
    on_cpu: Optional[bool] = False
    """Keep models read on CPU?"""
    adv_misc_config: Optional[MiscTrainConfig] = None
    """If given, specifies extra training params (adv, DP, etc) for adv models"""
    num_total_adv_models: Optional[int] = 1000
    """Total number of adversarial models to load"""
    victim_local_attack: Optional[bool] = False
    """Perform attack as if victim is using its own data/models"""
    victim_model_arch: str = None
    """Architecture of victim model (defaults to dataset-specific model)"""
    adv_model_arch: str = None
    """Architecture for adversary model (defaults to dataset-specific model)"""
    adv_processed_variant: Optional[bool] = False
    """Use processed variant for adv data?"""

    adv_target_epoch: Optional[int] = None
    """Which epoch to target for adversary. If not None, automatically use last epoch"""
    victim_target_epoch: Optional[int] = None
    """Which epoch to target for victim. If not None, automatically use last epoch"""


    
@dataclass
class UnlearningConfig(Serializable):
    """
        Configuration values for Property Unlerning.
        https://arxiv.org/pdf/2205.08821.pdf
    """
    learning_rate: float
    """LR for optimizer"""
    stop_tol: float
    """Delta in prediction differences that should be achieved to terminate"""
    flip_weight_ratio: float = 0.002
    """Ratio of weights to randomly flip when perfect predictions appear"""
    max_iters: int = 500
    """Maximum number of iterations to run"""
    k: int = 2
    """Number of classes"""
    flip_tol: float = 1e-3
    """Tolerance for checking with equality"""
    min_lr: float = 1e-5
    """Minimum learning rate"""


@dataclass
class DefenseConfig(Serializable):
    """
        Configuration file for defense
    """
    train_config: TrainConfig
    """Train config used to train victim/adv models"""
    wb_config: WhiteBoxAttackConfig
    """Configuration used for adversary"""
    values: List
    """List of values (on property specified)"""
    num_models: int
    """Number of victim models to implement defense for"""
    victim_local_attack: bool
    """Load meta-classifiers corresponding to victim_local setting"""
    on_cpu: Optional[bool] = False
    """Keep models read on CPU?"""
    unlearning_config: Optional[UnlearningConfig] = None
    """Configuration for unlearning"""


@dataclass
class CombineAttackConfig(AttackConfig):
    """
        Configuration for decision tree attack combining whitebox and blackbox
    """
    wb_path: Optional[str] = None
    """path for metaclassifier"""
    save_bb: Optional[bool] = False
    """save bb results independently"""
    save_data: Optional[bool] = True
    """save model, data point, and predictions"""
    restore_data: Optional[str] = None
    """path to restore all the data"""
    use_wb_latents: Optional[bool] = False
    """Use feature outputs from WB models (instead of logits)?"""
    num_for_meta: Optional[int] = 50
    """Number of data points to use for meta-classifier on top of WB and BB"""
