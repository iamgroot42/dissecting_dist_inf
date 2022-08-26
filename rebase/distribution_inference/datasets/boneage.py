import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import pandas as pd
import torch as ch
import torch.nn as nn
import gc
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import densenet121
from torch.utils.data import DataLoader

import distribution_inference.datasets.base as base
from distribution_inference.models.core import BoneModel, DenseNet, SVMClassifier
from distribution_inference.config import DatasetConfig, TrainConfig
from distribution_inference.utils import ensure_dir_exists
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model


def get_transforms(augment):
    """
        Return train and test transforms
    """
    input_size = 224
    test_transform_list = [
        transforms.Resize((input_size, input_size)),
    ]
    train_transform_list = test_transform_list[:]

    post_transform_list = [transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
    # Add image augmentations if requested
    if augment:
        train_transform_list += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(
                shear=0.01, translate=(0.15, 0.15), degrees=5)
        ]

    train_transform = transforms.Compose(
        train_transform_list + post_transform_list)
    test_transform = transforms.Compose(
        test_transform_list + post_transform_list)

    return train_transform, test_transform


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        super().__init__(name="RSNA-Boneage",
                         data_path="rsnabone/data",
                         models_path="models_boneage",
                         properties=["gender", "age"],
                         values={"gender": ratios, "age": ratios},
                         supported_models=["densenet", "bonemodel", "svm"],
                         default_model="bonemodel",
                         epoch_wise=epoch_wise)
        self.supported_properties = ["gender", "age"]

    def get_model(self,
                  cpu: bool = False,
                  model_arch: str = None) -> nn.Module:
        if model_arch is None:
            model_arch = self.default_model

        if model_arch == "densenet":
            model = DenseNet(1024)
        elif model_arch == "bonemodel":
            model = BoneModel(1024)
        elif model_arch == "svm":
            model = SVMClassifier()
        else:
            raise NotImplementedError("Model architecture not supported")

        if cpu:
            model = model.cpu()
        else:
            model = model.cuda()
        return model

    def _stratified_df_split(df, second_ratio: float):
        # Get new column for stratification purposes
        def fn(row): return str(row.gender) + str(row.age)
        col = df.apply(fn, axis=1)
        df = df.assign(stratify=col.values)

        stratify = df['stratify']
        df_1, df_2 = train_test_split(
            df, test_size=second_ratio,
            stratify=stratify)

        # Delete remporary stratification column
        df.drop(columns=['stratify'], inplace=True)
        df_1 = df_1.drop(columns=['stratify'])
        df_2 = df_2.drop(columns=['stratify'])

        return df_1.reset_index(), df_2.reset_index()

    def _process_data(self, path, split_second_ratio: float):
        df = pd.read_csv(os.path.join(
            path, 'boneage-training-dataset.csv'))

        df['path'] = df['id'].map(
            lambda x: os.path.join(
                path,
                'boneage-training-dataset',
                'boneage-training-dataset',
                '{}.png'.format(x)))

        df['gender'] = df['male'].map(lambda x: 0 if x else 1)

        # Binarize into bone-age <=132 and >132 (roughly-balanced split)
        df['age'] = df['boneage'].map(lambda x: 1 * (x > 132))
        df.dropna(inplace=True)

        # Drop temporary columns
        df.drop(columns=['male', 'id'], inplace=True)

        # Return stratified split
        return self._stratified_df_split(df, split_second_ratio)

    def _get_pre_processor(self):
        # Load model
        model = densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Get rid of existing classification layer
        # Extract only features
        model.classifier = nn.Identity()
        return model


    def _get_df(self, split: str):
        df_train = pd.read_csv(os.path.join(
            self.base_data_dir, "%s/train.csv" % split))
        df_val = pd.read_csv(os.path.join(
            self.base_data_dir, "%s/val.csv" % split))

        return df_train, df_val

    def _extract_pretrained_features(self, df, split: str):
        # Load model
        model = self._get_pre_processor()
        model = model.cuda()
        model = nn.DataParallel(model)

        # Ready dataset objects
        train_df, val_df = df
        wrapper = _RawBoneWrapper(train_df, val_df)

        # Ready loaders
        batch_size = 100
        train_loader, val_loader = wrapper.get_loaders(
            batch_size, shuffle=False)

        # Collect features
        print("Collecting train-data features")
        train_features = self._collect_features(train_loader, model)
        print("Collecting val-data features")
        val_features = self._collect_features(val_loader, model)

        # Create mapping between filepaths and features for those images
        train_map = {train_df['path'][i]: train_features[i]
                     for i in range(len(train_df))}

        val_map = {val_df['path'][i]: val_features[i]
                   for i in range(len(val_df))}

        # Save features
        ft_path = os.path.join(self.base_data_dir, split, "features_train.pt")
        fv_path = os.path.join(
            self.base_data_dir, split, "features_val.pt")
        ch.save(train_map, ft_path)
        ch.save(val_map, fv_path)

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = 0.33,
                                         test_ratio: float = 0.2,
                                         num_tries: int = None):
        base_path = os.path.abspath(
            os.path.join(self.base_data_dir, os.pardir))
        df_victim, df_adv = self._process_data(
            base_path, split_second_ratio=adv_ratio)

        def useful_stats(df):
            print("Size: %d | Age ratio: %.2f | Gender ratio: %.2f" % (
                len(df),
                df["age"].mean(),
                df["gender"].mean()))

        # Save these splits
        def save_split(df, split):
            useful_stats(df)
            print()

            # Get train-val splits
            train_df, val_df = self._stratified_df_split(df, test_ratio)

            # Ensure directory exists
            dir_prefix = os.path.join(self.base_data_dir, "data", split)
            ensure_dir_exists(dir_prefix)

            # Save train-test splits
            train_df.to_csv(os.path.join(dir_prefix, "train.csv"))
            val_df.to_csv(os.path.join(dir_prefix, "val.csv"))

        # Generate DF files for vic, adv
        save_split(df_victim, "victim")
        save_split(df_adv, "adv")

        # Extract pre-trained features for splits
        self._extract_pretrained_features(df_victim, "victim")
        self._extract_pretrained_features(df_adv, "adv")


class BoneDataset(base.CustomDataset):
    def __init__(self, df,
                 argument = None,
                 processed: bool = False,
                 classify: str = "age",
                 property: str = "gender"):
        super().__init__()
        if processed:
            self.features = argument
        else:
            self.transform = argument
        self.processed = processed
        self.df = df
        self.num_samples = len(self.df)
        self.classify = classify
        self.property = property
    
    def mask_data_selection(self, mask):
        self.mask = mask
        self.num_samples = len(self.mask)

    def __getitem__(self, index):
        index_ = self.mask[index].item() if self.mask is not None else index

        if self.processed:
            X = self.features[self.df['path'].iloc[index_]]
        else:
            # X = Image.open(self.df['path'][index_])
            X = Image.open(self.df['path'][index_]).convert('RGB')
            if self.transform:
                X = self.transform(X)

        y = ch.tensor(int(self.df[self.classify][index_]))
        prop_label = ch.tensor(int(self.df[self.property][index_]))

        return X, y, prop_label


class _RawBoneWrapper:
    def __init__(self, df_train, df_val, augment=False):
        self.df_train = df_train
        self.df_val = df_val

        # Get transforms
        train_transform, test_transform = get_transforms(augment)

        self.ds_train = BoneDataset(self.df_train, augment=train_transform)
        self.ds_val = BoneDataset(self.df_val, augment=test_transform)

    def get_loaders(self, batch_size, shuffle=False):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)
        val_loader = DataLoader(
            self.ds_val, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)

        return train_loader, val_loader


class BoneWrapper(base.CustomDatasetWrapper):
    def __init__(self,
                 data_config: DatasetConfig,
                 skip_data: bool = False,
                 epoch: bool = False,
                 label_noise: float = 0,
                 shuffle_defense: ShuffleDefense = None):
        # Call parent constructor
        super().__init__(data_config, skip_data, label_noise, shuffle_defense=shuffle_defense)
        self.sample_sizes = {
            "gender": {
                "adv": (700, 200),
                "victim": (1400, 400)
            },
            "age": {
                "adv": (700, 200),
                "victim": (1400, 400)
            }
        }
        # Define DI object
        self.info_object = DatasetInformation(epoch_wise=epoch)

    def _filter(self, x):
        return x[self.prop] == 1

    def load_model(self, path: str,
                   on_cpu: bool = False,
                   model_arch: str = None) -> nn.Module:
        model = self.info_object.get_model(cpu=on_cpu, model_arch=model_arch)
        return load_model(model, path, on_cpu=on_cpu)

    def prepare_processed_data(self, loader):
        # Load model
        model = self.info_object._get_pre_processor()
        model = model.cuda()
        # Collect all processed information
        features, task_labels, prop_labels = self.info_object._collect_features(
            loader, model, collect_all_info=True)
        self.ds_val_processed = ch.utils.data.TensorDataset(features,
            task_labels, prop_labels)
        # Clear cache
        del model
        gc.collect()
        ch.cuda.empty_cache()
        return features.shape[1:]

    def load_data(self):
        # Get DF files for train, val
        df_train, df_val = self.info_object._get_df(self.split)
        # Filter to achieve desired ratios and sample-sizes
        if self.cwise_samples is None:
            n_train, n_test = self.sample_sizes[self.prop][self.split]
        else:
            n_train, n_test = self.cwise_samples, self.cwise_samples

        self.df_train = utils.heuristic(
            df_train, self._filter, self.ratio,
            n_train, class_imbalance=1.0,
            class_col=self.classify,
            n_tries=300)
        self.df_val = utils.heuristic(
            df_val, self._filter, self.ratio,
            n_test, class_imbalance=1.0,
            class_col=self.classify,
            n_tries=300)

        # Create datasets using these DF objects
        if self.processed_variant:
            features = self._get_features()
            ds_train = BoneDataset(
                self.df_train, features["train"],
                processed=True,
                classify=self.classify,
                property=self.prop)
            ds_val = BoneDataset(
                self.df_val, features["val"],
                processed=True,
                classify=self.classify,
                property=self.prop)
        else:
            # Get transforms
            train_transform, test_transform = get_transforms(self.augment)

            ds_train = BoneDataset(
                self.df_train, train_transform,
                processed=False,
                classify=self.classify,
                property=self.prop)
            ds_val = BoneDataset(
                self.df_val, test_transform,
                processed=False,
                classify=self.classify,
                property=self.prop)
        return ds_train, ds_val

    def get_loaders(self, batch_size,
                    shuffle: bool = False,
                    eval_shuffle: bool = False,
                    num_workers: int = 2,
                    prefetch_factor: int = 2):
        self.ds_train, self.ds_val = self.load_data()
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch_factor)

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        shuffle_defense_config = None
        if train_config.misc_config is not None:
            shuffle_defense_config = train_config.misc_config.shuffle_defense_config
        subfolder_prefix = os.path.join(self.split, self.prop, str(self.ratio))

        # Standard logic
        if model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")
        if model_arch is None:
            model_arch = self.info_object.default_model
        base_models_dir = os.path.join(base_models_dir, model_arch)

        if shuffle_defense_config is None or shuffle_defense_config.desired_value == self.ratio:
            base_models_dir = os.path.join(base_models_dir, "normal")
        else:
            if self.ratio == shuffle_defense_config.desired_value:
                # When ratio of models loaded is same as target ratio of defense,
                # simply load 'normal' model of that ratio
                base_models_dir = os.path.join(base_models_dir, "normal")
            else:
                base_models_dir = os.path.join(base_models_dir, "shuffle_defense",
                                            "%s" % shuffle_defense_config.sample_type,
                                            "%.2f" % shuffle_defense_config.desired_value)

        # Get final savepath
        save_path = os.path.join(base_models_dir, subfolder_prefix)

        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        return save_path

    def _get_features(self):
        # Load features
        features = {}
        features["train"] = ch.load(os.path.join(
            self.info_object.base_data_dir,
            "%s/features_train.pt" % self.split),
            map_location="cpu")
        features["val"] = ch.load(os.path.join(
            self.info_object.base_data_dir,
            "%s/features_val.pt" % self.split),
            map_location="cpu")

        return features
