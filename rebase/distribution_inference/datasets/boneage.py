# TODO- Complete this implementation

import os
import pandas as pd
import torch as ch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

import distribution_inference.datasets.base as base
from distribution_inference.models.core import BoneModel, DenseNet
from distribution_inference.config import DatasetConfig
from distribution_inference.utils import ensure_dir_exists

from torch.utils.data import Dataset, DataLoader


class DatasetInformation(base.DatasetInformation):
    def __init__(self):
        ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        super().__init__(name="RSNA-Boneage",
                         data_path="rsnabone/data",
                         models_path="models_boneage",
                         properties=["gender"],
                         values={"gender": ratios})
        self.supported_properties = ["gender"]
    
    def get_model(self, cpu: bool = False, full_model: bool = False) -> nn.Module:
        if full_model:
            model = DenseNet(1024)
        else:
            model = BoneModel(1024)
        if not cpu:
            model = model.cuda()
        return model
    
    def _stratified_df_split(df, second_ratio):
        # Get new column for stratification purposes
        def fn(row): return str(row.gender) + str(row.label)
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
        df['label'] = df['boneage'].map(lambda x: 1 * (x > 132))
        df.dropna(inplace=True)

        # Drop temporary columns
        df.drop(columns=['male', 'id'], inplace=True)

        # Return stratified split
        return self._stratified_df_split(df, split_second_ratio)

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = 0.33,
                                         test_ratio: float = 0.2,
                                         num_tries: int = None):
        base_path = os.path.abspath(os.path.join(self.base_data_dir, os.pardir))
        df_victim, df_adv = self._process_data(base_path, split_second_ratio=adv_ratio)

        def useful_stats(df):
            print("%d | %.2f | %.2f" % (
                len(df),
                df["label"].mean(),
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

        save_split(df_victim, "victim")
        save_split(df_adv, "adv")


class BoneDataset(base.CustomDataset):
    def __init__(self, ratio, transform=None, processed=False):
        # Call parent constructor
        super().__init__(classify=None, prop=None,
                         ratio, cwise_sample=None,
                         transform=transform)
        self.processed = processed
        self.num_samples = len(self.df)

    def __init__(self, df, argument=None, processed=False):
        if processed:
            self.features = argument
        else:
            self.transform = argument
        self.processed = processed
        self.df = df

    def __getitem__(self, index):
        if self.processed:
            X = self.features[self.df['path'].iloc[index]]
        else:
            # X = Image.open(self.df['path'][index])
            X = Image.open(self.df['path'][index]).convert('RGB')
            if self.transform:
                X = self.transform(X)

        y = ch.tensor(int(self.df['label'][index]))
        gender = ch.tensor(int(self.df['gender'][index]))

        return X, y, (gender)


class BoneWrapper(base.CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig, skip_data: bool = False):
        # Call parent constructor
        super().__init__(data_config, skip_data)
        self.input_size = 224
        test_transform_list = [
            transforms.Resize((self.input_size, self.input_size)),
        ]
        train_transform_list = test_transform_list[:]

        post_transform_list = [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
        # Add image augmentations if requested
        if self.augment:
            train_transform_list += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(
                    shear=0.01, translate=(0.15, 0.15), degrees=5)
            ]
        
        self.train_transform = transforms.Compose(
            train_transform_list + post_transform_list)
        self.test_transform = transforms.Compose(
            test_transform_list + post_transform_list)
        
    def get_loaders(self, batch_size, shuffle=False,
                val_factor=2, num_workers=2, prefetch_factor=2):
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=shuffle,
                                   val_factor=val_factor,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch_factor)
class ExcusiWhat:
    def __init__(self, df_train, df_val, features=None, augment=False):
        self.df_train = df_train
        self.df_val = df_val
        self.input_size = 224
        test_transform_list = [
            transforms.Resize((self.input_size, self.input_size)),
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

        if features is None:
            self.ds_train = BoneDataset(self.df_train, train_transform)
            self.ds_val = BoneDataset(self.df_val, test_transform)
        else:
            self.ds_train = BoneDataset(
                self.df_train, features["train"], processed=True)
            self.ds_val = BoneDataset(
                self.df_val, features["val"], processed=True)


# Get DF file
def get_df(split):
    if split not in ["victim", "adv"]:
        raise ValueError("Invalid split specified!")

    df_train = pd.read_csv(os.path.join(BASE_DATA_DIR, "%s/train.csv" % split))
    df_val = pd.read_csv(os.path.join(BASE_DATA_DIR, "%s/val.csv" % split))

    return df_train, df_val


# Load features file
def get_features(split):
    if split not in ["victim", "adv"]:
        raise ValueError("Invalid split specified!")

    # Load features
    features = {}
    features["train"] = ch.load(os.path.join(
        BASE_DATA_DIR, "%s/features_train.pt" % split))
    features["val"] = ch.load(os.path.join(
        BASE_DATA_DIR, "%s/features_val.pt" % split))

    return features
