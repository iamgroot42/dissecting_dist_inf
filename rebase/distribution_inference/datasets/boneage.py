# TODO- Complete this implementation

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

import distribution_inference.datasets.base as base




import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch as ch
import os
import utils



class BoneDataset(base.CustomDataset):
    def __init__(self, ratio, transform=None, processed=False):
        # Call parent constructor
        super().__init__(classify=None, prop=None, ratio, cwise_sample=None, transform=transform)
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
    def __init__(self, classify,
                 prop, ratio, cwise_sample,
                 shuffle=False, transform=None):
        # Call parent constructor
        super().__init__(classify, prop, ratio, cwise_sample,
                 shuffle=shuffle, transform=transform)
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


def stratified_df_split(df, second_ratio):
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


def process_data(path, split_second_ratio=0.5):
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
    return stratified_df_split(df, split_second_ratio)


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


def generate_victim_adversary_splits(adv_ratio=0.33, test_ratio=0.2, num_tries=None):
    """
        Generate and store data offline for victim and adversary
        using the given dataset. Use this method only once for the
        same set of experiments.
    """
    base = os.path.abspath(os.path.join(BASE_DATA_DIR, os.pardir))
    df_victim, df_adv = process_data(base, split_second_ratio=adv_ratio)

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
        train_df, val_df = stratified_df_split(df, test_ratio)

        # Ensure directory exists
        dir_prefix = os.path.join(BASE_DATA_DIR, "data", split)
        utils.ensure_dir_exists(dir_prefix)

        # Save train-test splits
        train_df.to_csv(os.path.join(dir_prefix, "train.csv"))
        val_df.to_csv(os.path.join(dir_prefix, "val.csv"))

    save_split(df_victim, "victim")
    save_split(df_adv, "adv")
