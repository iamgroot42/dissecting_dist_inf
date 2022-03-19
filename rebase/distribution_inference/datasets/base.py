import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

from distribution_inference import utils


# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Constants:
    splits = ["victim", "adv"]
    if utils.check_if_inside_cluster():
        base_data_directory = "/scratch/as9rw/datasets/"
    else:
        base_data_directory = "/p/adversarialml/as9rw/datasets/"


class CustomDataset(Dataset):
    def __init__(self, classify, prop, ratio, cwise_sample,
                 shuffle=False, transform=None):
        self.num_samples = None

    def __len__(self):
        """
            self.num_samples should be populated in
            init to compute the number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
            Should return (datum, attribute, class-label)
        """
        raise NotImplementedError("Dataset does not implement __getitem__")


class CustomDatasetWrapper:
    def __init__(self, prop, ratio, split,
                 classify, augment=False,
                 cwise_samples=None):
        """
            self.ds_train and self.ds_val should be set to
            datasets to be used to train and evaluate.
        """
        self.prop = prop
        self.ratio = ratio
        self.split = split
        self.classify = classify
        self.augment = augment
        self.cwise_samples = cwise_samples
        self.ds_train = None
        self.ds_val = None

    def get_loaders(self, batch_size, shuffle=True,
                    eval_shuffle=False, val_factor=1, num_workers=0,
                    prefetch_factor=2):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        test_loader = DataLoader(
            self.ds_val, batch_size=batch_size * val_factor,
            shuffle=eval_shuffle, num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor)

        return train_loader, test_loader


def generate_victim_adversary_splits(adv_ratio, test_ratio, num_tries):
    """
        Generate and store data offline for victim and adversary
        using the given dataset. Use this method only once for the
        same set of experiments.
    """
    raise NotImplementedError("Dataset does not have a method to generate victim and adversary splits")


def filter(df, condition, ratio, verbose=True):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            return pd.concat([df.iloc[qualify], df.iloc[nqi]])
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            return pd.concat([df.iloc[qi], df.iloc[notqualify]])
        return df.iloc[notqualify]


def heuristic(df, condition, ratio,
              cwise_sample,
              class_imbalance=2.0,
              n_tries=1000,
              class_col="label",
              verbose=True):
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        pckd_df = filter(df, condition, ratio, verbose=False)

        # Class-balanced sampling
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]

        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(
                    one_ids)[:cwise_sample]
            else:
                zero_ids = np.random.permutation(
                    zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(
                    one_ids)[:int(1 / class_imbalance * cwise_sample)]

        # Combine them together
        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
        pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)
