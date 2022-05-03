import numpy as np
from tqdm import tqdm
import pandas as pd
import torch as ch

from distribution_inference.datasets import new_census, celeba, boneage,census, texas

DATASET_INFO_MAPPING = {
    "new_census": new_census.DatasetInformation,
    "celeba": celeba.DatasetInformation,
    "boneage": boneage.DatasetInformation,
    "old_census": census.DatasetInformation,
    "texas": texas.DatasetInformation,
}

DATASET_WRAPPER_MAPPING = {
    "new_census": new_census.CensusWrapper,
    "celeba": celeba.CelebaWrapper,
    "boneage": boneage.BoneWrapper,
    "old_census": census.CensusWrapper,
    "texas": texas.TexasWrapper,
}


def get_dataset_wrapper(dataset_name: str):
    wrapper = DATASET_WRAPPER_MAPPING.get(dataset_name, None)
    if not wrapper:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return wrapper


def get_dataset_information(dataset_name: str):
    info = DATASET_INFO_MAPPING.get(dataset_name, None)
    if not info:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return info


# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def filter(df, condition, ratio, verbose: bool = True):
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


def heuristic(df, condition, ratio: float,
              cwise_sample: int,
              class_imbalance: float = 2.0,
              n_tries: int = 1000,
              class_col: str = "label",
              verbose: bool = True):
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        # Binary class- simply sample (as requested)
        # From each class
        pckd_df = filter(df, condition, ratio, verbose=False)
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]

        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(
                    one_ids)[:cwise_sample]
            elif class_imbalance < 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(
                    one_ids)[:int(1 / class_imbalance * cwise_sample)]
            else:
                raise ValueError(f"Invalid class_imbalance value: {class_imbalance}")

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


def collect_data(loader, expect_extra: bool = True):
    X, Y = [], []
    for datum in loader:
        if expect_extra:
            x, y, _ = datum
        else:
            x, y = datum
        X.append(x)
        Y.append(y)
    # Concatenate both torch tensors across batches
    X = ch.cat(X, dim=0)
    Y = ch.cat(Y, dim=0)
    return X, Y
