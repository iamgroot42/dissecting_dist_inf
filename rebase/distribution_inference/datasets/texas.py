from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch as ch
import torch.nn as nn

from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.models.core import MLPFourLayer
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch: bool = False):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.num_features = 3611
        self.num_classes = 20
        super().__init__(name="Texas 100 v2",
                         data_path="texas_100_v2/",
                         models_path="models_texas",
                         properties=["sex", "race", "ethnicity"],
                         values={"sex": ratios, "race": ratios,
                                 "ethnicity": ratios},
                         property_focus={
                             "sex": 'female',
                             "race": 'white',
                             "ethnicity": 'hispanic'},
                         epoch_wise=epoch)

    def get_model(self, cpu: bool = False) -> nn.Module:
        model = MLPFourLayer(n_inp=self.num_features,
                             num_classes=self.num_classes)
        if not cpu:
            model = model.cuda()
        return model

    def get_model_for_dp(self, cpu: bool = False) -> nn.Module:
        model = MLPFourLayer(n_inp=self.num_features,
                             num_classes=self.num_classes)
        if not cpu:
            model = model.cuda()
        return model

    # Process, handle one-hot conversion of data etc
    def _process_df(self, df):
        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = [
            'type_of_admission', 'source_of_admission',
            'pat_status', 'admitting_diagnosis']
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        return df

    def generate_victim_adversary_splits(self,
                                         adv_ratio=0.25,
                                         test_ratio: float = 0.25,
                                         num_tries: int = 300):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        # Use THCIC-ID to create victim/adv splits, since
        # these are different hospitals and would be a great
        # way to split the data
        x = pickle.load(open(os.path.join(
            self.base_data_dir, "texas_100_v2_features.p"), 'rb'))
        y = pickle.load(open(os.path.join(
            self.base_data_dir, "texas_100_v2_labels.p"), 'rb'))

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        # Dataset-specific column information
        columns = ["thcic_id", "sex", "type_of_admission",
                   "source_of_admission", "length_of_stay",
                   "pat_age", "pat_status", "race", "ethnicity",
                   "total_charges", "admitting_diagnosis", "label"]
        relevant_columns = ["sex", "race", "ethnicity"]
        relevant_indices = [columns.index(x) for x in relevant_columns]

        # Skip classes that have either have mostly 1s or 0s for relevant columns
        labels, counts = np.unique(y, return_counts=True)
        labels_to_skip = []
        for ri in relevant_indices:
            for label in labels:
                mean_ratio = np.mean(x[(y == label), ri])
                if mean_ratio < 0.2 or mean_ratio > 0.8:
                    labels_to_skip.append(label)
        labels_to_skip = list(set(labels_to_skip))
        if len(labels_to_skip) > 0:
            pick_these = np.logical_not(np.isin(y, labels_to_skip))
            x = x[pick_these]
            y = y[pick_these]
            # Recompute labels and their counts
            labels, counts = np.unique(y, return_counts=True)

        # Pick only top 20 classes out of the remaining ones
        # Ditch everything else
        top_n = labels[np.argsort(counts)[-20:]]
        which_to_keep = np.isin(y, top_n)
        x = x[which_to_keep]
        y = y[which_to_keep]
        # Renumber labels to 0, 1, 2, according to their frequency
        label_map = {x: i for i, x in enumerate(top_n)}
        y = np.array([label_map[x] for x in y])

        # Convert race to white/non-white: 0.75 is White (0), rest are non-White (1)
        x[:, columns.index("race")] = 1 * (x[:, columns.index("race")] != 0.75)

        # TODO: Account for labels in split creation as well
        # Function to convery labels to one-hot
        def to_onehot(z):
            onehot = np.zeros((z.size, y.max()+1))
            onehot[np.arange(z.size), z] = 1
            return onehot

        # Function to split data, on the basis of hospital IDs
        # Useful for victim/adv splits as well as train/val splits
        def create_split(data_, ratio, labels_):
            # We ideally want a split of the hospitals such that the
            # resulting distributions are as similar as possible
            data = np.copy(data_)
            thcic_ids = np.unique(data[:, 0])
            # Convert labels to onehot
            onehot_labels = to_onehot(labels_)

            best_splits, best_dist = None, np.inf
            # Keep trying different random splits
            iterator = tqdm(range(num_tries))
            for _ in iterator:
                # Create random splits on thcic-ids
                vic_ids, adv_ids = train_test_split(
                    list(range(len(thcic_ids))), test_size=ratio)
                # Look at the "L2" distance in the ratios of relevance for specified columns
                relevant_indices_vic = np.any(
                    [data[:, 0] == thcic_ids[id_] for id_ in vic_ids], 0)
                relevant_indices_adv = np.any(
                    [data[:, 0] == thcic_ids[id_] for id_ in adv_ids], 0)
                vic_sim = np.mean(
                    data[relevant_indices_vic][:, relevant_indices], 0)
                adv_sim = np.mean(
                    data[relevant_indices_adv][:, relevant_indices], 0)
                # Also use onehot representation in similarities
                vic_label_sim = np.mean(onehot_labels[relevant_indices_vic], 0)
                adv_label_sim = np.mean(onehot_labels[relevant_indices_adv], 0)
                # Combine label-sim to normal-sim
                vic_sim = np.concatenate((vic_sim, vic_label_sim))
                adv_sim = np.concatenate((adv_sim, adv_label_sim))
                # Look at their L-2 distance
                vic_dist = np.linalg.norm(vic_sim - adv_sim)
                if vic_dist < best_dist:
                    best_dist = vic_dist
                    best_splits = relevant_indices_vic, relevant_indices_adv
                    iterator.set_description(
                        "L-2 Loss for split: %.5f" % best_dist)
            return best_splits

        # First, create victim-adv split
        victim_ids, adv_ids = create_split(x, adv_ratio, y)
        x_victim, y_victim = x[victim_ids], y[victim_ids]
        x_adv, y_adv = x[adv_ids], y[adv_ids]

        # Now, create train-test splits for both victim and adv
        victim_train_ids, _ = create_split(
            x_victim, test_ratio, y_victim)
        adv_train_ids, _ = create_split(
            x_adv, test_ratio, y_adv)

        # Add 'label' to end of x
        x = np.concatenate((x, np.expand_dims(y, axis=1)), axis=1)

        # Create a dataframe out of original data (to process)
        df = pd.DataFrame(x, columns=columns)

        # Process the DF
        df = self._process_df(df)

        # Split into victim & adv DFs
        df_victim = df.iloc[victim_ids]
        df_adv = df.iloc[adv_ids]

        # Add train/test identifier
        is_train_col_vic, is_train_col_adv = np.zeros(
            len(df_victim)), np.zeros(len(df_adv))
        is_train_col_vic[victim_train_ids] = 1
        is_train_col_adv[adv_train_ids] = 1
        df_victim['is_train'] = is_train_col_vic
        df_adv['is_train'] = is_train_col_adv

        # Reset index
        df_victim.reset_index(drop=True, inplace=True)
        df_adv.reset_index(drop=True, inplace=True)

        # Save processed data
        df_victim.to_pickle(os.path.join(
            self.base_data_dir, "texas_100_v2_victim.pkl"))
        df_adv.to_pickle(os.path.join(
            self.base_data_dir, "texas_100_v2_adv.pkl"))


class _Texas:
    def __init__(self, drop_senstive_cols=False):
        self.drop_senstive_cols = drop_senstive_cols
        self.columns = ["thcic_id", "sex", "type_of_admission",
                        "source_of_admission", "length_of_stay",
                        "pat_age", "pat_status", "race", "ethnicity",
                        "total_charges", "admitting_diagnosis"]
        self.relevant_columns = ["sex", "race", "ethnicity"]
        self.load_data()

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        Y = P['label'].to_numpy()
        cols_drop = ['label']
        if self.drop_senstive_cols:
            cols_drop += ['sex', 'race', 'ethnicity']
        X = P.drop(columns=cols_drop, axis=1)
        # Convert specific columns to one-hot
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop,
                 custom_limit=None, scale: float = 1.0):

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to data
            TRAIN_DF = self.get_filter(TRAIN_DF, filter_prop,
                                       split, prop_ratio, is_test=0,
                                       custom_limit=custom_limit,
                                       scale=scale)
            TEST_DF = self.get_filter(TEST_DF, filter_prop,
                                      split, prop_ratio, is_test=1,
                                      custom_limit=custom_limit,
                                      scale=scale)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(
                TRAIN_DF), self.get_x_y(TEST_DF)

            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "victim":
            return prepare_one_set(self.train_df_vic, self.test_df_vic)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    def load_data(self):
        info_object = DatasetInformation()

        def split_and_remove(data):
            data_train = data[data['is_train'] == True]
            data_test = data[data['is_train'] == False]
            # Drop 'is_train' column
            data_train.drop(columns=['is_train'], inplace=True)
            data_test.drop(columns=['is_train'], inplace=True)
            return data_train, data_test

        # Load victim & adv DFs
        # TODO- This step is quite slow right now
        # Need to speed this up (load only the requested split, perhaps)
        victim_df = pd.read_pickle(os.path.join(
            info_object.base_data_dir, "texas_100_v2_victim.pkl"))
        adv_df = pd.read_pickle(os.path.join(
            info_object.base_data_dir, "texas_100_v2_adv.pkl"))
        # Drop 'thcic_id' from both DFs
        victim_df.drop(columns=['thcic_id'], inplace=True)
        adv_df.drop(columns=['thcic_id'], inplace=True)

        self.train_df_vic, self.test_df_vic = split_and_remove(victim_df)
        self.train_df_adv, self.test_df_adv = split_and_remove(adv_df)

    # Fet appropriate filter with sub-sampling according to ratio and property
    def get_filter(self, df, filter_prop, split, ratio, is_test,
                   custom_limit=None, scale: float = 1.0):
        if filter_prop == "none":
            return df
        elif filter_prop == "sex":
            def lambda_fn(x): return x['sex'] == 1
        elif filter_prop == "race":
            def lambda_fn(x): return x['race'] == 0
        elif filter_prop == "ethnicity":
            def lambda_fn(x): return x['ethnicity'] == 0

        # TODO- Figure out appropriate sub-sample sizes
        # For this dataaset
        prop_wise_sample_sizes = {
            "adv": {
                "sex": (14000, 3000),
                "race": (1400, 900),
                "ethnicity": (1000, 1000)
            },
            "victim": {
                "sex": (40000, 8000),
                "race": (2100, 1400),
                "ethnicity": (100, 100)
            },
        }

        if custom_limit is None:
            sample_size = prop_wise_sample_sizes[split][filter_prop][is_test]
            sample_size = int(scale*sample_size)
        else:
            sample_size = custom_limit
        return utils.multiclass_heuristic(
            df, lambda_fn, ratio,
            sample_size,
            n_tries=5,
            class_col='label',
            verbose=True)


class TexasSet(base.CustomDataset):
    def __init__(self, data, targets, squeeze=False):
        self.data = ch.from_numpy(data).float()
        self.targets = ch.from_numpy(targets).float()
        self.squeeze = squeeze
        self.num_samples = len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.squeeze:
            y = y.squeeze()

        # Set property label to -1
        # Not really used, but ensures compatibility with methods
        # from utils
        return x, y, -1


# Wrapper for easier access to dataset
class TexasWrapper(base.CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig, skip_data: bool = False):
        super().__init__(data_config, skip_data)
        if not skip_data:
            self.ds = _Texas(drop_senstive_cols=self.drop_senstive_cols)
            # self.ds = _Texas(drop_senstive_cols=True)

    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.prop,
                                custom_limit=custom_limit,
                                scale=self.scale)

    def get_loaders(self, batch_size: int,
                    custom_limit=None,
                    shuffle: bool = True,
                    eval_shuffle: bool = False):
        train_data, val_data, _ = self.load_data(custom_limit)
        self.ds_train = TexasSet(*train_data, squeeze=self.squeeze)
        self.ds_val = TexasSet(*val_data, squeeze=self.squeeze)
        print(len(self.ds_train), len(self.ds_val))
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,)

    def load_model(self, path: str, on_cpu: bool = False) -> nn.Module:
        info_object = DatasetInformation()
        model = info_object.get_model(cpu=on_cpu)
        return load_model(model, path)

    def get_save_dir(self, train_config: TrainConfig) -> str:
        info_object = DatasetInformation()
        base_models_dir = info_object.base_models_dir
        dp_config = None
        if train_config.misc_config is not None:
            dp_config = train_config.misc_config.dp_config

        if dp_config is None:
            base_path = os.path.join(base_models_dir, "normal")
        else:
            base_path = os.path.join(
                base_models_dir, "DP_%.2f" % dp_config.epsilon)

        save_path = os.path.join(base_path, self.prop, self.split)
        if self.ratio is not None:
            save_path = os.path.join(save_path, str(self.ratio))

        if self.scale != 1.0:
            save_path = os.path.join(
                self.scalesave_path, "sample_size_scale:{}".format(self.scale))
        if self.drop_senstive_cols:
            save_path = os.path.join(save_path, "drop")

        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path
