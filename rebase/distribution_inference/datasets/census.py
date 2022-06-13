import os
import argparse
import pandas as pd
import numpy as np
import torch as ch
import torch.nn as nn
from joblib import load
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.models.core import PortedMLPClassifier
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model, save_model


def skload_model(path):
    return load(path)


def get_models(folder_path):
    """
        Load models from given directory.
    """
    paths = os.listdir(folder_path)
    models = []
    names = []
    for mpath in tqdm(paths):
        p = os.path.join(folder_path, mpath)
        if os.path.isfile(p):
            model = skload_model(p)
            models.append(model)
            names.append(mpath)
    return models, names


def port_mlp_to_ch(clf):
    """
        Extract weights from MLPClassifier and port
        to PyTorch model.
    """
    nn_model = PortedMLPClassifier()
    i = 0
    for (w, b) in zip(clf.coefs_, clf.intercepts_):
        w = ch.from_numpy(w.T).float()
        b = ch.from_numpy(b).float()
        nn_model.layers[i].weight = nn.Parameter(w)
        nn_model.layers[i].bias = nn.Parameter(b)
        i += 2  # Account for ReLU as well

    #nn_model = nn_model.cuda()
    return nn_model


def convert_to_torch(clfs):
    """
        Port given list of MLPClassifier models to
        PyTorch models
    """
    return np.array([port_mlp_to_ch(clf) for clf in clfs], dtype=object)


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch: bool = False):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="Census",
                         data_path="census",
                         models_path="models_census/50_50_new/normal",
                         properties=["sex", "race"],
                         values={"sex": ratios, "race": ratios, },
                         property_focus={"sex": 'Female', "race": 'White'},
                         epoch_wise=epoch)

    def get_model(self, cpu: bool = False, full_model: bool = False) -> nn.Module:
        clf = PortedMLPClassifier()
        if not cpu:
            clf = clf.cuda()
        return clf


class _CensusIncome:
    def __init__(self):
        self.columns = [
            "age", "workClass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        self.dropped_cols = ["education", "native-country"]
        self.info_object = DatasetInformation()
        self.path = self.info_object.base_data_dir
        # self.load_data(test_ratio=0.4)
        self.load_data(test_ratio=0.5)

    def process_df(self, df):
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = ['workClass', 'occupation', 'race', 'sex',
                    'marital-status', 'relationship']
        # Drop columns that do not help with task
        df = df.drop(columns=self.dropped_cols, axis=1)
        # Club categories not directly relevant for property inference
        df["race"] = df["race"].replace(
            ['Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other')
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        # Drop features pruned via feature engineering
        prune_feature = [
            "workClass:Never-worked",
            "workClass:Without-pay",
            "occupation:Priv-house-serv",
            "occupation:Armed-Forces"
        ]
        df = df.drop(columns=prune_feature, axis=1)
        return df

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        Y = P['income'].to_numpy()
        X = P.drop(columns='income', axis=1)
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop, custom_limit=None):

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to data
            TRAIN_DF = get_filter(TRAIN_DF, filter_prop,
                                  split, prop_ratio, is_test=0,
                                  custom_limit=custom_limit)
            TEST_DF = get_filter(TEST_DF, filter_prop,
                                 split, prop_ratio, is_test=1,
                                 custom_limit=custom_limit)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(
                TRAIN_DF), self.get_x_y(TEST_DF)

            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    # Create adv/victim splits, normalize data, etc
    def load_data(self, test_ratio, random_state=42):
        # Load train, test data
        train_data = pd.read_csv(os.path.join(self.path, 'adult.data'),
                                 names=self.columns, sep=' *, *',
                                 na_values='?', engine='python')
        test_data = pd.read_csv(os.path.join(self.path, 'adult.test'),
                                names=self.columns, sep=' *, *', skiprows=1,
                                na_values='?', engine='python')

        # Add field to identify train/test, process together
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        df = pd.concat([train_data, test_data], axis=0)
        df = self.process_df(df)

        # Take note of columns to scale with Z-score
        z_scale_cols = ["fnlwgt", "capital-gain", "capital-loss"]
        for c in z_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        # Take note of columns to scale with min-max normalization
        minmax_scale_cols = ["age",  "hours-per-week", "education-num"]
        for c in minmax_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].min()) / df[c].max()

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']
                                         == 1], df[df['is_train'] == 0]

        # Drop 'train/test' columns
        self.train_df = self.train_df.drop(columns=['is_train'], axis=1)
        self.test_df = self.test_df.drop(columns=['is_train'], axis=1)

        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=test_ratio,
                                         random_state=rs)
            # Stratification on the properties we care about for this dataset
            # so that adv/victim split does not introduce
            # unintended distributional shift
            splitter = sss.split(
                this_df, this_df[["sex:Female", "race:White", "income"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        # Create train/test splits for victim/adv
        self.train_df_victim, self.train_df_adv = s_split(self.train_df)
        self.test_df_victim, self.test_df_adv = s_split(self.test_df)


class CensusSet(base.CustomDataset):
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
class CensusWrapper(base.CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig, skip_data: bool = False):
        super().__init__(data_config, skip_data)
        if not skip_data:
            self.ds = _CensusIncome()

    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.prop,
                                custom_limit=custom_limit)

    def get_loaders(self, batch_size, custom_limit=None,
                    shuffle: bool = True,
                    eval_shuffle: bool = False):
        train_data, val_data, _ = self.load_data(custom_limit)
        self.ds_train = CensusSet(*train_data, squeeze=self.squeeze)
        self.ds_val = CensusSet(*val_data, squeeze=self.squeeze)
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   num_workers=1)

    def load_model(self, path: str, on_cpu: bool = False) -> nn.Module:
        info_object = DatasetInformation()
        model = info_object.get_model(cpu=on_cpu)
        return load_model(model, path, on_cpu=on_cpu)

    def get_save_dir(self, train_config: TrainConfig, full_model: bool) -> str:
        info_object = DatasetInformation()
        base_models_dir = info_object.base_models_dir

        save_path = os.path.join(base_models_dir, self.prop, self.split)
        if self.ratio is not None:
            save_path = os.path.join(save_path, str(self.ratio))
        # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        return save_path


def con_b(df):
    return (df['sex:Female'] == 1) & (df['race:White'] == 1)


def con_f(df):
    return (df['sex:Female'] == 1) & (df['race:White'] == 0)


def con_w(df):
    return (df['sex:Female'] == 0) & (df['race:White'] == 1)


def con_n(df):
    return (df['sex:Female'] == 0) & (df['race:White'] == 0)


con_l = [con_b, con_f, con_w, con_n]
# Get appropriate filter with sub-sampling according to ratio and property


def get_filter(df, filter_prop, split, ratio, is_test, custom_limit=None):
    if filter_prop == "none":
        return df
    elif filter_prop == "sex":
        def lambda_fn(x): return x['sex:Female'] == 1
    elif filter_prop == "race":
        def lambda_fn(x): return x['race:White'] == 1
    elif filter_prop == "bothfw":
        def lambda_fn(x):
            a = x['sex:Female'] == 1
            b = x['race:White'] == 1
            return a & b
    elif filter_prop == "bothmw":
        def lambda_fn(x):
            a = x['sex:Female'] == 0
            b = x['race:White'] == 1
            return a & b
    elif filter_prop == "bothfn":
        def lambda_fn(x):
            a = x['sex:Female'] == 1
            b = x['race:White'] == 0
            return a & b
    elif filter_prop == "bothmn":
        def lambda_fn(x):
            a = x['sex:Female'] == 0
            b = x['race:White'] == 0
            return a & b
    # Rerun with 0.5:0.5
    prop_wise_subsample_sizes = {
        "adv": {
            "sex": (1100, 500),
            "race": (2000, 1000),
            "bothfw": (900, 400),
            "bothfn": (210, 100),
            "bothmn": (260, 130),
            "bothmw": (2000, 960),
        },
        "victim": {
            "sex": (1100, 500),
            "race": (2000, 1000),
            "bothfw": (900, 400),
            "bothfn": (210, 100),
            "bothmn": (260, 130),
            "bothmw": (2000, 960),
        },
    }

    if custom_limit is None:
        subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
    else:
        subsample_size = custom_limit
    return utils.heuristic(df, lambda_fn, ratio,
                           subsample_size, class_imbalance=3,
                           n_tries=100, class_col='income',
                           verbose=False)


def cal_q(df, condition):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    return len(qualify), len(notqualify)


def get_df(df, condition, x):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    np.random.shuffle(qualify)
    return df.iloc[qualify[:x]]


def cal_n(df, con, ratio):
    q, n = cal_q(df, con)
    current_ratio = q / (q+n)
    # If current ratio less than desired ratio, subsample from non-ratio
    if current_ratio <= ratio:
        if ratio < 1:
            nqi = (1-ratio) * q/ratio
            return q, nqi
        return q, 0
    else:
        if ratio > 0:
            qi = ratio * n/(1 - ratio)
            return qi, n
        return 0, n


if __name__ == "__main__":
    "run to convert sklearn model to pytorch, have to comment out distribution_inference.datasets.util due to circular importing"
    info = DatasetInformation()
    parser = argparse.ArgumentParser()

    parser.add_argument('--suffix',
                        default='50_50_new', help="device number")
    args = parser.parse_args()
    BMD = "/p/adversarialml/as9rw/models_census"
    BMD = os.path.join(BMD, args.suffix)
    b = os.path.join(BMD, 'normal')
    for s in ["adv", "victim"]:
        pa = os.path.join(BMD, s)
        for p in ['sex', 'race']:
            pat = os.path.join(pa, p)
            print(pat)
            for x in os.listdir(pat):
                models, names = get_models(os.path.join(pat, x))
                save_path = os.path.join(b, p, s, x)
                if models != []:
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    models = convert_to_torch(models)
                    for (m, n) in zip(models, names):
                        save_model(m, os.path.join(save_path, n))
