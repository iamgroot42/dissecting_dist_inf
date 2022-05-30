from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
import os
import torch as ch
import torch.nn as nn
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.models.core import MLPTwoLayer
import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
from distribution_inference.training.utils import load_model


class DatasetInformation(base.DatasetInformation):
    def __init__(self,epoch_wise:bool=False):
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        super().__init__(name="New Census",
                         data_path="census_new/census_2019_1year",
                         models_path="models_new_census/60_40",
                         properties=["sex", "race"],
                         values={"sex": ratios, "race": ratios},
                         property_focus={"sex": 'female', "race": 'white'},
                         epoch_wise=epoch_wise)

    def get_model(self, cpu: bool = False) -> nn.Module:
        model = MLPTwoLayer(n_inp=105)
        if not cpu:
            model = model.cuda()
        return model

    def get_model_for_dp(self, cpu: bool = False) -> nn.Module:
        model = MLPTwoLayer(n_inp=105)
        if not cpu:
            model = model.cuda()
        return model

    def generate_victim_adversary_splits(self,
                                         adv_ratio=None,
                                         test_ratio: float = 0.33,
                                         num_tries: int = None):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        def cal_q(df, condition):
            qualify = np.nonzero((condition(df)).to_numpy())[0]
            notqualify = np.nonzero(np.logical_not(
                (condition(df)).to_numpy()))[0]
            return len(qualify), len(notqualify)

        x = pickle.load(
            open(os.path.join(self.base_data_dir, 'census_features.p'), 'rb'))
        y = pickle.load(
            open(os.path.join(self.base_data_dir, 'census_labels.p'), 'rb'))
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_ratio)
        train = np.hstack((X_train, y_train[..., None]))
        test = np.hstack((X_test, y_test[..., None]))
        pickle.dump(train, open('./train.p', "wb"))
        pickle.dump(test, open('./test.p', "wb"))

        dt = _CensusIncome()

        def fe(x):
            return x['sex'] == 1

        def ra(x):
            return x['race'] == 0

        adv_tr, adv_te, vic_tr, vic_te = dt.train_df_adv, dt.test_df_adv, dt.train_df_victim, dt.test_df_victim
        print('adv train female and male: {}'.format(cal_q(adv_tr, fe)))
        print('adv test female and male: {}\n'.format(cal_q(adv_te, fe)))
        print('vic train female and male: {}'.format(cal_q(vic_tr, fe)))
        print('vic test female and male: {}\n'.format(cal_q(vic_te, fe)))
        print('adv train white and non: {}'.format(cal_q(adv_tr, ra)))
        print('adv test white and non: {}\n'.format(cal_q(adv_te, ra)))
        print('vic train white and non: {}'.format(cal_q(vic_tr, ra)))
        print('vic test white and non: {}'.format(cal_q(vic_te, ra)))


class _CensusIncome:
    def __init__(self, drop_senstive_cols=False):
        self.drop_senstive_cols = drop_senstive_cols
        self.columns = [
            "age", "workClass", "education-attainment",
            "marital-status", "race", "sex", "cognitive-difficulty",
            "ambulatory-difficulty", "hearing-difficulty", "vision-difficulty",
            "work-hour", "world-area-of-birth", "state-code", "income"
        ]
        self.load_data(test_ratio=0.4)

    # Return data with desired property ratios
    def get_x_y(self, P):
        # Scale X values
        Y = P['income'].to_numpy()
        cols_drop = ['income']
        if self.drop_senstive_cols:
            cols_drop += ['sex', 'race']
        X = P.drop(columns=cols_drop, axis=1)
        # Convert specific columns to one-hot
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop,
                 custom_limit=None, scale: float = 1.0,label_noise:float=0):

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
            if label_noise:
                #shape of y: (length,1)
                idx = np.random.choice(len(y_tr),int (label_noise*len(y_tr)),replace=False)
                y_tr[idx][0] = not y_tr[idx][0]
            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    # Process, handle one-hot conversion of data etc
    def process_df(self, df):
        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = ['state-code', 'world-area-of-birth',
                    'marital-status', 'workClass',
                    'education-attainment']
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        return df

    # Create adv/victim splits
    def load_data(self, test_ratio, random_state: int = 42):
        info_object = DatasetInformation()
        # Load train, test data
        train_data = pickle.load(
            open(os.path.join(info_object.base_data_dir, "data", 'train.p'), 'rb'))
        test_data = pickle.load(
            open(os.path.join(info_object.base_data_dir, "data", 'test.p'), 'rb'))
        self.train_df = pd.DataFrame(train_data, columns=self.columns)
        self.test_df = pd.DataFrame(test_data, columns=self.columns)

        # Add field to identify train/test, process together
        self.train_df['is_train'] = 1
        self.test_df['is_train'] = 0
        df = pd.concat([self.train_df, self.test_df], axis=0)
        df = self.process_df(df)

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']
                                         == 1], df[df['is_train'] == 0]
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
                this_df, this_df[["sex", "race", "income"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        # Create train/test splits for victim/adv
        self.train_df_victim, self.train_df_adv = s_split(self.train_df)
        self.test_df_victim, self.test_df_adv = s_split(self.test_df)

    # Fet appropriate filter with sub-sampling according to ratio and property
    def get_filter(self, df, filter_prop, split, ratio, is_test,
                   custom_limit=None, scale: float = 1.0):
        if filter_prop == "none":
            return df
        elif filter_prop == "sex":
            def lambda_fn(x): return x['sex'] == 1
        elif filter_prop == "race":
            def lambda_fn(x): return x['race'] == 0

        # For 1-year Census data
        prop_wise_subsample_sizes = {
            "adv": {
                "sex": (20000, 12000),
                "race": (14000, 9000),
            },
            "victim": {
                "sex": (30000, 20000),
                "race": (21000, 14000),
            },
        }

        if custom_limit is None:
            subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
            subsample_size = int(scale*subsample_size)
        else:
            subsample_size = custom_limit
        return utils.heuristic(df, lambda_fn, ratio,
                               subsample_size,
                               class_imbalance=0.7211,  # Calculated based on original distribution
                               n_tries=100,
                               class_col='income',
                               verbose=False)


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
    def __init__(self, data_config: DatasetConfig, skip_data: bool = False,epoch:bool=False,label_noise:float=0):
        super().__init__(data_config, skip_data,label_noise)
        if not skip_data:
            self.ds = _CensusIncome(drop_senstive_cols=self.drop_senstive_cols)
        self.info_object = DatasetInformation(epoch_wise=epoch)
        
    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.prop,
                                custom_limit=custom_limit,
                                scale=self.scale,
                                label_noise = self.label_noise)

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False):
        train_data, val_data, _ = self.load_data(self.cwise_samples)
        self.ds_train = CensusSet(*train_data, squeeze=self.squeeze)
        self.ds_val = CensusSet(*val_data, squeeze=self.squeeze)
        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,)

    def load_model(self, path: str, on_cpu: bool = False) -> nn.Module:
        info_object = self.info_object
        model = info_object.get_model(cpu=on_cpu)
        return load_model(model, path,on_cpu=on_cpu)

    def get_save_dir(self, train_config: TrainConfig) -> str:
        info_object = self.info_object
        base_models_dir = info_object.base_models_dir
        dp_config = None
        if train_config.misc_config is not None:
            dp_config = train_config.misc_config.dp_config

        if dp_config is None:
            base_path = os.path.join(base_models_dir, "normal")
           
        else:
            base_path = os.path.join(
                base_models_dir, "DP_%.2f" % dp_config.epsilon)
        if self.label_noise:
            base_path = os.path.join(
                base_models_dir, "label_noise:{}".format(train_config.label_noise))
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
