import utils
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import   Dataset
# BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/census_new/census_2019_5year"
BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/census_new/census_2019_1year"
SUPPORTED_PROPERTIES = ["sex", "race"]
PROPERTY_FOCUS = {"sex": 'female', "race": 'white'} # in original dataset, 0 for male, 1 for female; 0 for white



# US Income dataset
class CensusIncome:
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
            cols = X.columns
            X = X.to_numpy()
            return (X.astype(float), np.expand_dims(Y, 1), cols)

    def get_data(self, split, prop_ratio, filter_prop, custom_limit=None,scale=1.0):

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to data
            TRAIN_DF = get_filter(TRAIN_DF, filter_prop,
                                  split, prop_ratio, is_test=0,
                                  custom_limit=custom_limit,
                                  scale=scale)
            TEST_DF = get_filter(TEST_DF, filter_prop,
                                 split, prop_ratio, is_test=1,
                                 custom_limit=custom_limit,
                                 scale=scale)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = self.get_x_y(
                TRAIN_DF), self.get_x_y(TEST_DF)

            return (x_tr, y_tr), (x_te, y_te), cols

        if split == "all":
            return prepare_one_set(self.train_df, self.test_df)
        if split == "victim":
            return prepare_one_set(self.train_df_victim, self.test_df_victim)
        return prepare_one_set(self.train_df_adv, self.test_df_adv)

    # Create adv/victim splits
    def load_data(self, test_ratio, random_state=42):
        # Load train, test data
        train_data = pickle.load(
            open(os.path.join(BASE_DATA_DIR, "data", 'train.p'), 'rb'))
        test_data = pickle.load(
            open(os.path.join(BASE_DATA_DIR, "data", 'test.p'), 'rb'))
        self.train_df = pd.DataFrame(train_data, columns=self.columns)
        self.test_df = pd.DataFrame(test_data, columns=self.columns)
        '''
        #set race to binary: 0 for white, 1 for the rest
        self.train_df['race'] = (self.train_df['race'] != 0).astype(int)
        self.test_df['race'] = (self.test_df['race'] != 0).astype(int)
        #set sex columns to int, 0 for male, 1 for female
        self.train_df['sex'] = self.train_df['sex'].astype(int)
        self.test_df['sex'] = self.test_df['sex'].astype(int)
        '''
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
def get_filter(df, filter_prop, split, ratio, is_test, custom_limit=None,scale=1.0):
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


def cal_q(df, condition):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    return len(qualify), len(notqualify)


def get_df(df, condition, x):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    np.random.shuffle(qualify)
    return df.iloc[qualify[:x]]

# Wrapper for easier access to dataset


class CensusWrapper:
    def __init__(self,
                 filter_prop="none",
                 ratio=0.5,
                 split="all",
                 drop_senstive_cols=False,
                 scale=1.0
                 ):
        self.ds = CensusIncome(drop_senstive_cols=drop_senstive_cols)
        self.split = split
        self.ratio = ratio
        self.filter_prop = filter_prop
        self.scale=scale
        

    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.filter_prop,
                                custom_limit=custom_limit,scale=self.scale)

class CensusSet(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets.ravel()
        
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y,0#zero because no property label to return, but compatible with methods from utils
    
    def __len__(self):
        return len(self.data)    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true',
                        help='create test and train split')
    args = parser.parse_args()
    #create test and train split
    # Do not run with 'split' for 1 year (already generated splits)
    if args.split:
        x = pickle.load(
            open(os.path.join(BASE_DATA_DIR, 'census_features.p'), 'rb'))
        y = pickle.load(
            open(os.path.join(BASE_DATA_DIR, 'census_labels.p'), 'rb'))
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33)
        train = np.hstack((X_train, y_train[..., None]))
        test = np.hstack((X_test, y_test[..., None]))
        pickle.dump(train, open('./train.p', "wb"))
        pickle.dump(test, open('./test.p', "wb"))
    dt = CensusIncome()

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
   