import utils
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import requests
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/census_new/census_2019_5year"
#SUPPORTED_PROPERTIES = ["sex", "race", "none","bothfw","bothmw","bothfn","bothmn","two_attr"]
#PROPERTY_FOCUS = {"sex": "Female", "race": "White","bothfw":"both f and w","bothmw":"both m and w","bothfn":"both f and n","bothmn":"both m and n","two_attr":"f and w"}


# US Income dataset
class CensusIncome:
    def __init__(self):
        
        self.columns = [
            "age", "workClass", "education-attainment",
            "marital-status", "race-code", "sex","cognitive-difficulty",
            "ambulatory-difficulty","hearing-difficulty","vision-difficulty",
            "work-hour","world-area-of-birth","state-code","income"
        ]
        
        # self.load_data(test_ratio=0.4)
        self.load_data(test_ratio=0.5)

    # Process, handle one-hot conversion of data etc
    def process_df(self, df):

        pass

    # Return data with desired property ratios
    def get_x_y(self,P):
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
        train_data = pickle.load(open('./train.p'), 'rb')

        test_data = pickle.load(open('./train.p'), 'rb')
        self.train_df = pd.DataFrame(train_data,self.columns)
        self.test_df = pd.DataFrame(test_data,self.columns)
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




# Fet appropriate filter with sub-sampling according to ratio and property
def get_filter(df, filter_prop, split, ratio, is_test, custom_limit=None):
    if filter_prop == "none":
        return df
    elif filter_prop == "sex":
        def lambda_fn(x): return x['sex:Female'] == 1
    elif filter_prop == "race":
        def lambda_fn(x): return x['race:White'] == 1
    
    prop_wise_subsample_sizes = {
        "adv": {
            "sex": (1100, 500),
            "race": (2000, 1000),
            "bothfw": (900, 400),
            "bothfn": (210, 100),
            "bothmn": (260, 130),
            "bothmw": (2000,960),
            
        },
        "victim": {
            "sex": (1100, 500),
            "race": (2000, 1000),
            "bothfw": (900, 400),
            "bothfn": (210, 100),
            "bothmn": (260, 130),
            "bothmw": (2000,960),
            
            
            
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
def cal_q(df,condition):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    return len(qualify),len(notqualify)
def get_df(df,condition,x):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    np.random.shuffle(qualify)
    return df.iloc[qualify[:x]]

# Wrapper for easier access to dataset
class CensusWrapper:
    def __init__(self, filter_prop="none", ratio=0.5, split="all"):
        self.ds = CensusIncome()
        self.split = split
        self.ratio = ratio
        self.filter_prop = filter_prop

    def load_data(self, custom_limit=None):
        return self.ds.get_data(split=self.split,
                                prop_ratio=self.ratio,
                                filter_prop=self.filter_prop,
                                custom_limit=custom_limit)

if __name__ == "__main__":
    #create test and train split
    x = pickle.load(open(os.path.join(BASE_DATA_DIR,'census_features.p'), 'rb'))
    y = pickle.load(open(os.path.join(BASE_DATA_DIR,'census_labels.p'), 'rb'))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    train = np.hstack((X_train,y_train[...,None]))
    test = np.hstack((X_test,y_test[...,None]))
    pickle.dump( train, open('./train.p', "wb" ) )
    pickle.dump( test, open( './test.p', "wb" ) )
