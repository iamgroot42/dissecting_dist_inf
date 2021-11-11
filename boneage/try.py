from model_utils import load_model, get_model_folder_path
from data_utils import BoneWrapper, get_df, get_features
import torch.nn as nn
import torch as ch
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc, flash_utils,heuristic
import matplotlib.pyplot as plt
import matplotlib as mpl
from perf_tests import get_models
mpl.rcParams['figure.dpi'] = 200
def filter(x): return x["gender"] == 1

    # Ready data
df_train, df_val = get_df("adv")
features = get_features("adv")

    # Get data with ratio
df_1 = heuristic(
    df_val, filter, float(0.5),
    cwise_sample=10000,
    class_imbalance=1.0, n_tries=300)

    

    # Prepare data loaders
ds_1 = BoneWrapper(
df_1, df_1, features=features)
loader = ds_1.get_loaders(256*32, shuffle=False)[1]
def get_preds(loader,ms):
    
    ps = []
    for m in (ms):
        m=m.cuda()
        m.eval()
        p=[]
        ch.cuda.empty_cache()
        with ch.no_grad():
            for data in loader:
                images, _, _ = data
                images = images.cuda()
                print(m(images).to(ch.device('cpu')).numpy().size)
models_1 = get_models(get_model_folder_path(
        "adv", '0.5'), 100 // 2)
get_preds(loader,models_1)
    