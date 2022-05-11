import utils
from data_utils import SUPPORTED_PROPERTIES
from model_utils import load_model, get_models_path, get_model_representations,BASE_MODELS_DIR, save_model
import argparse
import numpy as np
import torch as ch
import torch.nn as nn
import os
import matplotlib as mpl
FILTER = 'two_attr'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--start_n', type=int, default=0,
                        help="Only consider starting from this layer")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Use only first N layers' parameters")
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--md_0', help='ratios to use for D_0')
    parser.add_argument('--mtrg', default=None, help='target ratios')
    parser.add_argument('--d_0', help='ratios to use for D_0')
    parser.add_argument('--trg', default=None, help='target ratios')
    binary = True
    args = parser.parse_args()
    tg = args.trg
    ptest , ptest_label ,_= get_model_representations(
        get_models_path(FILTER, "victim", args.md_0), 1, args.first_n)
    ntest , ntest_label ,_= get_model_representations(
        get_models_path(FILTER, "victim", args.mtrg), 1, args.first_n)
    X_te = np.concatenate((ptest, ntest))
    Y_te = ch.cat((ptest_label, ntest_label)).cuda()
    X_te = utils.prepare_batched_data(X_te)
    Y_te = Y_te.float()
    def acc_fn(x, y):
        if binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))
    
    acc = []
    folder_path =  os.path.join(BASE_MODELS_DIR,args.filter, "meta_model","-".join([args.d_0,str(args.start_n),str(args.first_n)]),tg)
    models_in_folder = os.listdir( folder_path)
    for path in models_in_folder:
        clf = load_model(os.path.join(folder_path, path))
        tacc,_ = utils.test_meta(clf,nn.BCEWithLogitsLoss(),X_te,Y_te,1000,acc_fn,binary=binary,regression=False,gpu=True, combined=True)
        acc.append(tacc)
    print(acc)
    log_path=os.path.join(BASE_MODELS_DIR,args.filter, "meta_on_two_attr",'-'.join(['vs'.join([args.d_0,tg]),str(args.start_n),str(args.first_n)]))
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path,'vs'.join([args.md_0,args.mtrg])),"w") as wr:
        wr.write(str(acc))
