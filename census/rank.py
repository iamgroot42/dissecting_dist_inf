from tkinter import Y
from model_utils import get_models_path,  BASE_MODELS_DIR
from data_utils import CensusTwo, CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc,get_threshold_pred, find_threshold_pred, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from perf_quart import get_models
mpl.rcParams['figure.dpi'] = 200


def get_preds(x,ms):
    
    ps = []
    for m in tqdm(ms):
        p=m.predict_proba(x)[:,1]
        ps.append(p)
    return np.squeeze(np.array(ps))


def order_points(p1s,p2s):
    abs_dif = np.absolute(np.median(p1s,axis=0)-np.median(p2s,axis=0))
    inds = np.argsort(abs_dif)
       
    return inds

def cal_acc(p,y):
    outputs = (p>=0.5).astype('int')
    return np.average((outputs==np.squeeze(np.repeat(np.expand_dims(y,axis=1),outputs.shape[1],axis=1))).astype(int),axis=0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    args = parser.parse_args()
    flash_utils(args)

    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))
    total_models = 100
    models_1 = get_models(get_models_path(
            args.filter, "adv", args.ratio_1))
    models_2 = get_models(get_models_path(
            args.filter, "adv", args.ratio_2))
    ds_1 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_1), split="adv")
    ds_2 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_2), split="adv")
    _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
    _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
        #y_te_1 = y_te_1.ravel()
        #y_te_2 = y_te_2.ravel()
    yg = [y_te_1,y_te_2]
    pv1 = [get_preds(x_te_1,models_victim_1), get_preds(x_te_2,models_victim_1)]
    pv2 = [get_preds(x_te_1,models_victim_2), get_preds(x_te_2,models_victim_2)]
    p1 = [get_preds(x_te_1,models_1), get_preds(x_te_2,models_1)]
    p2 = [get_preds(x_te_1,models_2), get_preds(x_te_2,models_2)]
    ordv = (order_points(pv1[0],pv2[0]),order_points(pv1[1],pv2[1]))
    ord = [order_points(p1[0],p2[0]),order_points(p1[1],p2[1])]
    print(np.array2string(ordv[1])+'\n')
    print(np.array2string(ord[1]))
    plt.plot(ordv[1],ord[1],'.')
    plt.xlabel('victim ranking')
    plt.ylabel('Adv ranking')
    plt.title('Rankings of adv vs victim')
    plt.savefig('./images/rankm_{}_{}vs{}.png'.format(args.filter,args.ratio_1,args.ratio_2))
    for i in range(2):
        p1[i] = np.transpose(p1[i])[ord[i]][::-1]
        pv1[i] = np.transpose(pv1[i])[ord[i]][::-1]
        p2[i] = np.transpose(p2[i])[ord[i]][::-1]
        pv2[i] = np.transpose(pv2[i])[ord[i]][::-1]
    for j in range(2):
        for i in range(p1[0].shape[j]):
            fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
            axs[0,0].hist(p1[j][i])
            axs[0,0].set_title("adv models from {}".format(args.ratio_1))
            axs[1,0].hist(pv1[j][i])
            axs[0,1].hist(p2[j][i])
            axs[1,1].hist(pv2[j][i])
            axs[1,0].set_title("victim models from {}".format(args.ratio_1))
            axs[0,1].set_title("adv models from {}".format(args.ratio_2))
            axs[1,1].set_title("victim models from {}".format(args.ratio_2))
            fig.suptitle("Ground truth:{}".format(yg[j][i]))
            i_path = './images/hist/{}/{}:{}/{}'.format(args.filter,args.ratio_1,args.ratio_2,str(j))
            if not os.path.isdir(i_path):
                os.makedirs(i_path)
            fig.savefig(os.path.join(i_path,'{}.png'.format(str(i))))