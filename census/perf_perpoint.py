from tkinter import Y
from model_utils import get_models_path,  BASE_MODELS_DIR
from data_utils import CensusTwo, CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_pred, find_threshold_pred, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from perf_quart import get_models
mpl.rcParams['figure.dpi'] = 200


def get_preds(x,ms):
    
    ps = []
    for m in tqdm(ms):
        p=m.predict_proba(x)[:,1]
        ps.append(p)
    return np.array(ps)


def order_points(p1s,p2s):
    abs_dif = np.absolute(np.sum(p1s,axis=0)-np.sum(p2s,axis=0))
    inds = np.argsort(abs_dif)
       
    return inds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    args = parser.parse_args()
    flash_utils(args)
    lst = [0.05,0.1,0.2,0.3,0.4,0.5,1.0] #ratio of data points to try
    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))
    each_thre = []
    each_adv = []
    avg_thre = []
    
    for _ in tqdm(range(args.tries)):
        thresholds = []
        adv_thresholds = []
        # Load adv models
        total_models = 100
        models_1 = get_models(get_models_path(
            args.filter, "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(get_models_path(
            args.filter, "adv", args.ratio_2), total_models // 2)

        if args.filter == "two_attr":
            ds_1 = CensusTwo()
            ds_2 = CensusTwo()
            [r11,r12] = args.ratio_1.split(',')
            r11,r12 = float(r11),float(r12)
            [r21,r22] = args.ratio_2.split(',')
            r21,r22 = float(r21),float(r22)
            _, (x_te_1, y_te_1), _ = ds_1.get_data('adv',r11,r12)
            _, (x_te_2, y_te_2), _ = ds_2.get_data('adv',r21,r22)
        else:
            # Prepare data wrappers
            ds_1 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_1), split="adv")
            ds_2 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_2), split="adv")

        # Fetch test data from both ratios
            _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
            _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
        #y_te_1 = y_te_1.ravel()
        #y_te_2 = y_te_2.ravel()
        p1 = [get_preds(x_te_1,models_1), get_preds(x_te_2,models_1)]
        p2 = [get_preds(x_te_1,models_2), get_preds(x_te_2,models_2)]
        pv1 = [get_preds(x_te_1,models_victim_1), get_preds(x_te_2,models_victim_1)]
        pv2 = [get_preds(x_te_1,models_victim_2), get_preds(x_te_2,models_victim_2)]
        ord = (order_points(p1[0],p2[0]),order_points(p1[1],p2[1]))
        for i in range(2):
            p1[i] = np.transpose(p1[i])[ord[i]][::-1]
            p2[i] = np.transpose(p2[i])[ord[i]][::-1]
            pv1[i] = np.transpose(pv1[i])[ord[i]][::-1]
            pv2[i] = np.transpose(pv2[i])[ord[i]][::-1]
        thres, rs = [],[]
        for j in range(2):
            _,threshold, rule = find_threshold_pred(
                # accs_1, accs_2, granularity=0.01)
            p1[j], p2[j], granularity=0.005)
            
            thres.append(threshold)
            rs.append(rule)
        for ratio in lst:
            f_accs = []
            
            adv_accs = []
             #tr,rl = [],[]
            
            
            for j in range(2):
            #get accuracies
                
                leng = int(ratio*p1[j].shape[0])
                
                cm = np.concatenate((p1[j][:leng], p2[j][:leng]),axis=1)
                cl = np.concatenate((
                np.zeros(p1[j].shape[1]), np.ones(p2[j].shape[1])))
                adv_accs.append(100 * get_threshold_pred(cm, cl, thres[j][:leng], rs[j][:leng]))
           
                
            # Threshold based on adv models
                combined = np.concatenate((pv1[j][:leng], pv2[j][:leng]),axis=1)
                classes = np.concatenate((
                np.zeros(pv1[j].shape[1]), np.ones(pv2[j].shape[1])))
                specific_acc = get_threshold_pred(
                combined, classes, thres[j][:leng], rs[j][:leng])
                
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
                f_accs.append(100 * specific_acc)


            ind = np.argmax(adv_accs)
            thresholds.append(f_accs[ind])
            adv_thresholds.append(adv_accs[ind])
        each_adv.append(adv_thresholds)
        each_thre.append(thresholds)
    each_adv = np.array(each_adv)
    each_thre = np.array(each_thre)
    avg_thre = np.mean(each_adv[:,:-1],axis=0)
    best= np.argmax(avg_thre)
    content = 'At {}, best thresholds accuracy: {}\nAt {}, thresholds accuracy: {}'.format(lst[best],each_thre[:,best],1.0,each_thre[:,-1])
    print(content)
    log_path = os.path.join(BASE_MODELS_DIR, args.filter,"perf_perpoint:"+args.ratio_1)
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
      
    
   
    
