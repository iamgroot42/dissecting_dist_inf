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
ch.cuda.set_device(ch.cuda.device('cuda:2'))
mpl.rcParams['figure.dpi'] = 200
def get_preds(loader,ms):
    
    ps = []
    for m in tqdm(ms):
        m=m.cuda()
        m.eval()
        p=[]
        ch.cuda.empty_cache()
        with ch.no_grad():
            for data in loader:
                images, _, _ = data
                images = images.cuda()
                p.append(m(images).to(ch.device('cpu')).numpy())
        p = np.concatenate(p)
        ps.append(p)
    return np.squeeze(np.array(ps))


def order_points(p1s,p2s):
    abs_dif = np.absolute(np.sum(p1s,axis=0)-np.sum(p2s,axis=0))
    inds = np.argsort(abs_dif)
       
    return inds

def cal_acc(p,y):
    outputs = (p>=0).astype('int')
    return np.average((outputs==np.repeat(np.expand_dims(y,axis=1),outputs.shape[1],axis=1)).astype(int),axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256*32)
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    parser.add_argument('--loss_t', action='store_true',
                        help="use loss test as well") 
    args = parser.parse_args()
    flash_utils(args)
    lst = [0.05,0.1,0.2,0.3,0.4,0.5,1.0] #ratio of data points to try
    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    
    # Load victim models
    models_victim_1 = get_models(get_model_folder_path("victim", args.ratio_1))
    models_victim_2 = get_models(get_model_folder_path("victim", args.ratio_2))
    each_thre = []
    each_adv = []
    avg_thre = []
    avgb = []
    for _ in range(args.tries):
        df_1 = heuristic(
        df_val, filter, float(args.ratio_1),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

        df_2 = heuristic(
        df_val, filter, float(args.ratio_2),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    # Prepare data loaders
        ds_1 = BoneWrapper(
        df_1, df_1, features=features)
        ds_2 = BoneWrapper(
        df_2, df_2, features=features)
        loaders = [
        ds_1.get_loaders(args.batch_size, shuffle=False)[1],
        ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]
        
        thresholds = []
        adv_thresholds = []
        basic = []
        # Load adv models
        total_models = 100
        models_1 = get_models(get_model_folder_path(
        "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(get_model_folder_path(
        "adv", args.ratio_2), total_models // 2)

        yg = []
        for i in range(2):
            yl = []
            for data in loaders[i]:
                _,y,_ = data
                yl.append(y.to(ch.device('cpu')).numpy())
            yl = np.concatenate(yl)
            yg.append(yl.astype('int'))
        
        
        p1 = [get_preds(loaders[0],models_1), get_preds(loaders[1],models_1)]
        p2 = [get_preds(loaders[0],models_2), get_preds(loaders[1],models_2)]
        pv1 = [get_preds(loaders[0],models_victim_1), get_preds(loaders[1],models_victim_1)]
        pv2 = [get_preds(loaders[0],models_victim_2), get_preds(loaders[1],models_victim_2)]
        ord = (order_points(p1[0],p2[0]),order_points(p1[1],p2[1]))
        for i in range(2):
            p1[i] = np.transpose(p1[i])[ord[i]][::-1]
            p2[i] = np.transpose(p2[i])[ord[i]][::-1]
            pv1[i] = np.transpose(pv1[i])[ord[i]][::-1]
            pv2[i] = np.transpose(pv2[i])[ord[i]][::-1]
            yg[i] = yg[i][ord[i]][::-1]
        for ratio in lst:
            f_accs = []
            allaccs_1, allaccs_2 = [], []
            adv_accs = []
             #tr,rl = [],[]
            
            
            for j in range(2):
            #get accuracies
                

                leng = int(ratio*p1[j].shape[0])
                accs_1 = cal_acc(p1[j][:leng],yg[j][:leng])
                accs_2 = cal_acc(p2[j][:leng],yg[j][:leng])

            # Look at [0, 100]
                accs_1 *= 100
                accs_2 *= 100

                tracc, threshold, rule = find_threshold_acc(
                # accs_1, accs_2, granularity=0.01)
                accs_1, accs_2,granularity=0.005)
                adv_accs.append(100 * tracc)
           # tr.append(threshold)
           # rl.append(rule)
            # Compute accuracies on this data for victim
                accs_victim_1 = cal_acc(pv1[j][:leng],yg[j][:leng])
                accs_victim_2 = cal_acc(pv2[j][:leng],yg[j][:leng])

            # Look at [0, 100]
                accs_victim_1 *= 100
                accs_victim_2 *= 100
                allaccs_1.append(accs_victim_1)
                allaccs_2.append(accs_victim_2)
            # Threshold based on adv models
                combined = np.concatenate((accs_victim_1, accs_victim_2))
                classes = np.concatenate(
                (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
                specific_acc = get_threshold_acc(
                combined, classes, threshold, rule)
                
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
                f_accs.append(100 * specific_acc)


            ind = np.argmax(adv_accs)
            thresholds.append(f_accs[ind])
            adv_thresholds.append(adv_accs[ind])
            allaccs_1 = np.array(allaccs_1)
            allaccs_2 = np.array(allaccs_2)
            preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
            preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])
            basic.append(100*(np.mean(preds_1) + np.mean(preds_2)) / 2)
        each_adv.append(adv_thresholds)
        each_thre.append(thresholds)
        avgb.append(basic)
    avgb = np.array(avgb)
    each_adv = np.array(each_adv)
    each_thre = np.array(each_thre)
    avg_thre = np.mean(each_adv[:,:-1],axis=0)
    avgb = np.mean(avgb,axis=0)
    bestl = np.argmax(avgb[:-1])
    best= np.argmax(avg_thre)
    content = 'At {}, best thresholds accuracy: {}\nAt {}, thresholds accuracy: {}'.format(lst[best],each_thre[:,best],1.0,each_thre[:,-1])
    print(content)
    log_path = os.path.join('./log',"perf_quart:"+args.ratio_1)
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    log_path = os.path.join('./log',"selective_loss:"+args.ratio_1)
    if args.loss_t:
        cl = 'At {}, best basline accuracy: {}\nAt {}, baseline accuracy: {}'.format(lst[bestl],avgb[bestl],1.0,avgb[-1])
        print(cl)
        if not os.path.isdir(log_path):
             os.makedirs(log_path)
        with open(os.path.join(log_path,args.ratio_2),"w") as wr:
            wr.write(cl)
      
    
   
    
