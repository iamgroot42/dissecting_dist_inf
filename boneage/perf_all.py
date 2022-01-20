from model_utils import load_model, get_model_folder_path
from data_utils import BoneWrapper, get_df, get_features
import torch.nn as nn
import torch as ch
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc,get_threshold_pred, find_threshold_pred, flash_utils,heuristic
import matplotlib.pyplot as plt
import matplotlib as mpl
from perf_tests import get_models
from perf_quart import order_points, cal_acc
mpl.rcParams['figure.dpi'] = 200
def get_preds(loader,ms):
    
    ps = []
    inp = []
    for data in loader:
        images, _, _ = data
        inp.append(images.cuda())
    for m in tqdm(ms):
        m=m.cuda()
        m.eval()
        p=[]
        
        ch.cuda.empty_cache()
        with ch.no_grad():
            for images in inp:
                #p.append(m(images).detach()[:,0].to(ch.device('cpu')).numpy())
                p.append(m(images).detach()[:, 0])
        p = ch.cat(p)
        #p = np.concatenate(p)
        
        ps.append(p)
    #ps = np.array(ps)
    ps = ch.stack(ps,0)
    return ps.to(ch.device('cpu')).numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256*32)
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    parser.add_argument('--gpu', type=int,
                        default=0, help="device number")                    
    args = parser.parse_args()
    flash_utils(args)
    ch.cuda.set_device(args.gpu)
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
    each_threq = []
    each_advq = []
    avg_threq = []
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
        ygs = []
        for i in range(2):
            yl = []
            for data in loaders[i]:
                _,y,_ = data
                yl.append(y.to(ch.device('cpu')).numpy())
            yl = np.concatenate(yl)
            ygs.append(yl)
        thresholds = []
        adv_thresholds = []
        thresholdsq = []
        adv_thresholdsq = []
        basic = []
        # Load adv models
        total_models = 100
        models_1 = get_models(get_model_folder_path(
        "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(get_model_folder_path(
        "adv", args.ratio_2), total_models // 2)

        
        yg = [[],[]]
        p1 = [get_preds(loaders[0],models_1), get_preds(loaders[1],models_1)]
        del models_1
        ch.cuda.empty_cache()
        p2 = [get_preds(loaders[0],models_2), get_preds(loaders[1],models_2)]
        del models_2
        ch.cuda.empty_cache()
        pv1 = [get_preds(loaders[0],models_victim_1), get_preds(loaders[1],models_victim_1)]
        pv2 = [get_preds(loaders[0],models_victim_2), get_preds(loaders[1],models_victim_2)]
        ord = (order_points(p1[0],p2[0]),order_points(p1[1],p2[1]))
        for i in range(2):
            p1[i] = np.transpose(p1[i])[ord[i]][::-1]
            p2[i] = np.transpose(p2[i])[ord[i]][::-1]
            pv1[i] = np.transpose(pv1[i])[ord[i]][::-1]
            pv2[i] = np.transpose(pv2[i])[ord[i]][::-1]
            yg[i] = ygs[i][ord[i]][::-1]
        thres, rs = [],[]
        for j in range(2):
            _,threshold, rule = find_threshold_pred(
                # accs_1, accs_2, granularity=0.01)
            p1[j], p2[j], granularity=0.005)
            
            thres.append(threshold)
            rs.append(rule) 
        for ratio in lst:
            f_accs = []
            
            allaccs_1, allaccs_2 = [], []
            adv_accs = []
             #tr,rl = [],[]
            
            f_accsq = []
            
            adv_accsq = []
            
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
                adv_accsq.append(100 * tracc)
                cm = np.concatenate((p1[j][:leng], p2[j][:leng]),axis=1)
                cl = np.concatenate((
                np.zeros(p1[j].shape[1]), np.ones(p2[j].shape[1])))
                adv_accs.append(100 * get_threshold_pred(cm, cl, thres[j][:leng], rs[j][:leng]))
           
                combined = np.concatenate((pv1[j][:leng], pv2[j][:leng]),axis=1)
                classes = np.concatenate((
                np.zeros(pv1[j].shape[1]), np.ones(pv2[j].shape[1])))
                specific_acc = get_threshold_pred(
                combined, classes, thres[j][:leng], rs[j][:leng])
                
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
                accs_victim_1 = cal_acc(pv1[j][:leng],yg[j][:leng])
                accs_victim_2 = cal_acc(pv2[j][:leng],yg[j][:leng])

            # Look at [0, 100]
                accs_victim_1 *= 100
                accs_victim_2 *= 100
                allaccs_1.append(accs_victim_1)
                allaccs_2.append(accs_victim_2)
                f_accs.append(100 * specific_acc)
                combined = np.concatenate((accs_victim_1, accs_victim_2))
                classes = np.concatenate(
                (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
                specific_acc = get_threshold_acc(
                combined, classes, threshold, rule)
                
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
                f_accsq.append(100 * specific_acc)

            ind = np.argmax(adv_accs)
            thresholds.append(f_accs[ind])
            adv_thresholds.append(adv_accs[ind])
            indq = np.argmax(adv_accsq)
            thresholdsq.append(f_accsq[indq])
            adv_thresholdsq.append(adv_accsq[indq])
            allaccs_1 = np.array(allaccs_1)
            allaccs_2 = np.array(allaccs_2)
            preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
            preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])
            basic.append(100*(np.mean(preds_1) + np.mean(preds_2)) / 2)
        each_adv.append(adv_thresholds)
        each_thre.append(thresholds)
        each_advq.append(adv_thresholdsq)
        each_threq.append(thresholdsq)
        avgb.append(basic)
    avgb = np.array(avgb)
    each_adv = np.array(each_adv)
    each_thre = np.array(each_thre)
    avg_thre = np.mean(each_adv[:,:-1],axis=0)
    best= np.argmax(avg_thre)
    each_advq = np.array(each_advq)
    each_threq = np.array(each_threq)
    avg_threq = np.mean(each_advq[:,:-1],axis=0)
    bestq= np.argmax(avg_threq)
    avgb = np.mean(avgb,axis=0)
    bestl = np.argmax(avgb[:-1])
    content = 'At {}, best perpoint thresholds accuracy: {}\nAt {}, perpoint thresholds accuracy: {}'.format(lst[best],each_thre[:,best],1.0,each_thre[:,-1])
    print(content)
    log_path = os.path.join('./log',"perf_perpoint:{}".format(args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    log_path = os.path.join('./log',"selective_loss:{}".format(args.ratio_1))
    cl = 'At {}, best basline accuracy: {}\nAt {}, baseline accuracy: {}'.format(lst[bestl],avgb[bestl],1.0,avgb[-1])
    print(cl)
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(cl)
    content = 'At {}, best thresholds accuracy: {}\nAt {}, thresholds accuracy: {}'.format(lst[bestq],each_threq[:,bestq],1.0,each_threq[:,-1])
    print(content)
    log_path = os.path.join('./log',"perf_quart:{}".format(args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    
   
    

      
    
   
    
