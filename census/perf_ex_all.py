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
from perf_all import get_preds,order_points,cal_acc
mpl.rcParams['figure.dpi'] = 200
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
    exr = [x/1000 for x in range(5,255,5)]
    exr.append(0.0)
    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))
    each_thre = []
    each_adv = []
    avg_thre = []
    each_threq = []
    each_advq = []
    avg_threq = []
    avgb = []
    for _ in tqdm(range(args.tries)):
        thresholds = []
        adv_thresholds = []
        thresholdsq = []
        adv_thresholdsq = []
        basic = []
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
        yg = [y_te_1,y_te_2]
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
            yg[i] = yg[i][ord[i]][::-1]
        thres, rs = [],[]
        for j in range(2):
            _,threshold, rule = find_threshold_pred(
                # accs_1, accs_2, granularity=0.01)
            p1[j], p2[j], granularity=0.005)
            
            thres.append(threshold)
            rs.append(rule)
        for r in exr:
            ex_t = []
            ex_a =[]
            ex_tq=[]
            ex_aq=[]
            e_b=[]
            for ratio in lst:
                f_accs = []
            
                allaccs_1, allaccs_2 = [], []
                adv_accs = []
             #tr,rl = [],[]
            
                f_accsq = []
            
                adv_accsq = []
            
                for j in range(2):
            #get accuracies
                    exc = int(r*p1[j].shape[0])
                    leng = int(ratio*p1[j].shape[0])+exc
                
                    cm = np.concatenate((p1[j][exc:leng], p2[j][exc:leng]),axis=1)
                    cl = np.concatenate((
                    np.zeros(p1[j].shape[1]), np.ones(p2[j].shape[1])))
                    adv_accs.append(100 * get_threshold_pred(cm, cl, thres[j][exc:leng], rs[j][exc:leng]))
                    accs_1 = cal_acc(p1[j][exc:leng],yg[j][exc:leng])
                    accs_2 = cal_acc(p2[j][exc:leng],yg[j][exc:leng])

            # Look at [0, 100]
                    accs_1 *= 100
                    accs_2 *= 100
                    tracc, threshold, rule = find_threshold_acc(
                # accs_1, accs_2, granularity=0.01)
                    accs_1, accs_2,granularity=0.005)
                    adv_accsq.append(100 * tracc)
                    combined = np.concatenate((pv1[j][exc:leng], pv2[j][exc:leng]),axis=1)
                    classes = np.concatenate((
                    np.zeros(pv1[j].shape[1]), np.ones(pv2[j].shape[1])))
                    specific_acc = get_threshold_pred(
                    combined, classes, thres[j][exc:leng], rs[j][exc:leng])
                
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
                    f_accs.append(100 * specific_acc)
                    accs_victim_1 = cal_acc(pv1[j][exc:leng],yg[j][exc:leng])
                    accs_victim_2 = cal_acc(pv2[j][exc:leng],yg[j][exc:leng])

            # Look at [0, 100]
                    accs_victim_1 *= 100
                    accs_victim_2 *= 100
                    allaccs_1.append(accs_victim_1)
                    allaccs_2.append(accs_victim_2)
                    combined = np.concatenate((accs_victim_1, accs_victim_2))
                    classes = np.concatenate(
                (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
                    specific_acc = get_threshold_acc(
                combined, classes, threshold, rule)
                
              
                    f_accsq.append(100 * specific_acc)

                ind = np.argmax(adv_accs)
                ex_t.append(f_accs[ind])
                ex_a.append(adv_accs[ind])
                indq = np.argmax(adv_accsq)
                ex_tq.append(f_accsq[indq])
                ex_aq.append(adv_accsq[indq])
                allaccs_1 = np.array(allaccs_1)
                allaccs_2 = np.array(allaccs_2)
                preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
                preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])
                e_b.append(100*(np.mean(preds_1) + np.mean(preds_2)) / 2)
            thresholds.append(ex_t)
            adv_thresholds.append(ex_a)
            
            thresholdsq.append(ex_tq)
            adv_thresholdsq.append(ex_aq)
            basic.append(e_b)
            
        each_adv.append(adv_thresholds)
        each_thre.append(thresholds)
        each_advq.append(adv_thresholdsq)
        each_threq.append(thresholdsq)
        avgb.append(basic)
    avgb = np.array(avgb)
    each_adv = np.array(each_adv)
    each_thre = np.array(each_thre)
    avg_thre = np.mean(each_adv,axis=0)
    (bx,by)= np.unravel_index(np.argmax(avg_thre.flatten()[:-1]),shape=avg_thre.shape)
    each_advq = np.array(each_advq)
    each_threq = np.array(each_threq)
    avg_threq = np.mean(each_advq,axis=0)
    (bxq,byq)= np.unravel_index(np.argmax(avg_threq.flatten()[:-1]),shape=avg_threq.shape) 
    avgb = np.mean(avgb,axis=0)
    (bxl,byl)= np.unravel_index(np.argmax(avgb.flatten()[:-1]),shape=avgb.shape)
    print(np.array2string(avg_threq))
    print(np.array2string(np.mean(each_threq,axis=0)))
    content = 'Using {} after excluding top {}, best perpoint thresholds accuracy: {}\nAt {}, perpoint thresholds accuracy: {}'.format(lst[by],exr[bx],each_thre[:,bx,by],1.0,each_thre[:,-1,-1])
    print(content)
    log_path = os.path.join('./log','ex',"perf_perpoint_{}:{}".format(args.filter,args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    log_path = os.path.join('./log','ex',"selective_loss_{}:{}".format(args.filter,args.ratio_1))
    cl = 'Using {} after excluding top {}, best basline accuracy: {}\nAt {}, baseline accuracy: {}'.format(lst[byl],exr[bxl],avgb[bxl,byl],1.0,avgb[-1,-1])
    print(cl)
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(cl)
    content = 'Using {} after excluding top {}, best thresholds accuracy: {}\nAt {}, thresholds accuracy: {}'.format(lst[byq],exr[bxq],each_threq[:,bxq,byq],1.0,each_threq[:,-1,-1])
    print(content)
    log_path = os.path.join('./log','ex',"perf_quart_{}:{}".format(args.filter,args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    
