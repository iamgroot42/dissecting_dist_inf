from tkinter import Y
from model_utils import get_models_path, load_model, BASE_MODELS_DIR
from data_utils import CensusTwo, CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_models(folder_path, n_models=1000):
   
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_pred(data, models):
    preds = []
    for model in tqdm(models):
        preds.append(model.predict(data))

    return np.transpose(np.array(preds))

def select_points(model1,model2,x,Y):
    models = list(zip(model1,model2))
    dims = x.shape
    total = dims[0]
    p1,p2 = np.zeros(total),np.zeros(total)
    for (m1, m2) in models:
        p1 += m1.predict_proba(x)[:,1]
        p2 += m2.predict_proba(x)[:,1]
    abs_dif = np.expand_dims(np.absolute(p1-p2),axis=1)
    da = np.append(x, Y, axis=1)
    da = np.append(da, abs_dif, axis=1)
    re = da[np.argsort(da[:, -1])][::-1]
    
    return re



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
    
    for _ in range(args.tries):
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
        re1 = select_points(models_1,models_2,x_te_1,y_te_1)
        re2 = select_points(models_1,models_2,x_te_2,y_te_2)
        x1,y1 = re1[:,:-2], re1[:,-2]
        x2,y2 = re2[:,:-2], re2[:,-2]
        yg = (y1,y2)
        p1 = (get_pred(x1,models_1), get_pred(x2,models_1))
        p2 = (get_pred(x1,models_2), get_pred(x2,models_2))
        pv1 = (get_pred(x1,models_victim_1), get_pred(x2,models_victim_1))
        pv2 = (get_pred(x1,models_victim_2), get_pred(x2,models_victim_2))
        
        if (p1[0].shape[0] != x1.shape[0]):
            print('wrong dimension')
            break
        for ratio in lst:
            f_accs = []
            
            adv_accs = []
             #tr,rl = [],[]
            
            
            for j in range(2):
            #get accuracies
                
                leng = int(ratio*p1[j].shape[0])
                accs_1 = np.average((p1[j][:leng]==np.repeat(np.expand_dims(yg[j][:leng],axis=1),p1[j].shape[1],axis=1)).astype(int),axis=0)
                accs_2 = np.average((p2[j][:leng]==np.repeat(np.expand_dims(yg[j][:leng],axis=1),p2[j].shape[1],axis=1)).astype(int),axis=0)

            # Look at [0, 100]
                accs_1 *= 100
                accs_2 *= 100

                tracc, threshold, rule = find_threshold_acc(
                # accs_1, accs_2, granularity=0.01)
                accs_1, accs_2, granularity=0.005)
                adv_accs.append(100 * tracc)
           # tr.append(threshold)
           # rl.append(rule)
            # Compute accuracies on this data for victim
                accs_victim_1 = np.average((pv1[j][:leng]==np.repeat(np.expand_dims(yg[j][:leng],axis=1),pv1[j].shape[1],axis=1)).astype(int),axis=0)
                accs_victim_2 = np.average((pv2[j][:leng]==np.repeat(np.expand_dims(yg[j][:leng],axis=1),pv2[j].shape[1],axis=1)).astype(int),axis=0)

            # Look at [0, 100]
                accs_victim_1 *= 100
                accs_victim_2 *= 100

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
        each_adv.append(adv_thresholds)
        each_thre.append(thresholds)
    each_adv = np.array(each_adv)
    each_thre = np.array(each_thre)
    avg_thre = np.mean(each_adv[:,:-1],axis=0)
    best= np.argmax(avg_thre)
    content = 'At {}, best thresholds accuracy: {}\nAt {}, thresholds accuracy: {}'.format(lst[best],each_thre[:,best],1.0,each_thre[:,-1])
    print(content)
    log_path = os.path.join(BASE_MODELS_DIR, args.filter,"perf_quart:"+args.ratio_1)
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
      
    
   
    
