from model_utils import load_model, get_models_path
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import perpoint_threshold_test,threshold_and_loss_test,flash_utils,ensure_dir_exists

def get_preds(x,ms):
    
    ps = []
    for m in tqdm(ms):
        p=m.predict_proba(x)[:,1]
        ps.append(p)
    return np.squeeze(np.array(ps))
def cal_acc(p,y):
    outputs = (p>=0.5).astype('int')
    return np.average((outputs==np.squeeze(np.repeat(np.expand_dims(y,axis=1),outputs.shape[1],axis=1))).astype(int),axis=0)

def get_models(folder_path, n_models=1000):
   
    paths = np.random.permutation(os.listdir(folder_path))
    i = 0
    models = []
    for mpath in tqdm(paths):
        if i>=n_models:
            break
        if os.path.isdir(os.path.join(folder_path, mpath)):
            continue
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
        i+=1
    return models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256*32)
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")  
    parser.add_argument('--drop', action="store_true")
    parser.add_argument('--scale',type=float,default=1.0)               
    args = parser.parse_args()
    flash_utils(args)
    ratios = [0.05,0.1,0.2,0.3,0.4,0.5,1.0] #ratio of data points to try
    
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1,drop=args.drop,scale=args.scale))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2,drop=args.drop,scale=args.scale))
    thre, perp,bas = [],[],[]
    for _ in range(args.tries):
        # Load adv models
        total_models = 100
        models_1 = get_models(get_models_path(
            args.filter, "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(get_models_path(
            args.filter, "adv", args.ratio_2), total_models // 2)
        ds_1 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_1), split="adv")
        ds_2 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_2), split="adv")

        # Fetch test data from both ratios
        _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
        _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
        p1 = [get_preds(x_te_1,models_1), get_preds(x_te_2,models_1)]
        p2 = [get_preds(x_te_1,models_2), get_preds(x_te_2,models_2)]
        pv1 = [get_preds(x_te_1,models_victim_1), get_preds(x_te_2,models_victim_1)]
        pv2 = [get_preds(x_te_1,models_victim_2), get_preds(x_te_2,models_victim_2)]
        (vpacc,_),_=perpoint_threshold_test( (p1, p2),
        (pv1, pv2),
        (y_te_1, y_te_2),
        ratios, granularity=0.005)
        perp.append(vpacc)
        (vacc,ba),_=threshold_and_loss_test( cal_acc
        ,(p1, p2),
        (pv1, pv2),
        (y_te_1, y_te_2),
        ratios, granularity=0.005)
        thre.append(vacc)
        bas.append(ba)
        

    l="./log"
    if args.scale!=1:
        l=os.path.join(l,'sample_size_scale:{}'.format(args.scale))
    if args.drop:
        l=os.path.join(l,'drop')
    content = 'Perpoint thresholds accuracy: {}'.format(perp)
    print(content)
    
    log_path = os.path.join(l,"perf_perpoint:{}".format(args.ratio_1))
    ensure_dir_exists(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    log_path = os.path.join(l,"selective_loss:{}".format(args.ratio_1))
    
    cl = 'basline accuracy: {}'.format(bas)
    print(cl)
    
    ensure_dir_exists(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(cl)
    
    content = 'thresholds accuracy: {}'.format(thre)
    print(content)
    
    log_path = os.path.join(l,"perf_quart:{}".format(args.ratio_1))
    ensure_dir_exists(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    
    
   
    

      
    
   
    
