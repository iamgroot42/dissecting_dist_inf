from tkinter import Y
from model_utils import get_model,  BASE_MODELS_DIR
from data_utils import CelebaWrapper,  SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import torch as ch
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc,get_threshold_pred, find_threshold_pred, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from model_utils import get_model
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
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--task', default="Smiling",
                        choices=SUPPORTED_PROPERTIES,
                        help='task to focus on')
    parser.add_argument('--gpu', type=int,
                        default=0, help="device number")
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    flash_utils(args)
    total_models = 100
    # Get victim models
    ch.cuda.set_device(args.gpu)
    ds_1 = CelebaWrapper(args.filter, float(
        args.ratio_1), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)
    ds_2 = CelebaWrapper(args.filter, float(
        args.ratio_2), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)

    # Get loaders
    loaders = [
    ds_1.get_loaders(args.batch_size, shuffle=False)[1],
    ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]
    print("Loading models")
    models_victim_1 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.filter, args.ratio_1))
    pvs1 = [get_preds(loaders[0],models_victim_1), get_preds(loaders[1],models_victim_1)]
    del models_victim_1
    ch.cuda.empty_cache()
    models_victim_2 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.filter, args.ratio_2))
    #pvs1 = [get_preds(loaders[0],models_victim_1), get_preds(loaders[1],models_victim_1)]
    pvs2 = [get_preds(loaders[0],models_victim_2), get_preds(loaders[1],models_victim_2)]

    del models_victim_2
    ch.cuda.empty_cache()
        # Load adv models
    models_1 = get_models(os.path.join(
    BASE_MODELS_DIR, "adv", args.filter, args.ratio_1), total_models // 2)
    models_2 = get_models(os.path.join(
    BASE_MODELS_DIR, "adv", args.filter, args.ratio_2), total_models // 2)
        

    p1 = [get_preds(loaders[0],models_1), get_preds(loaders[1],models_1)]
    p2 = [get_preds(loaders[0],models_2), get_preds(loaders[1],models_2)]
        
    ord = (order_points(p1[0],p2[0]),order_points(p1[1],p2[1]))
    ordv = (order_points(pvs1[0],pvs2[0]),order_points(pvs1[1],pvs2[1]))
    
    plt.plot(ordv[1],ord[1],'.')
    plt.xlabel('victim ranking')
    plt.ylabel('Adv ranking')
    plt.title('Rankings of adv vs victim')
    plt.savefig('./images/rank_{}_{}vs{}.png'.format(args.filter,args.ratio_1,args.ratio_2))
