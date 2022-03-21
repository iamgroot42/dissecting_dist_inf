import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils
import numpy as np
from model_utils import BASE_MODELS_DIR
from data_utils import PROPERTY_FOCUS, SUPPORTED_PROPERTIES
import matplotlib.patches as mpatches
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 200
ATTACK=['loss','threshold','perpoint']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio', default = 0.5,
                        help='test ratio')
    parser.add_argument('--attack',choices=ATTACK,
                        required=True)
    args = parser.parse_args()
    flash_utils(args)

    title = "{} attack on {}".format(args.attack,PROPERTY_FOCUS[args.filter])
    data = []
    columns = [
        r'%s proportion of training data ($\alpha$)' % PROPERTY_FOCUS[args.filter],
        "Accuracy (%)",
        "Variant"
    ]
    if args.attack=='loss':
        apth="selective_loss_{}:{}".format(args.filter,args.ratio)
    elif args.attack=='threshold':
        apth="perf_quart_{}:{}".format(args.filter,args.ratio)
    elif args.attack=="perpoint":
        apth="perf_perpoint_{}:{}".format(args.filter,args.ratio)
    log_path = os.path.join('./log',apth)
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            l = f.readline()
            l = l.split(':')[1]
            lst=eval(l)
            for i in lst:
                data.append([p,i,'original']) 
    log_path = os.path.join('./log','drop',apth)
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            l = f.readline()
            l = l.split(':')[1]
            lst=eval(l)
            for i in lst:
                data.append([p,i,'drop'])  
    log_path = os.path.join('./log','sample_size_scale:{}'.format(str(0.1)),apth)
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            l = f.readline()
            l = l.split(':')[1]
            lst=eval(l)
            for i in lst:
                data.append([p,i,'sample size 0.1'])     
    log_path = os.path.join('./log','sample_size_scale:{}'.format(str(3.0)),apth)
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            l = f.readline()
            l = l.split(':')[1]
            lst=eval(l)
            for i in lst:
                data.append([p,i,'sample size 3.0'])           
    df = pd.DataFrame(data,columns=columns)
    sns_plot = sns.boxplot(x=columns[0], y=columns[1],
            hue=columns[2], 
            data=df)
    sns_plot.set(ylim=(35, 101))
    sns_plot.set(title=title)
    sns_plot.figure.savefig("./images/ncensus_{}_{}_{}.jpg".format(args.filter,args.ratio,args.attack))


