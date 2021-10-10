import pandas as pd
import seaborn as sns
import argparse
import os
from utils import flash_utils
import numpy as np
from model_utils import BASE_MODELS_DIR
from data_utils import PROPERTY_FOCUS, SUPPORTED_PROPERTIES
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams['figure.dpi'] = 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_0')
    parser.add_argument('--ratio')
    args = parser.parse_args()
    flash_utils(args)
    data = []
    columns = [
        'results of {} vs {} model'.format(args.d_0,args.ratio),
        "Accuracy (%)"
    ]

    d_0 = args.d_0.split(',')
    ratio = args.ratio.split(',')

    def get_dt(f):
        thresholds = []
        if f == 'race':
            i = 1
        else:
            i=0
        with open(os.path.join(BASE_MODELS_DIR,f,'baseline_on_dif','vs'.join([d_0[i],ratio[i]]),'two_attr:{}vs{}'.format(args.d_0,args.ratio)),'r') as r_b:                
            line = r_b.readline()[1:-1]
            t_s = line.split(',')
            thresholds=[float(x) for x in t_s]
            return thresholds
    m_t = []
    with open(os.path.join(BASE_MODELS_DIR,'two_attr',"baseline_result:{}".format(args.d_0),args.ratio),"r") as r_b:
                line = r_b.readline()
                re = line.split("; ")
                [_,t_l] = re[1].split(":")
                t_s = t_l.split(",")
                m_t=[float(x) for x in t_s]
    for i in m_t:
        data.append(['two_attr threshold',i])
    m_ts = []
    with open(os.path.join(BASE_MODELS_DIR,'two_attr',"baseline_on_dif",'vs'.join([args.d_0,args.ratio]),'sex:{}vs{}'.format(d_0[0],ratio[0])),"r") as r_b:
        line = r_b.readline()[1:-1]
        t_s = line.split(',')
        m_ts=[float(x) for x in t_s]
    for i in m_ts:
        data.append(['two_attr model on sex data threshold',i])
    m_tr = []
    with open(os.path.join(BASE_MODELS_DIR,'two_attr',"baseline_on_dif",'vs'.join([args.d_0,args.ratio]),'race:{}vs{}'.format(d_0[1],ratio[1])),"r") as r_b:
        line = r_b.readline()[1:-1]
        t_s = line.split(',')
        m_tr=[float(x) for x in t_s]
    for i in m_tr:
        data.append(['two_attr model on race data threshold',i])
    for i in get_dt('sex'):
        data.append(['sex model threshold',i])
    for i in get_dt('race'):
        data.append(['race model threshold',i])
    
    meta = []
    folder_path = os.path.join(BASE_MODELS_DIR,'two_attr','meta_result')
    file_name = os.listdir(folder_path)
    log_path = ''
    for p in file_name:
        if p.startswith('two_attr-{}'.format(args.d_0)):
            log_path=p
            break
    log_path = os.path.join(folder_path,log_path)
    with open(log_path,"r") as lg:
        lines = lg.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if line.startswith(args.ratio):
                l = line.split(':')[1]
                meta=[float(i) for i in l.split(",")]
                break
    for i in meta:
        data.append(['two_attr meta',i])

    meta_s = []
    folder_path = os.path.join(BASE_MODELS_DIR,'sex','meta_on_two_attr')
    mp = os.listdir(folder_path)
    for p in mp:
        if p.startswith('vs'.join([d_0[0],ratio[0]])):
            file_name = os.listdir(os.path.join(folder_path,p))
            for fn in file_name:
                if fn.startswith('vs'.join([args.d_0,args.ratio])):
                    with open(os.path.join(folder_path,p,fn),'r') as f:
                        line = f.read()
                        line = line.strip()[1:-1]
                        meta_s=[float(i) for i in line.split(',')]
    for i in meta_s:
        data.append(['sex meta on two_attr',i])


    meta_r = []
    folder_path = os.path.join(BASE_MODELS_DIR,'race','meta_on_two_attr')
    mp = os.listdir(folder_path)
    for p in mp:
        if p.startswith('vs'.join([d_0[1],ratio[1]])):
            file_name = os.listdir(os.path.join(folder_path,p))
            for fn in file_name:
                if fn.startswith('vs'.join([args.d_0,args.ratio])):
                    with open(os.path.join(folder_path,p,fn),'r') as f:
                        line = f.read()
                        line=line.strip()[1:-1]
                        meta_r=[float(i) for i in line.split(',')]

    for i in meta_r:
        data.append(['race meta on two_attr',i])
    

    df = pd.DataFrame(data, columns=columns)
    fig, ax = plt.subplots(figsize=(25, 15))
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df,ax=ax, color='C0', showfliers=False,)
    sns_plot.set(ylim=(10, 101))
    sns_plot.figure.savefig("./tests_{}vs{}.png".format(args.d_0,args.ratio))

    
