import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils, log
import numpy as np
import matplotlib.patches as mpatches
from data_utils import SUPPORTED_PROPERTIES
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', default = 0.5,
                        help='test ratio')
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)                    
    args = parser.parse_args()
    flash_utils(args)

    #title = PROPERTY_FOCUS[args.filter]
    data = []
    columns = [
        'proportion of training data',
        "Accuracy (%)",
        "Selective loss test on quartiles"
    ]
    log_path = os.path.join('./log',"selective_loss_{}:{}".format(args.filter,args.ratio))
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            ls = f.readlines()
            lst = []
            for l in ls:
                l = l.split(':')[1].strip()
                lst.append(float(l))
            
            data.append([p,lst[0],True]) 
            data.append([p,lst[1],False])  
    df = pd.DataFrame(data,columns=columns)
    sns_plot = sns.lineplot(x=columns[0], y=columns[1],
            hue=columns[2], 
            data=df)
    sns_plot.set(ylim=(35, 101))
    sns_plot.figure.savefig("./images/celeba_loss_{}".format(args.filter))


