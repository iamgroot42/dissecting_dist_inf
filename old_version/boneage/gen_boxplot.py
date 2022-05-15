import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils, log
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio',default = '0.5',
                        help='test ratio')
    args = parser.parse_args()
    flash_utils(args)

    #title = PROPERTY_FOCUS[args.filter]
    data = []
    columns = [
        'Proportion of training data',
        "Accuracy (%)",
        "method",
        
    ]
    ran_data = [
    [97, 96.89, 96.85],
    [98.2, 98.05, 98.65],
    [55.8, 54.6, 53.75],
    [57.95, 57.49, 58.2],
    [83.95, 83.25, 83.05],
    [ 74.65, 73.4, 57.05]
]
    meta_data = [
        [99.55, 99.7, 99.7, 99.55, 99.9, 99.95, 99.7, 98.05, 99.95, 99.8],
        [94.7, 95.7, 93.45, 92.4, 95.35, 97.25, 96.6, 94.25, 95.9, 98.0],
        [70.95, 63.85, 63.25, 69.75, 59.2, 66.3, 74.6, 63.4, 69.45, 65.55],
        [50.0, 58.9, 61.55, 67.0, 58.55, 62.15, 59.85, 58.75, 62.65, 65.65],
        [70.2, 90.3, 75.7, 83.75, 86.0, 89.45, 85.1, 83.7, 78.35, 72.35],
        [99.45, 99.2, 98.75, 98.4, 97.1, 92.05, 96.0, 96.75, 96.6, 97.35]
    ]
    log_path = os.path.join('./log',"perf_perpoint:"+args.ratio)
    in_folder = os.listdir(log_path)
    in_folder.sort()
    for p in in_folder:
        with open(os.path.join(log_path,p),'r') as f:
            ls = f.readlines()
            lst = []
            for l in ls:
                l = l.split(':')[1].strip()[1:-1]
                acc = [float(x) for x in l.split(' ') if x!='']
                lst.append(acc)
            for i in lst[0]:
                data.append([p,i,'Selective confidence']) 
    tar = [0.2,0.3,0.4,0.6,0.7,0.8]
    tar = [str(x) for x in tar]
    for j in range(len(tar)):
       
        for r in ran_data[j]:

            data.append([tar[j],r,"step random"])
        for m in meta_data[j]:
            data.append([tar[j],m,"meta classifier"])
    df = pd.DataFrame(data,columns=columns)
    sns_plot = sns.boxplot(x=columns[0], y=columns[1],
            #color = 'C0',
            hue = columns[2],
            data=df)
    sns_plot.set(ylim=(35, 101))
    """lower, upper = plt.gca().get_xlim()
    targets_scaled = range(int((upper - lower)))
    dran = [99.4
,94.15027736
,85.49618321
,66.46556058
,50.78401619
,55.5272542
,69.43187531
,78.3
,87.89260385
,59.1

]
    dfix = [97.9
,86.73726677
,73.7913486
,63.65007541
,51.54274153
,53.74426898
,60.53293112
,None
,73.10030395
,62.8


]
    plt.plot(targets_scaled, dran, color='C1', marker='x', linestyle='--')
    plt.plot(targets_scaled, dfix, color='C2', marker='x', linestyle='--')
    c_patch = mpatches.Patch(color='C0', label=r'Perpoint confidence')
    ran_patch = mpatches.Patch(color='C1', label=r'step random')
    fix_patch = mpatches.Patch(color='C2', label=r'step fixed')
    plt.legend(handles=[c_patch,ran_patch, fix_patch])
    """
    sns_plot.figure.savefig("./images/boneage_gen")


