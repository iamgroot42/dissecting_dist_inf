import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils, log
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
from data_utils import SUPPORTED_PROPERTIES
#mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio',default = '0.5',
                        help='test ratio')
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)                    
    args = parser.parse_args()
    flash_utils(args)

    #title = PROPERTY_FOCUS[args.filter]
    data = []
    columns = [
        '{} proportion of training data'.format(args.filter),
        "Accuracy (%)",
        "method",
        
    ]
    fix_data = [ [96.1, 97.15],
[84.42, 84.83],
[73.28, 72.93, 75.06],
[61.74, 60.68],
[54.83, 52.81, 53.82],
[54.51, 53.69, 54.15],
[60.73, 61.14],
[63.5, 64.35],
[79.84, 71.43],
[66.15, 71.55]]
    ran_data = [
    [97.95, 97.45, 97.45],
    [89.72, 88.70, 88.70],
    [78.88, 79.08, 79.95],
    [64.10, 61.49, 58.92],
    [53.62, 55.08, 53.87],
    [52.57, 54.15],
    [61.45, 59.68, 59.23],
    [65.4, 62.25, 63.35],
    [75.18, 64.34, 64.94],
    [64.8, 66.45, 69.85 ]
]
    
    log_path = os.path.join('./log',"perf_perpoint_{}:{}".format(args.filter,args.ratio))
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
                data.append([p,i,"Perpoint confidence"])  
    tar = [0.0,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0]
    tar = [str(x) for x in tar]
    for j in range(len(tar)):
       
        for r in ran_data[j]:

            data.append([tar[j],r,"step random"])
        for f in fix_data[j]:
            data.append([tar[j],f,"step fixed"])
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
    sns_plot.figure.savefig("./images/celeba_50m_{}_quartiles".format(args.filter))


