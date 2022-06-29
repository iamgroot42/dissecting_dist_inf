import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# For ratio-based datasets
# targets = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

# for arXiv
targets = ["9", "10", "11", "12", "13", "14", "15", "16", "17"]

# for Boneage
# targets = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]

fill_data = np.zeros((len(targets), len(targets)))
mask = np.zeros((len(targets), len(targets)), dtype=bool)
annot_data = [[""] * len(targets) for _ in range(len(targets))]

for i in range(len(targets)):
    for j in range(len(targets)-(i+1)):
        mask[j+i+1][i] = True

# plt.style.use('dark_background')

plt.rcParams.update({'font.size': 8.5}) # for census
# plt.rcParams.update({'font.size': 10}) # for boneage
# plt.rcParams.update({'font.size': 8.5}) # for arxiv
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=11)

fill_data_other = np.zeros((len(targets), len(targets)))

# Census Meta
# fill_data[0][1:] = [100, 100, 99, 100, 100, 99, 100, 100, 100, 99]
# fill_data[1][2:] = [52, 59, 66, 73, 76, 78, 79, 83, 74]
# fill_data[2][3:] = [50, 56, 57, 64, 66, 65, 70, 83]
# fill_data[3][4:] = [50, 52, 59, 60, 59, 64, 79]
# fill_data[4][5:] = [49, 53, 55, 55, 62, 79]
# fill_data[5][6:] = [55, 62, 67, 70, 83]
# fill_data[6][7:] = [50, 52, 56, 77]
# fill_data[7][8:] = [51, 59, 76]
# fill_data[8][9:] = [55, 76]
# fill_data[9][10:] = [77]
# annot_data[0][1:] = [r'100 $\pm$ 0', r'100 $\pm$ 0', r'99 $\pm$ 0', r'100 $\pm$ 0', r'100 $\pm$ 0', r'99 $\pm$ 0', r'100 $\pm$ 0', r'100 $\pm$ 0', r'100 $\pm$ 0', r'99 $\pm$ 0']
# annot_data[1][2:] = [r'52 $\pm$ 0', r'59 $\pm$ 1', r'66 $\pm$ 1', r'73 $\pm$ 3', r'76 $\pm$ 1', r'78 $\pm$ 3', r'79 $\pm$ 3', r'83 $\pm$ 2', r'74 $\pm$ 5']
# annot_data[2][3:] = [r'50 $\pm$ 1', r'56 $\pm$ 1', r'57 $\pm$ 3', r'64 $\pm$ 1', r'66 $\pm$ 3', r'65 $\pm$ 3', r'70 $\pm$ 3', r'83 $\pm$ 13']
# annot_data[3][4:] = [r'50 $\pm$ 0', r'52 $\pm$ 2', r'59 $\pm$ 0', r'60 $\pm$ 3', r'59 $\pm$ 3', r'64 $\pm$ 3', r'79 $\pm$ 11']
# annot_data[4][5:] = [r'49 $\pm$ 2', r'53 $\pm$ 1', r'55 $\pm$ 1', r'55 $\pm$ 1', r'62 $\pm$ 2', r'79 $\pm$ 13']
# annot_data[5][6:] = [r'55 $\pm$ 3', r'62 $\pm$ 1', r'67 $\pm$ 0', r'70 $\pm$ 0', r'83 $\pm$ 7']
# annot_data[6][7:] = [r'50 $\pm$ 0', r'52 $\pm$ 1', r'56 $\pm$ 1', r'77 $\pm$ 13']
# annot_data[7][8:] = [r'51 $\pm$ 0', r'59 $\pm$ 1', r'76 $\pm$ 9']
# annot_data[8][9:] = [r'55 $\pm$ 0', r'76 $\pm$ 9']
# annot_data[9][10:] = [r'77 $\pm$ 11']
# fill_data_other[1][:1] = [800]
# fill_data_other[2][:2] = [800, 0.06]
# fill_data_other[3][:3] = [800, 0.21, 0.02]
# fill_data_other[4][:4] = [800, 0.34, 0.10, 0.00]
# fill_data_other[5][:5] = [800, 0.65, 0.17, 0.05, 0.02]
# fill_data_other[6][:6] = [800, 0.47, 0.16, 0.08, 0.04, 0.32]
# fill_data_other[7][:7] = [800, 0.51, 0.24, 0.12, 0.05, 0.27, 0.01]
# fill_data_other[8][:8] = [800, 0.42, 0.13, 0.11, 0.04, 0.30, 0.03, 0.02]
# fill_data_other[9][:9] = [800, 0.39, 0.17, 0.17, 0.14, 0.37, 0.10, 0.21, 0.15]
# fill_data_other[10][:10] = [800, 0.30, 800, 800, 800, 6.54, 800, 17.43, 800, 800]

# Census Race Regression (incomplete)
# fill_data[0][1:] = [100, 100, 99, 100, 100, 99, 100, 100, 100, 99]
# fill_data[1][2:] = [52, 59, 66, 73, 76, 78, 79, 83, 74]
# fill_data[2][3:] = [50, 56, 57, 64, 66, 65, 70, 83]
# fill_data[3][4:] = [50, 52, 59, 60, 59, 64, 79]
# fill_data[4][5:] = [49, 53, 55, 55, 62, 79]
# fill_data[5][6:] = [50, 49, 50, 49 49]
# fill_data[6][7:] = [49, 50, 50, 50]
# fill_data[7][8:] = [50, 50, 50]
# fill_data[8][9:] = [50, 50]
# fill_data[9][10:] = [50]
# annot_data[0][1:] = [r'50 $\pm$ 0',r'52 $\pm$ 0',r'56 $\pm$ 0',r'62 $\pm$ 0',r'67 $\pm$ 0',r'70 $\pm$ 0',r'68 $\pm$ 0',r'64 $\pm$ 0',r'59 $\pm$ 0',r'68 $\pm$ 0']
# annot_data[1][2:] = [r'51 $\pm$ 0',r'51 $\pm$ 0',r'53 $\pm$ 0',r'55 $\pm$ 0',r'56 $\pm$ 0',r'53 $\pm$ 0',r'52 $\pm$ 0',r'51 $\pm$ 0',r'57 $\pm$ 0']
# annot_data[2][3:] = [r'49 $\pm$ 0',r'52 $\pm$ 0',r'53 $\pm$ 0',r'53 $\pm$ 0',r'51 $\pm$ 0',r'50 $\pm$ 0',r'50 $\pm$ 0',r'52 $\pm$ 0']
# annot_data[3][4:] = [r'51 $\pm$ 0',r'52 $\pm$ 0',r'42 $\pm$ 0',r'50 $\pm$ 0',r'49 $\pm$ 0',r'50 $\pm$ 0',r'50 $\pm$ 0']
# annot_data[4][5:] = [r'50 $\pm$ 0',r'49 $\pm$ 0',r'49 $\pm$ 0',r'49 $\pm$ 0',r'49 $\pm$ 0',r'49 $\pm$ 0']
# annot_data[5][6:] = [r'50 $\pm$ 0',r'49 $\pm$ 0',r'50 $\pm$ 0',r'49 $\pm$ 0',r'49 $\pm$ 0']
# annot_data[6][7:] = [r'49 $\pm$ 0',r'50 $\pm$ 0',r'50 $\pm$ 0',r'50 $\pm$ 0']
# annot_data[7][8:] = [r'50 $\pm$ 0',r'50 $\pm$ 0',r'50 $\pm$ 0']
# annot_data[8][9:] = [r'50 $\pm$ 0',r'50 $\pm$ 0']
# annot_data[9][10:] = [r'50 $\pm$ 0']
# fill_data_other[1][:1] = [800]
# fill_data_other[2][:2] = [800, 0.06]
# fill_data_other[3][:3] = [800, 0.21, 0.02]
# fill_data_other[4][:4] = [800, 0.34, 0.10, 0.00]
# fill_data_other[5][:5] = [800, 0.65, 0.17, 0.05, 0.02]
# fill_data_other[6][:6] = [800, 0.47, 0.16, 0.08, 0.04, 0.32]
# fill_data_other[7][:7] = [800, 0.51, 0.24, 0.12, 0.05, 0.27, 0.01]
# fill_data_other[8][:8] = [800, 0.42, 0.13, 0.11, 0.04, 0.30, 0.03, 0.02]
# fill_data_other[9][:9] = [800, 0.39, 0.17, 0.17, 0.14, 0.37, 0.10, 0.21, 0.15]
# fill_data_other[10][:10] = [800, 0.30, 800, 800, 800, 6.54, 800, 17.43, 800, 800]

# CelebA Meta (Male)
#fill_data[0][1:] =  [52, 54, 56, 59, 65, 61, 67, 70, 72, 68]
#fill_data[1][2:] =  [48, 56, 63, 71, 69, 81, 83, 85, 91]
#fill_data[2][3:] =  [49, 49, 55, 58, 67, 78, 71, 83]
#fill_data[3][4:] =  [52, 59, 68, 73, 76, 89, 94]
#fill_data[4][5:] =  [53, 60, 70, 73, 83, 91]
#fill_data[5][6:] =  [53, 59, 62, 76, 91]
#fill_data[6][7:] =  [54, 58, 70, 82]
#fill_data[7][8:] =  [52, 61, 72]
#fill_data[8][9:] =  [53, 64]
#fill_data[9][10:] = [57]
#annot_data[0][1:]  = [r'52 $\pm$ 1',r'54 $\pm$ 1',r'56 $\pm$ 2',r'59 $\pm$ 1',r'65 $\pm$ 3',r'61 $\pm$ 2',r'67 $\pm$ 2',r'70 $\pm$ 3',r'72 $\pm$ 7',r'68 $\pm$ 6']
#annot_data[1][2:]  = [r'48 $\pm$ 0',r'56 $\pm$ 1',r'63 $\pm$ 2',r'71 $\pm$ 1',r'69 $\pm$ 4',r'81 $\pm$ 4',r'83 $\pm$ 2',r'85 $\pm$ 3',r'91 $\pm$ 4']
#annot_data[2][3:]  = [r'49 $\pm$ 0',r'49 $\pm$ 1',r'55 $\pm$ 4',r'58 $\pm$ 9',r'67 $\pm$ 7',r'78 $\pm$ 2',r'71 $\pm$ 5',r'83 $\pm$ 9']
#annot_data[3][4:]  = [r'52 $\pm$ 1',r'59 $\pm$ 2',r'68 $\pm$ 0',r'73 $\pm$ 6',r'76 $\pm$ 4',r'89 $\pm$ 1',r'94 $\pm$ 0']
#annot_data[4][5:]  = [r'53 $\pm$ 1',r'60 $\pm$ 2',r'70 $\pm$ 3',r'73 $\pm$ 2',r'83 $\pm$ 4',r'91 $\pm$ 2']
#annot_data[5][6:]  = [r'53 $\pm$ 0',r'59 $\pm$ 1',r'62 $\pm$ 3',r'76 $\pm$ 4',r'91 $\pm$ 1']
#annot_data[6][7:]  = [r'54 $\pm$ 1',r'58 $\pm$ 0',r'70 $\pm$ 3',r'82 $\pm$ 3']
#annot_data[7][8:]  = [r'52 $\pm$ 0', r'61 $\pm$ 1', r'72 $\pm$ 3']
#annot_data[8][9:] =  [r'53 $\pm$ 1', r'64 $\pm$ 1']
#annot_data[9][10:] = [r'57 $\pm$ 1']
#fill_data_other[1][:1] =   [0.06]
#fill_data_other[2][:2] =   [0.08, 0.00]
#fill_data_other[3][:3] =   [0.08, 0.13, 0.00]
#fill_data_other[4][:4] =   [0.12, 0.26, 0.00, 0.03]
#fill_data_other[5][:5] =   [0.23, 0.45, 0.15, 0.18, 0.05]
#fill_data_other[6][:6] =   [0.11, 0.41, 0.41, 0.30, 0.16, 0.04]
#fill_data_other[7][:7] =   [0.23, 0.67, 0.35, 0.55, 0.49, 0.15, 0.08]
#fill_data_other[8][:8] =   [0.29, 0.52, 0.38, 0.48, 0.52, 0.28, 0.11, 0.02]
#fill_data_other[9][:9] =   [0.33, 0.52, 0.51, 1.03, 0.94, 0.85, 0.69, 0.30, 0.10]
#fill_data_other[10][:10] = [800, 0.89, 1.13, 1.51, 1.87, 1.99, 1.57, 0.94, 0.49, 0.27]

# CelebA Young
# fill_data[0][1:]  = [51, 54, 61, 64, 70, 75, 77, 89, 92, 95]
# fill_data[1][2:]  = [51, 53, 60, 65, 67, 77, 82, 87, 92]
# fill_data[2][3:]  = [50, 51, 55, 65, 66, 79, 75, 87]
# fill_data[3][4:]  = [50, 52, 57, 61, 69, 79, 80]
# fill_data[4][5:]  = [50, 53, 56, 59, 71, 82]
# fill_data[5][6:]  = [49, 53, 62, 69, 72]
# fill_data[6][7:]  = [50, 55, 58, 72]
# fill_data[7][8:]  = [50, 54, 64]
# fill_data[8][9:]  = [49, 54]
# fill_data[9][10:] = [50]
# annot_data[0][1:]  = [r'51 $\pm$ 1', r'54 $\pm$ 3', r'61 $\pm$ 3', r'64 $\pm$ 7', r'70 $\pm$ 9', r'75 $\pm$ 5', r'77 $\pm$ 14', r'89 $\pm$ 0', r'92 $\pm$ 0', r'95 $\pm$ 0']
# annot_data[1][2:]  = [r'51 $\pm$ 0', r'53 $\pm$ 2', r'60 $\pm$ 2', r'65 $\pm$ 1', r'67 $\pm$ 8', r'77 $\pm$ 2', r'82 $\pm$ 2', r'87 $\pm$ 2', r'92 $\pm$ 1']
# annot_data[2][3:]  = [r'50 $\pm$ 0', r'51 $\pm$ 1', r'55 $\pm$ 4', r'65 $\pm$ 1', r'66 $\pm$ 6', r'79 $\pm$ 0', r'75 $\pm$ 12', r'87 $\pm$ 3']
# annot_data[3][4:]  = [r'50 $\pm$ 1', r'52 $\pm$ 2', r'57 $\pm$ 3', r'61 $\pm$ 7', r'69 $\pm$ 9', r'79 $\pm$ 0', r'80 $\pm$ 15']
# annot_data[4][5:]  = [r'50 $\pm$ 0', r'53 $\pm$ 1', r'56 $\pm$ 4', r'59 $\pm$ 8', r'71 $\pm$ 4', r'82 $\pm$ 2']
# annot_data[5][6:]  = [r'49 $\pm$ 0', r'53 $\pm$ 1', r'62 $\pm$ 1', r'69 $\pm$ 0', r'72 $\pm$ 11']
# annot_data[6][7:]  = [r'50 $\pm$ 0', r'55 $\pm$ 2', r'58 $\pm$ 6', r'72 $\pm$ 5']
# annot_data[7][8:]  = [r'50 $\pm$ 1', r'54 $\pm$ 3', r'64 $\pm$ 4']
# annot_data[8][9:]  = [r'49 $\pm$ 1', r'54 $\pm$ 3']
# annot_data[9][10:] = [r'50 $\pm$ 1']
# fill_data_other[1][:1]   = [0.03]
# fill_data_other[2][:2]   = [0.13, 0.01]
# fill_data_other[3][:3]   = [0.31, 0.07, 0.01]
# fill_data_other[4][:4]   = [0.34, 0.18, 0.01, 0.01]
# fill_data_other[5][:5]   = [0.47, 0.21, 0.12, 0.03, 0.01]
# fill_data_other[6][:6]   = [0.51, 0.32, 0.19, 0.08, 0.04, 0]
# fill_data_other[7][:7]   = [0.58, 0.40, 0.22, 0.15, 0.07, 0.05, 0]
# fill_data_other[8][:8]   = [0.67, 0.44, 0.33, 0.31, 0.20, 0.17, 0.11, 0.01]
# fill_data_other[9][:9]   = [0.59, 0.46, 0.45, 0.42, 0.40, 0.33, 0.24, 0.11, 0]
# fill_data_other[10][:10] = [800, 0.68, 0.72, 0.79, 0.71, 0.63, 0.61, 0.45, 0.18, 0.04]

# CelebA Threshold (young)
# fill_data[0][1:] = [49, 50, 50, 49, 50, 49, 50, 46, 58, 63]
# fill_data[1][2:] = [50, 49, 47, 49, 49, 49, 49, 58, 78]
# fill_data[2][3:] = [50, 50, 50, 51, 52, 53, 57, 64]
# fill_data[3][4:] = [49, 49, 46, 50, 49, 53, 56]
# fill_data[4][5:] = [51, 59, 66, 65, 76, 88]
# fill_data[5][6:] = [50, 51, 52, 51, 55]
# fill_data[6][7:] = [51, 55, 55, 66]
# fill_data[7][8:] = [51, 51, 52]
# fill_data[8][9:] = [49, 49]
# fill_data[9][10:] = [49]
# annot_data[0][1:] = [r'49 $\pm$ 0', r'50 $\pm$ 3', r'50 $\pm$ 0', r'49 $\pm$ 0', r'50 $\pm$ 0', r'49 $\pm$ 0', r'50 $\pm$ 0', r'46 $\pm$ 4', r'58 $\pm$ 12', r'63 $\pm$ 18']
# annot_data[1][2:] = [r'50 $\pm$ 1', r'49 $\pm$ 0', r'47 $\pm$ 3', r'49 $\pm$ 0', r'49 $\pm$ 1', r'49 $\pm$ 0', r'49 $\pm$ 0', r'58 $\pm$ 7', r'78 $\pm$ 2']
# annot_data[2][3:] = [r'50 $\pm$ 1',r'50 $\pm$ 1',r'50 $\pm$ 1',r'51 $\pm$ 2',r'52 $\pm$ 1',r'53 $\pm$ 1',r'57 $\pm$ 3',r'64 $\pm$ 6']
# annot_data[3][4:] = [r'49 $\pm$ 0', r'49 $\pm$ 0', r'46 $\pm$ 4', r'50 $\pm$ 0', r'49 $\pm$ 0', r'53 $\pm$ 1', r'56 $\pm$ 5']
# annot_data[4][5:] = [r'51 $\pm$ 1', r'59 $\pm$ 2', r'66 $\pm$ 4', r'65 $\pm$ 2', r'76 $\pm$ 4', r'88 $\pm$ 3']
# annot_data[5][6:] = [r'50 $\pm$ 1', r'51 $\pm$ 0', r'52 $\pm$ 1', r'51 $\pm$ 0', r'55 $\pm$ 2']
# annot_data[6][7:] = [r'51 $\pm$ 1', r'55 $\pm$ 1', r'55 $\pm$ 2', r'66 $\pm$ 4']
# annot_data[7][8:] = [r'51 $\pm$ 0', r'51 $\pm$ 1', r'52 $\pm$ 0']
# annot_data[8][9:] = [r'49 $\pm$ 0', r'49 $\pm$ 0']
# annot_data[9][10:] = [r'49 $\pm$ 0']
# fill_data_other[1][:1] = [0.08]
# fill_data_other[2][:2] = [0.19, 0.01]
# fill_data_other[3][:3] = [0.21, 0.1, 0.01]
# fill_data_other[4][:4] = [0.28, 0.2, 0.06, 0.00]
# fill_data_other[5][:5] = [0.65, 0.35, 0.17, 0.02, 0.02]
# fill_data_other[6][:6] = [0.56, 0.4, 0.19, 0.07, 0.02, 0.01]
# fill_data_other[7][:7] = [0.54, 0.41, 0.26, 0.13, 0.07, 0.02, 0.01]
# fill_data_other[8][:8] = [0.48, 0.35, 0.21, 0.16, 0.12, 0.07, 0.01, 0.01]
# fill_data_other[9][:9] = [0.35, 0.36, 0.31, 0.24, 0.23, 0.16, 0.12, 0.05, 0.01]
# fill_data_other[10][:10] = [800, 0.36, 0.35, 0.27, 0.33, 0.35, 0.22, 0.19, 0.12, 0.01]

# CelebA Young (Regression Extension)
# fill_data[0][1:] =  [50, 53, 59, 66, 73, 78, 83, 87, 90, 93]
# fill_data[1][2:] =  [53, 58, 63, 69, 75, 80, 84, 88, 92]
# fill_data[2][3:] =  [55, 59, 66, 71, 76, 80, 84, 88]
# fill_data[3][4:] =  [54, 60, 66, 71, 75, 80, 84]
# fill_data[4][5:] =  [57, 61, 66, 70, 75, 79]
# fill_data[5][6:] =  [54, 60, 65, 69, 74]
# fill_data[6][7:] =  [55, 60, 64, 67]
# fill_data[7][8:] =  [54, 58, 61]
# fill_data[8][9:] =  [53, 54]
# fill_data[9][10:] = [50]
# annot_data[0][1:]  = [r'50 $\pm$ 0',r'53 $\pm$ 2',r'59 $\pm$ 2',r'66 $\pm$ 2',r'73 $\pm$ 2',r'78 $\pm$ 2',r'83 $\pm$ 1',r'87 $\pm$ 1',r'90 $\pm$ 1',r'93 $\pm$ 1']
# annot_data[1][2:]  = [r'53 $\pm$ 1',r'58 $\pm$ 1',r'63 $\pm$ 1',r'69 $\pm$ 1',r'75 $\pm$ 1',r'80 $\pm$ 1',r'84 $\pm$ 1',r'88 $\pm$ 2',r'92 $\pm$ 1']
# annot_data[2][3:]  = [r'55 $\pm$ 0',r'59 $\pm$ 0',r'66 $\pm$ 0',r'71 $\pm$ 1',r'76 $\pm$ 2',r'80 $\pm$ 2',r'84 $\pm$ 3',r'88 $\pm$ 3']
# annot_data[3][4:]  = [r'54 $\pm$ 0',r'60 $\pm$ 1',r'66 $\pm$ 1',r'71 $\pm$ 3',r'75 $\pm$ 3',r'80 $\pm$ 3',r'84 $\pm$ 4']
# annot_data[4][5:]  = [r'57 $\pm$ 3',r'61 $\pm$ 1',r'66 $\pm$ 2',r'70 $\pm$ 4',r'75 $\pm$ 4',r'79 $\pm$ 5']
# annot_data[5][6:]  = [r'54 $\pm$ 1',r'60 $\pm$ 2',r'65 $\pm$ 3',r'69 $\pm$ 4',r'74 $\pm$ 6']
# annot_data[6][7:]  = [r'55 $\pm$ 1',r'60 $\pm$ 3',r'64 $\pm$ 4',r'67 $\pm$ 6']
# annot_data[7][8:]  = [r'54 $\pm$ 1', r'58 $\pm$ 3', r'61 $\pm$ 5']
# annot_data[8][9:] =  [r'53 $\pm$ 1', r'54 $\pm$ 2']
# annot_data[9][10:] = [r'50 $\pm$ 1']
# fill_data_other[1][:1] =   [0.02]
# fill_data_other[2][:2] =   [0.07, 0.08]
# fill_data_other[3][:3] =   [0.20, 0.19, 0.13]
# fill_data_other[4][:4] =   [0.30, 0.24, 0.15, 0.07]
# fill_data_other[5][:5] =   [0.45, 0.36, 0.26, 0.18, 0.57]
# fill_data_other[6][:6] =   [0.58, 0.47, 0.36, 0.25, 0.20, 0.10]
# fill_data_other[7][:7] =   [0.60, 0.49, 0.41, 0.31, 0.29, 0.25, 0.15]
# fill_data_other[8][:8] =   [0.61, 0.51, 0.44, 0.47, 0.48, 0.42, 0.33, 0.12]
# fill_data_other[9][:9] =   [0.57, 0.51, 0.57, 0.61, 0.59, 0.54, 0.45, 0.28, 0.09]
# fill_data_other[10][:10] = [800, 0.67, 0.76, 0.78, 0.77, 0.76, 0.61, 0.38, 0.11, 0.03]

# arXiv Meta
fill_data[0][1:] = [52, 89, 97, 98, 99, 98, 99, 99]
fill_data[1][2:] = [75, 92, 95, 95, 99, 99, 99]
fill_data[2][3:] = [76, 92, 96, 98, 99, 99]
fill_data[3][4:] = [91, 100, 95, 99, 100]
fill_data[4][5:] = [90, 97, 98, 99]
fill_data[5][6:] = [94, 97, 100]
fill_data[6][7:] = [64, 79]
fill_data[7][8:] = [89]
annot_data[0][1:] = [r'52 $\pm$ 1', r'89 $\pm$ 9', r'97 $\pm$ 2', r'98 $\pm$ 0', r'99 $\pm$ 0', r'98 $\pm$ 1', r'99 $\pm$ 0', r'99 $\pm$ 0']
annot_data[1][2:] = [r'75 $\pm$ 22', r'92 $\pm$ 7', r'95 $\pm$ 2', r'95 $\pm$ 4', r'99 $\pm$ 1', r'99 $\pm$ 0', r'99 $\pm$ 0']
annot_data[2][3:] = [r'76 $\pm$ 19', r'92 $\pm$ 1', r'96 $\pm$ 6', r'98 $\pm$ 3', r'99 $\pm$ 0', r'99 $\pm$ 0']
annot_data[3][4:] = [r'91 $\pm$ 5', r'100 $\pm$ 0', r'95 $\pm$ 5', r'99 $\pm$ 0', r'100 $\pm$ 0']
annot_data[4][5:] = [r'90 $\pm$ 2', r'97 $\pm$ 3', r'98 $\pm$ 2', r'99 $\pm$ 0']
annot_data[5][6:] = [r'94 $\pm$ 10', r'97 $\pm$ 1', r'100 $\pm$ 0']
annot_data[6][7:] = [r'64 $\pm$ 12', r'79 $\pm$ 11']
annot_data[7][8:] = [r'89 $\pm$ 11']
fill_data_other[1][:1] = [800]
fill_data_other[2][:2] = [15.72, 15.99]
fill_data_other[3][:3] = [800, 800, 9.99]
fill_data_other[4][:4] = [800, 15.99, 7.07, 5.80]
fill_data_other[5][:5] = [800, 800, 800, 800, 4.53]
fill_data_other[6][:6] = [800, 800, 800, 800, 800, 800]
fill_data_other[7][:7] = [800, 800, 800, 800, 779, 800, 3.3]
fill_data_other[8][:8] = [800, 800, 800, 800, 800, 800, 40.12, 800]

# boneAge Meta
# fill_data[0][1:] = [81, 98, 99, 99, 99, 99]
# fill_data[1][2:] = [76, 95, 93, 99, 99]
# fill_data[2][3:] = [66, 88, 96, 99]
# fill_data[3][4:] = [60, 81, 97]
# fill_data[4][5:] = [55, 75]
# fill_data[5][6:] = [54]
# annot_data[0][1:] = [r'81 $\pm$ 11', r'98 $\pm$ 0', r'99 $\pm$ 0', r'99 $\pm$ 0', r'99 $\pm$ 0', r'100 $\pm$ 0']
# annot_data[1][2:] = [r'76 $\pm$ 11', r'95 $\pm$ 1', r'93 $\pm$ 14', r'99 $\pm$ 0', r'99 $\pm$ 0']
# annot_data[2][3:] = [r'66 $\pm$ 4', r'88 $\pm$ 6', r'96 $\pm$ 2', r'99 $\pm$ 0']
# annot_data[3][4:] = [r'60 $\pm$ 4', r'81 $\pm$ 6', r'97 $\pm$ 2']
# annot_data[4][5:] = [r'55 $\pm$ 2', r'75 $\pm$ 6']
# annot_data[5][6:] = [r'54 $\pm$ 2']
# fill_data_other[1][:1] = [6.65]
# fill_data_other[2][:2] = [15.38, 5.68]
# fill_data_other[3][:3] = [13.22, 7.57, 1.52]
# fill_data_other[4][:4] = [8.97, 7.91, 4.37, 0.67]
# fill_data_other[5][:5] = [800, 800, 7.19, 3.12, 0.31]
# fill_data_other[6][:6] = [800, 800, 800, 8.13, 3.56, 0.22]

annot_data_other = [[""] * len(targets) for _ in range(len(targets))]
mask_other = np.logical_not(mask)
for i in range(mask_other.shape[0]):
    mask_other[i][i] = False

temp = fill_data_other.copy()
temp[temp > 100] = 0
maz = np.max(temp)

for i in range(1, len(targets)):
    for j in range(i):
        if fill_data_other[i][j] > 100:
            annot_data_other[i][j] = r"$>100^\dagger$"
            fill_data_other[i][j] = maz
        else:
            annot_data_other[i][j] = "%.2f" % fill_data_other[i][j]

# # First heatmap (top-right)
# for i in range(len(targets)):
#     for j in range(len(targets)-(i+1)):
#         m = raw_data_loss[i][j]
#         fill_data[i][j+i+1] = m
#         mask[i][j+i+1] = False
#         annot_data[i][j+i+1] = r'%d' % m


# Second heatmap (bottom-left)
# mask = np.zeros_like(mask)
# for i in range(len(targets)):
#     for j in range(len(targets)-(i+1)):
#         m = eff_vals[i][j]
#         if m > 100:
#             fill_data[i][j+i+1] = 100
#             annot_data[i][i] = r'$\dagger$'
#         else:
#             fill_data[i][j+i+1] = m
#             annot_data[i][i] = "%.1f" % m
#         mask[i][j+i+1] = False

for i in range(mask.shape[0]):
    mask[i][i] = True
    mask_other[i][i] = True

sns_plot = sns.heatmap(fill_data, xticklabels=targets,
                       yticklabels=targets,
                       annot=annot_data,
                       cbar=False,
                    #    cbar=True,
                       mask=mask, fmt="^",
                       vmin=50, vmax=100,)

sns_plot = sns.heatmap(fill_data_other, xticklabels=targets,
                       yticklabels=targets,
                       annot=annot_data_other,
                    #    cbar=True,
                       cbar=False,
                       mask=mask_other, fmt="^",
                       cmap="YlGnBu",)

sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
plt.tight_layout()
sns_plot.figure.savefig("./arxiv_heatmap_meta.pdf")
