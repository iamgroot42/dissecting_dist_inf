import seaborn
import matplotlib.pyplot as plt
#Example Setup
from simple_parsing import ArgumentParser
import json
import pandas as pd
from distribution_inference.logging.core import AttackResult
from distribution_inference.visualize.plothelper import PlotHelper
#from plothelper import PlotHelper
import os

if __name__ == "__main__":
    #Arguments for plotting
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--logpath", help="Specify logger json file path",type=str,required=True)
    parser.add_argument("--plot", help="Specify plot type", choices = ['violin', 'box', 'reg'], type=str, required=True)
    parser.add_argument("--savepath", help="Specify save path (ending with /)",type=str)

    parser.add_argument("--title", default = '', help = "Plot title",type=str)
    parser.add_argument("--x", default = 'Ratios', help = "x axis title",type=str)
    parser.add_argument("--y", default = 'Values', help = "y axis title",type=str)
    parser.add_argument("--legend", default = 'Hues',help = "legend title", type = str)
    parser.add_argument("--dark", default = True, help = "dark background", type = bool)
    parser.add_argument("--dash", default = True, help = "add dashed line midway?", type = bool)
    args = parser.parse_args()
    print(args)
    #Columns for axis and names
    columns = [args.x, args.y, args.legend]

    #Create plothelper object
    plothelper = PlotHelper(path = args.logpath, columns = columns)
    #Check plot type and plot
    if(args.plot == 'violin'):
        graph = plothelper.violinplot(title = args.title, darkplot = args.dark, dash = args.dash)
    elif(args.plot == 'box'):
        graph = plothelper.boxplot(title = args.title, darkplot = args.dark, dash = args.dash)
    elif(args.plot == 'reg'):
        graph = plothelper.regplot(title = args.title, darkplot = args.dark, dash = args.dash)
    #plt.show()
    
    #Save plot
    graph.figure.savefig('.' + args.savepath + os.path.basename(args.logpath) + '_%s.png' %(args.plot))
