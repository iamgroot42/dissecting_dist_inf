import seaborn
import matplotlib.pyplot as plt
#Example Setup
from simple_parsing import ArgumentParser
import json
import pandas as pd
from distribution_inference.logging.core import AttackResult

#Helper class for plotting logging objects
#Can either directly pass the path to the logger or the the logger object 
#Also takes a 3-length columns argument for chart customization
#Example:
#plothelper = PlotHelper('test2.json')
#graph = plothelper.violinplot(title = 'title')
#plt.show()

class PlotHelper():
    def __init__(self, path:str = '', logger:AttackResult = None, columns = ['Ratios', 'Values', 'Hues']):
        self.df = pd.DataFrame(columns = columns)
        self.path = path
        self.logger = logger
        self.columns = columns
        self.parse()
    #Parse logger file
    def parse(self):
        if(len(self.columns) != 3):
            raise ValueError(
                    "columns argument must be of length 3")
        #Values for plot
        ratios = []
        #Check logger
        if(self.path != ''): #using json file
            logger = json.load(open(self.path, 'r'))
        elif(logger != None): #using logger object
            logger = logger.dic
        else:
            raise ValueError(
                    "Must pass either a logger class or a path")
        for attack_res in logger['result']: #look in results

            if(attack_res == "loss_and_threshold"): #parsing for loss_and_threshold attack type
                for ratio in logger['result'][attack_res]:
                    ratios.append(ratio) #add ratio
                    for results in logger['result'][attack_res][ratio]['victim_acc']:
                        loss = results[0]
                        threshold = results[1]
                        self.df = self.df.append(pd.DataFrame({self.columns[0]: [float(ratio)], self.columns[1]: [loss], self.columns[2]: ['Loss']}), ignore_index = True)
                        self.df = self.df.append(pd.DataFrame({self.columns[0]: [float(ratio)], self.columns[1]: [threshold], self.columns[2]: ['Threshold']}), ignore_index = True)

            elif(attack_res == "threshold_perpoint"): #parsing for threshold_perpoint attack type
                for ratio in logger['result'][attack_res]:
                    ratios.append(ratio) #add ratio
                    
                    for results in logger['result'][attack_res][ratio]['adv_acc']:
                        self.df = self.df.append(pd.DataFrame({self.columns[0]: [float(ratio)], self.columns[1]: [results], self.columns[2]: ['Per Point (Adv)']}), ignore_index = True)
                    for results in logger['result'][attack_res][ratio]['victim_acc']:
                        self.df = self.df.append(pd.DataFrame({self.columns[0]: [float(ratio)], self.columns[1]: [results], self.columns[2]: ['Per Point (Vict)']}), ignore_index = True)
    
    #Box plot, returns a graph object given a logger object
    def boxplot(self, title = '', darkplot = True, dash = True):
        
        graph = seaborn.boxplot(x = self.columns[0], y = self.columns[1], hue = self.columns[2], data = self.df)

        graph.set_title(title)
        #Plot settings (from celeba)
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        # Add dividing line in centre
        lower, upper = plt.gca().get_xlim()
        if dash:
            midpoint = (lower + upper) / 2
            plt.axvline(x=midpoint,
                        color='white' if darkplot else 'black',
                        linewidth=1.0, linestyle='--')
        # Make sure axis label not cut off
        plt.tight_layout()

        return graph

    #Violin plot, returns a graph object given a logger object
    def violinplot(self, title = '', darkplot = True, dash = True):
        
        graph = seaborn.violinplot(x = self.columns[0], y = self.columns[1], hue = self.columns[2], data = self.df)

        graph.set_title(title)
        #Plot settings (from celeba)
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        # Add dividing line in centre
        lower, upper = plt.gca().get_xlim()
        if dash:
            midpoint = (lower + upper) / 2
            plt.axvline(x=midpoint,
                        color='white' if darkplot else 'black',
                        linewidth=1.0, linestyle='--')
        # Make sure axis label not cut off
        plt.tight_layout()

        return graph

    #Regression plot, returns a graph object given a logger object
    #This plot does not take hues
    def regplot(self, title = '', darkplot = True, dash = True):
        graph = seaborn.regplot(x = self.columns[0], y = self.columns[1], data = self.df)

        graph.set_title(title)
        #Plot settings (from celeba)
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        # Add dividing line in centre
        lower, upper = plt.gca().get_xlim()
        if dash:
            midpoint = (lower + upper) / 2
            plt.axvline(x=midpoint,
                        color='white' if darkplot else 'black',
                        linewidth=1.0, linestyle='--')
        # Make sure axis label not cut off
        plt.tight_layout()

        return graph


