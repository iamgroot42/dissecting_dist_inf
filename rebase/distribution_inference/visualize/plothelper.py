import seaborn
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import warnings
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

from distribution_inference.logging.core import AttackResult
from distribution_inference.utils import warning_string
from distribution_inference.attacks.utils import get_attack_name


class PlotHelper():
    def __init__(self, path: str = '',
                 logger: AttackResult = None,
                 columns=['Ratios', 'Values', 'Hues']):
        self.df = []
        self.path = path
        self.logger = logger
        self.columns = columns
        self.supported_plot_types = {
            'violin': self.violinplot,
            'box': self.boxplot,
            'reg': self.regplot
        }
        # Parse results
        self._parse()

    def _parse(self):
        if(len(self.columns) != 3):
            raise ValueError(
                "columns argument must be of length 3")
        # Values for plot
        ratios = []
        # Check logger
        if(self.path != ''):
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Provided path {self.path} does not exist")
            # Using JSON file
            logger = json.load(open(self.path, 'r'))
        elif (logger is not None) and type(logger) == AttackResult:
            # Using logger object directly
            logger = logger.dic
        else:
            raise ValueError(
                "Must pass either a logger class or a path")

        # Look at all the results
        for attack_res in logger['result']:
            attack_names = get_attack_name(attack_res)
            # Loss & Threshold attacks
            if(attack_res == "loss_and_threshold"):
                for ratio in logger['result'][attack_res]:
                    ratios.append(ratio)  # add ratio
                    victim_results = logger['result'][attack_res][ratio]['victim_acc']
                    for results in victim_results:
                        loss = results[0]
                        threshold = results[1]
                        self.df.append({
                            self.columns[0]: float(ratio),
                            self.columns[1]: loss,
                            self.columns[2]: attack_names[0]})
                        self.df.append({
                            self.columns[0]: float(ratio),
                            self.columns[1]: threshold,
                            self.columns[2]: attack_names[1]})
            # Per-point threshold attack, or white-box attack
            elif attack_res in ["threshold_perpoint", "affinity", "permutation_invariant"]:
                for ratio in logger['result'][attack_res]:
                    ratios.append(ratio)  # add ratio
                    victim_results = logger['result'][attack_res][ratio]['victim_acc']
                    for results in victim_results:
                        self.df.append({
                            self.columns[0]: float(ratio),
                            self.columns[1]: results,
                            self.columns[2]: attack_names})
            else:
                warnings.warn(warning_string(f"\nAttack type {attack_res} not supported\n"))
        if len(self.df) == 0:
            raise ValueError("None of the attacks in given results are supported for plotting")

        # Convert data to dataframe
        self.df = pd.DataFrame(self.df)

    def get_appropriate_plotter_fn(self, plot_type):
        plotter_fn = self.supported_plot_types.get(plot_type, None)
        if plotter_fn is None:
            raise ValueError("Requested plot-type not supported")
        return plotter_fn

    # Box plot, returns a graph object given a logger object
    def boxplot(self, title='', darkplot=True, dash=True):
        graph = seaborn.boxplot(
            x=self.columns[0], y=self.columns[1],
            hue=self.columns[2], data=self.df)
        # Distinguishing accuracy range
        # TODO: Make this generic (to support loss values etc)
        graph.set(ylim=(45, 101))

        graph.set_title(title)
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

    # Violin plot, returns a graph object given a logger object
    def violinplot(self, title='', darkplot=True, dash=True):
        graph = seaborn.violinplot(
            x=self.columns[0], y=self.columns[1],
            hue=self.columns[2], data=self.df)

        graph.set_title(title)
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

    # Regression plot, returns a graph object given a logger object
    # This plot does not take hues
    def regplot(self, title='', darkplot=True, dash=True):
        graph = seaborn.regplot(
            x=self.columns[0], y=self.columns[1], data=self.df)

        graph.set_title(title)
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
