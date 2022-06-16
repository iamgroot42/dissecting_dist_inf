from distribution_inference.attacks.utils import get_attack_name
from distribution_inference.utils import warning_string
from distribution_inference.logging.core import AttackResult
import seaborn
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import warnings
from typing import List
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class PlotHelper():
    def __init__(self,
                 paths: List[str] = [''],
                 loggers: List[AttackResult] = [None],
                 columns=['Ratios', 'Values', 'Hues', 'Epoch'],
                 legend_titles: List = None,
                 attacks_wanted: List = None,
                 ratios_wanted: List = None,
                 no_legend: bool = False):
        self.df = []
        self.paths = paths
        self.loggers = loggers
        self.columns = columns
        self.attacks_wanted = attacks_wanted
        self.ratios_wanted = ratios_wanted
        self.no_legend = no_legend
        if(len(self.columns) < 3):
            raise ValueError(
                "columns argument must be of length 3")
        self.supported_plot_types = {
            'violin': self.violinplot,
            'box': self.boxplot,
            'reg': self.regplot,
            'line': self.lineplot
        }
        self.legend_titles = legend_titles
        # If legend titles given, must be same length as paths/loggers
        if self.legend_titles is not None:
            if len(self.legend_titles) != len(self.paths) and len(self.legend_titles) != len(self.loggers):
                raise ValueError(
                    f"legend_titles ({len(legend_titles)}) must be of length equal to paths or loggers")
        # Must not provide empty lists
        if type(self.paths) == list and len(self.paths) == 0:
            raise ValueError("Must provide at least one path")
        if type(self.loggers) == list and len(self.loggers) == 0:
            raise ValueError("Must provide at least one logger")
        # Cannot provide both logger and path
        if self.paths[0] != '' and self.loggers[0] is not None:
            raise ValueError(
                "Must pass either a logger class or a path")
        if self.loggers[0] is not None:
            self._parse_results(self.loggers, are_paths=False)
        elif self.paths[0] != '':
            self._parse_results(self.paths, are_paths=True)

        # Convert data to dataframe
        self.df = pd.DataFrame(self.df)
        # Print out means
        print(self.df.groupby(self.columns[2])[self.columns[1]].mean())

    def _parse_results(self, list_of_things, are_paths: bool):
        for i, thing in enumerate(list_of_things):
            if are_paths:
                logger = self._get_logger_from_path_or_obj(thing, None)
            else:
                logger = thing.dic
            # Parse data from given results-object
            self._parse(logger, i)
        pass

    def _get_logger_from_path_or_obj(self, path, logger_obj):
        if path != '':
            if not os.path.exists(path):
                raise FileNotFoundError(f"Provided path {path} does not exist")
            # Using JSON file
            logger = json.load(open(path, 'r'))
        elif (logger_obj is not None) and type(logger_obj) == AttackResult:
            # Using logger object directly
            logger = logger_obj.dic
        else:
            raise ValueError(
                "Must pass either a logger class or a path")
        return logger

    def _parse(self, logger, legend_entry_index: int = None):
        # Look at all the results
        for attack_res in logger['result']:
            print(attack_res)
            if self.attacks_wanted is not None and attack_res not in self.attacks_wanted:
                print(f"Not plotting {attack_res}")
                continue
            title_prefix = ""
            if self.legend_titles is not None:
                title_prefix = self.legend_titles[legend_entry_index] + " : "
            attack_names = get_attack_name(attack_res)
            # Loss & Threshold attacks
            if(attack_res == "loss_and_threshold"):
                for ratio in logger['result'][attack_res]:
                    if self.ratios_wanted is not None and ratio not in self.ratios_wanted:
                        continue
                    victim_results = logger['result'][attack_res][ratio]['victim_acc']
                    for results in victim_results:
                        loss = results[1]
                        threshold = results[0]
                        if type(loss) == list:
                            assert len(loss) == len(threshold)
                            for epoch, (l, t) in enumerate(zip(loss, threshold)):
                                self.df.append({
                                    self.columns[0]: float(ratio),
                                    self.columns[1]: l,
                                    self.columns[2]: title_prefix + attack_names[0],
                                    self.columns[3]: epoch + 1})
                                self.df.append({
                                    self.columns[0]: float(ratio),
                                    self.columns[1]: t,
                                    self.columns[2]: title_prefix + attack_names[1],
                                    self.columns[3]: epoch + 1})
                        else:
                            assert type(threshold) != list
                            self.df.append({
                                self.columns[0]: float(ratio),
                                self.columns[1]: loss,
                                self.columns[2]: title_prefix + attack_names[0]})
                            self.df.append({
                                self.columns[0]: float(ratio),
                                self.columns[1]: threshold,
                                self.columns[2]: title_prefix + attack_names[1]})
            # Per-point threshold attack, or white-box attack
            elif attack_res in ["threshold_perpoint", "affinity", "permutation_invariant","single_update_loss","single_update_threshold","single_update_perpoint","epoch_meta", "combine"]:
                for ratio in logger['result'][attack_res]:
                    if self.ratios_wanted is not None and ratio not in self.ratios_wanted:
                        continue
                    victim_results = logger['result'][attack_res][ratio]['victim_acc']
                    for results in victim_results:
                        if type(results) == list:
                            for epoch, result in enumerate(results):
                                self.df.append({
                                    self.columns[0]: float(ratio),
                                    # Temporary (below) - ideally all results should be in [0, 100] across entire module
                                    self.columns[1]: result,  # * 100,
                                    self.columns[2]: title_prefix + attack_names,
                                    self.columns[3]: epoch + 1})
                        else:
                            self.df.append({
                                self.columns[0]: float(ratio),
                                # Temporary (below) - ideally all results should be in [0, 100] across entire module
                                self.columns[1]: results,  # * 100,
                                self.columns[2]: title_prefix + attack_names})
            else:
                warnings.warn(warning_string(
                    f"\nAttack type {attack_res} not supported\n"))
        if len(self.df) == 0:
            raise ValueError(
                "None of the attacks in given results are supported for plotting")

    def get_appropriate_plotter_fn(self, plot_type):
        plotter_fn = self.supported_plot_types.get(plot_type, None)
        if plotter_fn is None:
            raise ValueError("Requested plot-type not supported")
        return plotter_fn

    def _graph_specific_options(self, graph, title='',
                                darkplot=True, dash=True):
        graph.set_title(title)
        # Add dividing line in centre
        lower, upper = plt.gca().get_xlim()
        if dash:
            midpoint = (lower + upper) / 2
            plt.axvline(x=midpoint,
                        color='white' if darkplot else 'black',
                        linewidth=1.0, linestyle='--')
        # Make sure axis label not cut off
        plt.tight_layout()

        if self.no_legend:
            plt.legend([],[], frameon=False)

    # Box plot, returns a graph object given a logger object
    def boxplot(self, title='', darkplot=True, dash=True):
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        graph = seaborn.boxplot(
            x=self.columns[0], y=self.columns[1],
            hue=self.columns[2], data=self.df)
        # Distinguishing accuracy range
        # TODO: Make this generic (to support loss values etc)
        graph.set(ylim=(45, 101))

        self._graph_specific_options(graph, title, darkplot, dash)

        return graph

    # Violin plot, returns a graph object given a logger object
    def violinplot(self, title='', darkplot=True, dash=True):
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        graph = seaborn.violinplot(
            x=self.columns[0], y=self.columns[1],
            hue=self.columns[2], data=self.df)

        self._graph_specific_options(graph, title, darkplot, dash)

        return graph

    # Regression plot, returns a graph object given a logger object
    # This plot does not take hues
    def regplot(self, title='', darkplot=True, dash=True):
        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        graph = seaborn.regplot(
            x=self.columns[0], y=self.columns[1], data=self.df)

        self._graph_specific_options(graph, title, darkplot, dash)

        return graph

    # Plot values (attack results, etc) across time (model training
    def lineplot(self, title='', darkplot=True, dash=False):
        assert self.columns[3] in self.df, "Epoch column not found"

        if darkplot:
            # Set dark background
            plt.style.use('dark_background')
        graph = seaborn.lineplot(
            x=self.columns[3],
            y=self.columns[1],
            hue=self.columns[2],
            data=self.df
        )
        self._graph_specific_options(graph, title, darkplot, dash)

        return graph
