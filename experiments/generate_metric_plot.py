# This script is used to generate lineplot of model performance metrics accross ratios
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
from scipy.stats import pearsonr
import os
from simple_parsing import ArgumentParser
import seaborn as sns
from distribution_inference.visualize.metricplotter import MetricPlotHelper
import seaborn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
mpl.rcParams['figure.dpi'] = 300
def lineplot(plotter, title='', darkplot=True, dash=False):
    assert plotter.columns[3] in plotter.df, "Epoch column not found"

    if darkplot:
        # Set dark background
        plt.style.use('dark_background')
    graph = seaborn.lineplot(
        x=plotter.columns[3],
        y=plotter.columns[1],
        data=plotter.df.query("Metric=='R_cross'"),
       # hue = plotter.columns[2],
        color="red"
    )
    graph.set_ylabel("R_cross")
    
    ax2 = graph.twinx()
    graph.legend(handles=[Line2D([], [], marker='_', color="r", label='R_Cross')])
    sns.move_legend(graph, "upper right", bbox_to_anchor=(0, 1))
    sns.lineplot(
        x=plotter.columns[3],
        y=plotter.columns[1],
        hue = plotter.columns[2],
        data=plotter.df.query("Metric!='R_cross'"), ax=ax2)
    ax2.set_ylabel("Accuracy (%)")
    plt.xticks(range(1,21))
    #ax2.legend(handles=[Line2D([], [], marker='_', color="r", label='R_Cross')])
    plotter._graph_specific_options(graph, title, darkplot, dash)

    return graph


if __name__ == "__main__":
    #Arguments for plotting
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--log_path",
                        nargs='+',
                        help="Specify file where results are stored",
                        type=str, required=True)
    parser.add_argument("--wanted",
                        nargs='+',
                        help="Specify which metric to plot",
                        type=str)
    parser.add_argument("--attack_path",
                        nargs='+',
                        help="Specify which attacks to plot",
                        type=str)
    parser.add_argument("--ratios",
                        nargs='+',
                        help="Specify which ratios to plot",
                        type=str)
    """
    parser.add_argument("--plot",
                        help="Specify plot type",
                        choices=['violin', 'box', 'reg', 'line'],
                        type=str,
                        required=True)
    """
    parser.add_argument("--savepath",
                        help="Specify save path (ending with /)",
                        type=str,
                        required=True)
    parser.add_argument("--title",
                        default='',
                        help="Plot title", type=str)
    parser.add_argument("--x",
                        default=r'$\alpha_1$',
                        help="Title for X-axis",
                        type=str)
    parser.add_argument("--y",
                        default='Accuracy (%)',
                        help="Title for Y-axis",
                        type=str)
    parser.add_argument("--legend",
                        default='Metric',
                        help="legend title",
                        type=str)
    parser.add_argument("--nolegend",
                        action="store_true",
                        help="Skip legend?",)
    parser.add_argument("--legend_titles",
                        nargs='+',
                        help="Titles for legends",
                        type=str)
    parser.add_argument("--dark",
                        action="store_true",
                        help="dark background")
    parser.add_argument("--dash",
                        action="store_true",
                        help="add dashed line midway?",)
    parser.add_argument("--pdf",
                        action="store_true",
                        help="Save PDF instead of PNG",)
    parser.add_argument("--skip_prefix",
                        action="store_true",
                        help="Skip prefix of attack name in plotting legend",)
    args = parser.parse_args()

    # Columns for axis and names
    columns = [args.x, args.y, args.legend, "Epoch"]

    # Create plothelper object
    plothelper = MetricPlotHelper(paths=args.log_path,
                            columns=columns,
                            legend_titles=args.legend_titles,
                            attack_paths = args.attack_path,
                            metrics_wanted=args.wanted,
                            ratios_wanted=args.ratios,
                            no_legend=args.nolegend,
                            skip_prefix=args.skip_prefix)
    plotter_fn = lineplot #plothelper.get_appropriate_plotter_fn(args.plot)
    graph = plotter_fn(plothelper,title=args.title,
                       darkplot=args.dark,
                       dash=args.dash)

    # Save plot
    suffix = "pdf" if args.pdf else "png"
    R_c = []
    af = []
    for i in range(1,21):
        R_c.append(np.mean(plothelper.df.query("Metric=='R_cross'").query("Epoch=={}".format(i)).to_numpy()[:,1]))

        af.append(np.mean(plothelper.df.query("Metric=='Activation-Correlation Meta-Classifier'").query("Epoch=={}".format(i)).to_numpy()[:,1]))
    print(pearsonr(R_c,af)[0])
    
    #graph.figure.savefig(os.path.join('%s_%s.%s' %
    #                     (args.savepath, "lineplot", suffix)))
    #graph.figure.savefig(os.path.join('%s_%s.%s' %
    #                     (args.savepath, args.plot, suffix)))
