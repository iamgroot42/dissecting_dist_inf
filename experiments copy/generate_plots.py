import os
from simple_parsing import ArgumentParser

from distribution_inference.visualize.plothelper import PlotHelper


if __name__ == "__main__":
    #Arguments for plotting
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--log_path",
                        nargs='+',
                        help="Specify file where results are stored",
                        type=str, required=True)
    parser.add_argument("--wanted",
                        nargs='+',
                        help="Specify which attacks to plot",
                        type=str)
    parser.add_argument("--ratios",
                        nargs='+',
                        help="Specify which ratios to plot",
                        type=str)
    parser.add_argument("--plot",
                        help="Specify plot type",
                        choices=['violin', 'box', 'reg', 'line'],
                        type=str,
                        required=True)
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
                        default='Attack',
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
    args = parser.parse_args()

    # Columns for axis and names
    columns = [args.x, args.y, args.legend, "Epoch"]

    # Create plothelper object
    plothelper = PlotHelper(paths=args.log_path,
                            columns=columns,
                            legend_titles=args.legend_titles,
                            attacks_wanted=args.wanted,
                            ratios_wanted=args.ratios,
                            no_legend=args.nolegend)
    plotter_fn = plothelper.get_appropriate_plotter_fn(args.plot)
    graph = plotter_fn(title=args.title,
                       darkplot=args.dark,
                       dash=args.dash)

    # Save plot
    suffix = "pdf" if args.pdf else "png"
    graph.figure.savefig(os.path.join('%s_%s.%s' %
                         (args.savepath, args.plot, suffix)))
