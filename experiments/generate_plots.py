# Sets font-type to 'Normal' to make it compatible with camera-ready versions
from distribution_inference.visualize.plothelper import PlotHelper
from simple_parsing import ArgumentParser
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = "Times New Roman"


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
                        default='$\alpha_1$',
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
    parser.add_argument("--colormap",
                        type=str,
                        default=None,
                        help="which colormap to use")
    parser.add_argument("--dash",
                        action="store_true",
                        help="add dashed line midway?",)
    parser.add_argument("--not_dodge",
                        action="store_true",
                        help="boxplots on same tick?",)
    parser.add_argument("--low_legend",
                        action="store_true",
                        help="Legend outside (below) graph?",)
    parser.add_argument("--pdf",
                        action="store_true",
                        help="Save PDF instead of PNG",)
    parser.add_argument("--skip_prefix",
                        action="store_true",
                        help="Skip prefix of attack name in plotting legend",)
    parser.add_argument("--skip_suffix",
                        action="store_true",
                        help="Skip suffix of logname in plotting legend",)
    parser.add_argument("--remove_legend_title",
                        action="store_true",
                        help="Remove legend title?",)
    parser.add_argument("--per_logfile_attacks",
                        action="store_true",
                        help="Extract one attack per logfile?",)
    parser.add_argument("--n_legend_cols",
                        type=int,
                        default=2,
                        help="Number of columns for graph legends",)
    parser.add_argument("--same_colors",
                        action="store_true",
                        help="Same color for box line/outliers (for boxplot)?",)
    args = parser.parse_args()

    # Columns for axis and names
    columns = [r'{}'.format(args.x), args.y, args.legend, "Epoch"]

    # Set color pallete
    color_options = {
        "green": sns.color_palette(["#228B22", "#90EE90"]),
        # "blue": sns.color_palette(["#0000CD", "#1E90FF", "#87CEEB"]),
        "blue": sns.color_palette(["#00deff", "#0091b8", "#004c6d", ]),
        "brown": sns.color_palette(["#F4A460", "#D2691E", "#8B4513"]),
        "purple": sns.color_palette(["#DDA0DD", "#9370DB", "#6A5ACD", "#4B0082"]),
        "gray": sns.color_palette(["#708090", "#B0C4DE"]),
        "sensible": sns.color_palette(["#003f5c", "#7a5195", "#ef5675", "#ffa600"]),
        "metrics": sns.color_palette(["#0087ab", "#00c7e5", "#de425b"]),
    }
    palette = color_options.get(args.colormap, None)

    # Create plothelper object
    plothelper = PlotHelper(paths=args.log_path,
                            columns=columns,
                            legend_titles=args.legend_titles,
                            attacks_wanted=args.wanted,
                            ratios_wanted=args.ratios,
                            no_legend=args.nolegend,
                            skip_prefix=args.skip_prefix,
                            skip_suffix=args.skip_suffix,
                            not_dodge=args.not_dodge,
                            low_legend=args.low_legend,
                            n_legend_cols=args.n_legend_cols,
                            palette=palette,
                            same_colors=args.same_colors,
                            per_logfile_attacks=args.per_logfile_attacks)
    plotter_fn = plothelper.get_appropriate_plotter_fn(args.plot)
    graph = plotter_fn(title=args.title,
                       darkplot=args.dark,
                       dash=args.dash)
    
    if args.remove_legend_title:
        graph.legend_.set_title(None)

    # Save plot
    suffix = "pdf" if args.pdf else "png"
    graph.figure.savefig(os.path.join('%s_%s.%s' %
                         (args.savepath, args.plot, suffix)))
