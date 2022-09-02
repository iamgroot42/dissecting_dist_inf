"""
    Look at experiment log files and get n_leaked values
"""
from simple_parsing import ArgumentParser
import pandas as pd
import json
from distribution_inference.nleaked.nleaked import Regression, process_logfile_for_neffs, BinaryRatio, GraphBinary


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
    parser.add_argument("--type",
                        choices=["mean", "median"],
                        default="mean",
                        help="Which statistic to report",
                        type=str)
    args = parser.parse_args()

    df = []
    is_regression = False
    is_graph = False
    for path in args.log_path:
        # Open log file
        logger = json.load(open(path, 'r'))

        alpha_0 = float(logger['attack_config']
                        ['train_config']['data_config']['value'])
        if logger['attack_config']['white_box'] is not None and logger['attack_config']['white_box']['regression_config'] is not None:
            is_regression = True

        if logger['attack_config']['train_config']['data_config']['name'] in ['arxiv']:
            is_graph = True

        for attack_res in logger['result']:
            if args.wanted is not None and attack_res not in args.wanted:
                print(f"Not plotting {attack_res}")
                continue

            # Process and collect results
            df = process_logfile_for_neffs(
                df, logger, attack_res, args.ratios, is_regression=is_regression)

    # Convert data to dataframe
    df = pd.DataFrame(df)

    # Apply n-leaked formula on each row
    if is_regression:
        # Ignore 0 and 1 ratios in DF
        df = df[df['prop_val'] != 0]
        df = df[df['prop_val'] != 1]
        # Take median per trial
        df = df.groupby(['prop_val']).min().reset_index()
        # df = df.groupby(['prop_val']).median().reset_index()
        df['n_leaked'] = df.apply(lambda row: Regression(
            row['prop_val']).get_n_effective(row['acc_or_loss']), axis=1)
        print(df)
        exit(0)
    else:
        # Replace very-high values with acc corresponding to getting only 1 predictiong wrong i.e. 1/500
        df.loc[df['acc_or_loss'] == 100, 'acc_or_loss'] = 100 - 2e-3
        # Take median per trial
        df = df.groupby(['prop_val']).median().reset_index()

        # Convert < 50 to 50
        df.loc[df['acc_or_loss'] < 50, 'acc_or_loss'] = 50

        # If graph-based, use appropriate formula
        if is_graph:
            df['n_leaked'] = df.apply(lambda row: GraphBinary(
                alpha_0, row['prop_val']).get_n_effective(row['acc_or_loss'] / 100.0,), axis=1)
        else:
            df['n_leaked'] = df.apply(lambda row: BinaryRatio(
                alpha_0, row['prop_val']).get_n_effective(row['acc_or_loss'] / 100.0,), axis=1)

    print(df)

    if args.type == "mean":
        print("%.1f" % df["n_leaked"].mean())
    else:
        print("%.1f" % df["n_leaked"].median())
