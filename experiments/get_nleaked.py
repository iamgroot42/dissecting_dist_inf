"""
    Look at experiment log files and get n_leaked values
"""
from distribution_inference.nleaked.nleaked import process_logfile_for_neffs, BinaryRatio
from simple_parsing import ArgumentParser
import pandas as pd
import json


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
                        default="median",
                        help="Which statistic to report",
                        type=str)
    args = parser.parse_args()

    df = []
    for path in args.log_path:
        # Open log file
        logger = json.load(open(path, 'r'))

        alpha__0 = float(logger['attack_config']['train_config']['data_config']['value'])

        for attack_res in logger['result']:
            if args.wanted is not None and attack_res not in args.wanted:
                print(f"Not plotting {attack_res}")
                continue         

            # Process and collect results
            df = process_logfile_for_neffs(df, logger, attack_res, args.ratios)
    
    # Convert data to dataframe
    df = pd.DataFrame(df)

    print(df)

    # Maintain maximum per ratio
    df = df.groupby(['prop_val']).max().reset_index()

    # Apply n-leaked formula on each row
    df['n_leaked'] = df.apply(lambda row: BinaryRatio(alpha__0, row['prop_val']).get_n_effective(row['acc_or_loss'] / 100.0,), axis=1)

    if args.type == "mean":
        print("%.1f" % df["n_leaked"].mean())
    else:
        print("%.1f" % df["n_leaked"].median())
