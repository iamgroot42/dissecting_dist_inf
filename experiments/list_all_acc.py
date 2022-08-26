import os
import numpy as np
from tqdm import tqdm
from simple_parsing import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--dir",
        type=str, required=True)
    parser.add_argument(
        "--r",
        type=str,
        default=None)
    args = parser.parse_args()
    ratios = os.listdir(args.dir)
    names = []
    print("Reading filenames")
    for r in ratios:
        if args.r:
            if r != args.r:
                continue
        for m in tqdm(os.listdir(os.path.join(args.dir, r)), desc=f"Ratio {r}"):
            if not os.path.isdir(os.path.join(args.dir, r, m)):
                names.append(m)
    names = [float(x.split(".")[1]) for x in names]
    # names = [100 * float(x.split("_")[1]) for x in names]
    print(np.average(names))
    print(np.std(names))
