import os
import numpy as np
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
    for r in ratios:
        if args.r:
            if r != args.r:
                continue
        for m in os.listdir(os.path.join(args.dir, r)):
            if not os.path.isdir(os.path.join(args.dir, r, m)):
                names.append(m)
    names = [float(x.split(".")[1]) for x in names]
    print(np.average(names))
    print(np.std(names))
