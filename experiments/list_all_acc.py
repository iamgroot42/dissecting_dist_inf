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
        inner_path = os.path.join(args.dir, r, "adv_train_16")
        # inner_path = os.path.join(args.dir, r)
        for m in tqdm(os.listdir(inner_path), desc=f"Ratio {r}"):
            if not os.path.isdir(os.path.join(inner_path, m)):
                names.append(m)
    # print(names[0])
    # exit(0)
    names = [float(x.split("_adv")[0].split(".")[1]) for x in names] # For CelebA adv models
    # names = [float(x.split(".")[1]) for x in names] # For Census19, BoneAge
    #names = [100 * float(x.split("_")[1]) for x in names] # For CelebA
    # names = [100 * float(x.split("_adv")[1]) for x in names] # For CelebA adv models
    print(np.average(names))
    print(np.std(names))
