from tqdm import tqdm
import os
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import ch_model
import torch as ch
import numpy as np
import utils


#usage is the same as train_models
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str,
                        required=True,
                        choices=SUPPORTED_PROPERTIES,
                        help='while filter to use')
    parser.add_argument('--ratio',
                        default="0.5",
                        help='what ratio of the new sampled dataset should be true')
    parser.add_argument('--num', type=int, default=1000,
                        help='how many classifiers to train?')
    parser.add_argument('--split',
                        required=True,
                        choices=["adv", "victim"],
                        help='which split of data to use')
    parser.add_argument('--verbose', action="store_true",
                        help='print out per-classifier stats?')
    parser.add_argument('--drop_senstive_cols', action="store_true",
                        help='drop age/sex attributes during training?')
    parser.add_argument('--offset', type=int, default=0,
                        help='start counting from here when saving models')
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    utils.flash_utils(args)

    ds = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.ratio),
        split=args.split,
        drop_senstive_cols=args.drop_senstive_cols,
        scale=args.scale)

    iterator = range(1, args.num + 1)
    if not args.verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        if args.verbose:
            print("Training classifier %d" % i)

        # Sample to qualify ratio, ultimately coming from fixed split
        # Ensures non-overlapping data for target and adversary
        # All the while allowing variations in dataset locally

        (x_tr, y_tr), (x_te, y_te), cols = ds.load_data()
        clf = ch_model.get_model(n_inp=x_tr.shape[1])
        vloss, tacc,vacc = ch_model.train(clf,((x_tr.astype(np.float32), y_tr.astype(np.float32)), (x_te.astype(np.float32), y_te.astype(np.float32))))
        if args.verbose:
            print("Classifier %d : loss %.2f , Tran acc %.2f, Test acc %.2f\n" %
                  (i, vloss, tacc ,vacc))  
        save_path = ch_model.get_models_path(
            args.filter, args.split, args.ratio)
        if args.scale != 1.0:
            save_path = os.path.join(save_path,"sample_size_scale:{}".format(args.scale))
        if args.drop_senstive_cols:
            save_path = os.path.join(save_path,"drop")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        ch_model.save_model(clf, os.path.join(save_path,
                                                 str(i + args.offset) + "_%.2f" % vacc))
              
