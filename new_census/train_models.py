from tqdm import tqdm
import os
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import model_utils
import utils


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #  Dataset-specific arguments
    parser.add_argument('--filter', type=str,
                        required=True,
                        choices=SUPPORTED_PROPERTIES,
                        help='while filter to use')
    parser.add_argument('--ratio',
                        default="0.5",
                        help='what ratio of the new sampled dataset should be true')
    parser.add_argument('--split',
                        required=True,
                        choices=["adv", "victim"],
                        help='which split of data to use')
    parser.add_argument('--drop_senstive_cols', action="store_true",
                        help='drop age/sex attributes during training?')
    parser.add_argument('--scale', type=float, default=1.0)
    #  Model-specific arguments
    parser.add_argument('--num', type=int, default=1000,
                        help='how many classifiers to train?')
    parser.add_argument('--verbose', action="store_true",
                        help='print out per-classifier stats?')
    parser.add_argument('--offset', type=int, default=0,
                        help='start counting from here when saving models')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for Optimizer")
    args = parser.parse_args()
    utils.flash_utils(args)

    # New Census dataset
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

        # Every call for get_loaders() will read a different subset of the data
        # So we don't need to redefine ds every time we train the model
        train_loader, test_loader, n_inp = ds.get_loaders(
            args.batch_size, get_num_features=True, squeeze=True)

        # Get model
        clf = model_utils.get_model(n_inp=n_inp)

        # Train model
        model, (vloss, vacc) = utils.train(
                                    clf, (train_loader, test_loader),
                                    lr=args.lr,
                                    epoch_num=args.epochs,
                                    weight_decay=1e-4,
                                    verbose=args.verbose,
                                    get_best=True)

        # Save model
        save_path = model_utils.get_models_path(
            args.filter, args.split, args.ratio)
        if args.scale != 1.0:
            save_path = os.path.join(
                save_path, "sample_size_scale:{}".format(args.scale))
        if args.drop_senstive_cols:
            save_path = os.path.join(save_path, "drop")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Save model
        save_path = os.path.join(save_path, str(
            i + args.offset) + "_%.2f.ch" % vacc)
        model_utils.save_model(clf, save_path)
