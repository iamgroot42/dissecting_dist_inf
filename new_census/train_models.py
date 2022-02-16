from tqdm import tqdm
import os
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import model_utils
import utils
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


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
    args = parser.parse_args()
    utils.flash_utils(args)

    ds = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.ratio),
        split=args.split,
        drop_senstive_cols=args.drop_senstive_cols)

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

        clf = model_utils.get_model()

        # Wrapper to ignore convergence warnings
        @ignore_warnings(category=ConvergenceWarning)
        def train(model):
            model.fit(x_tr, y_tr.ravel())
            return model

        # TODO (for later): Adjust model dirs to be different
        # if drop_senstive_cols is True

        clf = train(clf)
        train_acc = 100 * clf.score(x_tr, y_tr.ravel())
        test_acc = 100 * clf.score(x_te, y_te.ravel())
        if args.verbose:
            print("Classifier %d : Train acc %.2f , Test acc %.2f\n" %
                  (i, train_acc, test_acc))
        save_path = model_utils.get_models_path(
            args.filter, args.split, args.ratio)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        model_utils.save_model(clf, os.path.join(save_path,
                                                 str(i + args.offset) + "_%.2f" % test_acc))
