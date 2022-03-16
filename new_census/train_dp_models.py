from tqdm import tqdm
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import ch_model
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
                        help="what ratio of the new sampled dataset"
                             "should be true")
    parser.add_argument('--split',
                        required=True,
                        choices=["adv", "victim"],
                        help='which split of data to use')
    parser.add_argument('--drop_senstive_cols', action="store_true",
                        help='drop age/sex attributes during training?')
    parser.add_argument('--scale', type=float, default=1.0)
    #  Training-specific arguments
    parser.add_argument('--num', type=int, default=1,
                        help='how many classifiers to train?')
    parser.add_argument('--verbose', action="store_true",
                        help='print out per-classifier stats?')
    parser.add_argument('--offset', type=int, default=0,
                        help='start counting from here when saving models')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help="Learning rate for GD")
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of epochs to train for")
    #  DP-specific arguments
    parser.add_argument('--epsilon', type=float,
                        default=0.1, help="Privacy budget")
    parser.add_argument('--delta', type=float,
                        default=1.3e-5, help="Delta for DP")
    parser.add_argument('--max_grad_norm', type=float, default=1,
                        help="The maximum L2 norm of per-sample gradients"
                        "before they are aggregated by the averaging step."
                        "Tuning MAX_GRAD_NORM is very important. Start with a"
                        "low noise multiplier like .1, this should give"
                        "comparable performance to a non-private model."
                        "Then do a grid search for the optimal MAX_GRAD_NORM"
                        "value. The grid can be in the range [.1, 10].")
    parser.add_argument('--physical_batch_size', type=int, default=500,
                        help="Peak memory is proportional to batch_size ** 2."
                        "This physical batch size should be set accordingly")
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

        # Get data loadeers
        train_loader, test_loader, n_inp = ds.get_loaders(
            args.batch_size, get_num_features=True)
        # Get model
        clf = ch_model.get_model(n_inp=n_inp)
        # Make sure model is compatible with DP training
        ch_model.validate_model(clf)
        # Train model with DP noise
        ch_model.opacus_stuff(clf, train_loader, test_loader, args)

        #  Will save models after hyper-parameters have been fixed
        """
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
        """
