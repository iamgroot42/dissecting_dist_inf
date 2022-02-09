"""
    Meta-classifier experiment using Permutation Invariant Networks for direct regression.
"""
import utils
from data_utils import SUPPORTED_PROPERTIES, SUPPORTED_RATIOS
from model_utils import get_models_path, get_model_representations
import argparse
import numpy as np
import torch as ch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='US Census')
    parser.add_argument('--train_sample', type=int, default=700,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=50,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=1000)
    # Sex: 1000 epochs, 1e-3
    # Race: 500* epochs, 1e-3
    parser.add_argument('--epochs', type=int, default=300,
                        help="Number of epochs to train meta-classifier")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Use only first N layers' parameters")
    parser.add_argument('--testing', action="store_true", help="Testing mode")
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.testing:
        num_train, num_val = 3, 2
        n_models = 5
        SUPPORTED_RATIOS = SUPPORTED_RATIOS[:3]
    else:
        num_train, num_val = args.train_sample, args.val_sample
        n_models = 1000

    X_train, X_test = [], []
    X_val, Y_val = [], []
    Y_train, Y_test = [], []
    for ratio in SUPPORTED_RATIOS:
        # Load up data for this ratio
        train_and_val_w, _, dims = get_model_representations(
            get_models_path("adv", args.filter, ratio), 0, args.first_n,
            n_models=n_models)
        test_w, _, _ = get_model_representations(
            get_models_path("victim", args.filter, ratio), 0, args.first_n,
            n_models=n_models)

        # Shuffle and divide train data into train and val
        shuffled = np.random.permutation(len(train_and_val_w))
        train_w = train_and_val_w[shuffled[:num_train]]
        val_w = train_and_val_w[shuffled[num_train:num_train+num_val]]

        # Keep collecting data...
        X_train.append(train_w)
        X_val.append(val_w)
        X_test.append(test_w)

        # ...and labels
        Y_train += [float(ratio)] * len(train_w)
        Y_val += [float(ratio)] * len(val_w)
        Y_test += [float(ratio)] * len(test_w)

    # Prepare for PIM
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)

    Y_train = ch.from_numpy(np.array(Y_train)).float()
    Y_val = ch.from_numpy(np.array(Y_val)).float()
    Y_test = ch.from_numpy(np.array(Y_test)).float()

    # Batch layer-wise inputs
    print("Batching data: hold on")
    X_train = utils.prepare_batched_data(X_train)
    X_val = utils.prepare_batched_data(X_val)
    X_test = utils.prepare_batched_data(X_test)

    # Train meta-classifier model
    metamodel = utils.PermInvModel(dims, dropout=0.5)
    metamodel = metamodel.cuda()

    # Train PIM
    batch_size = 10 if args.testing else args.batch_size
    _, tloss = utils.train_meta_model(
        metamodel,
        (X_train, Y_train), (X_test, Y_test),
        epochs=args.epochs,
        binary=True, lr=0.001,
        regression=True,
        batch_size=batch_size,
        val_data=(X_val, Y_val), combined=True,
        eval_every=10, gpu=True)
    print("Test loss %.4f" % (tloss))

    # Save meta-model
    ch.save(metamodel.state_dict(), "./metamodel_%s_%.3f.pth" % (args.filter, tloss))

    # Results seem to get better as batch size increases
    # Sex
    # Best: python meta_regression.py --filter sex --epochs 200 --batch_size 4000
    # Train: 0.01842, 0.01648, 0.01476, 0.01362, 0.01863
    # Test:  0.14628, 0.16283, 0.16097, 0.17544, 0.15258

    # Race
    # Train: 0.01823, 0.01609, 0.01897, 0.01824, 0.01297
    # Test: 0.29089, 0.30709, 0.33302, 0.31789, 0.30140