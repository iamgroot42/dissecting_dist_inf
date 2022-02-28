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
    parser.add_argument('--eval_only', action="store_true",
                        help="Only loading up model and evaluating")
    parser.add_argument('--model_path', help="Path to saved model")
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
    num_per_dist = None
    for ratio in SUPPORTED_RATIOS:
        # Load up data for this ratio
        if not args.eval_only:
            train_and_val_w, _, dims = get_model_representations(
                get_models_path("adv", args.filter, ratio), 0, args.first_n,
                n_models=n_models)
            # Shuffle and divide train data into train and val
            shuffled = np.random.permutation(len(train_and_val_w))
            train_w = train_and_val_w[shuffled[:num_train]]
            val_w = train_and_val_w[shuffled[num_train:num_train+num_val]]

            X_train.append(train_w)
            X_val.append(val_w)
            Y_train += [float(ratio)] * len(train_w)
            Y_val += [float(ratio)] * len(val_w)

        test_w, _, dims = get_model_representations(
            get_models_path("victim", args.filter, ratio), 0, args.first_n,
            n_models=n_models)
        X_test.append(test_w)
        Y_test += [float(ratio)] * len(test_w)

        # Make sure same number of models read per distribution
        if num_per_dist:
            assert num_per_dist == len(test_w)
        num_per_dist = len(test_w)

    # Prepare for PIM
    if not args.eval_only:
        X_train = np.concatenate(X_train)
        X_val = np.concatenate(X_val)
        Y_train = ch.from_numpy(np.array(Y_train)).float()
        Y_val = ch.from_numpy(np.array(Y_val)).float()

    X_test = np.concatenate(X_test)
    Y_test = ch.from_numpy(np.array(Y_test)).float()

    # Batch layer-wise inputs
    print("Batching data: hold on")
    if not args.eval_only:
        X_train = utils.prepare_batched_data(X_train)
        X_val = utils.prepare_batched_data(X_val)
    X_test = utils.prepare_batched_data(X_test)

    # Train meta-classifier model
    metamodel = utils.PermInvModel(dims, dropout=0.5)

    if args.eval_only:
        # Load model
        metamodel.load_state_dict(ch.load(args.model_path))
        # Evaluate
        metamodel = metamodel.cuda()
        loss_fn = ch.nn.MSELoss(reduction='none')
        _, losses, preds = utils.test_meta(
            metamodel, loss_fn, X_test, Y_test.cuda(),
            args.batch_size, None,
            binary=True, regression=True, gpu=True,
            combined=True, element_wise=True,
            get_preds=True)
        y_np = Y_test.numpy()
        losses = losses.numpy()
        print("Mean loss: %.4f" % np.mean(losses))
        # Get all unique ratios in GT, and their average losses from model
        ratios = np.unique(y_np)
        losses_dict = {}
        for ratio in ratios:
            losses_dict[ratio] = np.mean(losses[y_np == ratio])
        print(losses_dict)
        # Conctruct a matrix where every (i, j) entry is the accuracy
        # for ratio[i] v/s ratio [j], where whichever ratio is closer to the 
        # ratios is considered the "correct" one
        # Assume equal number of models per ratio, stored in order of
        # SUPPORTED_RATIOS
        acc_mat = np.zeros((len(ratios), len(ratios)))
        for i in range(acc_mat.shape[0]):
            for j in range(i + 1, acc_mat.shape[0]):
                # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
                gt_z = (Y_test[num_per_dist * i:num_per_dist * (i + 1)].numpy() == float(ratios[j]))
                gt_o = (Y_test[num_per_dist * j:num_per_dist * (j + 1)].numpy() == float(ratios[j]))
                # Get relevant preds
                pred_z = preds[num_per_dist * i:num_per_dist * (i + 1)]
                pred_o = preds[num_per_dist * j:num_per_dist * (j + 1)]
                pred_z = (pred_z >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
                pred_o = (pred_o >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
                # Compute accuracies and store
                acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate((pred_z, pred_o), 0)
                acc_mat[i, j] = np.mean(acc)
        print(acc_mat)
    else:
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
