import torch as ch
import numpy as np
import os
import argparse
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model
from data_utils import SUPPORTED_RATIOS


def load_stuff(model_dir, args):
    max_read = 5 if args.testing else None
    dims, vecs = get_model_features(model_dir, first_n=args.first_n,
                                    shift_to_gpu=False, max_read=max_read)
    return dims, np.array(vecs, dtype=object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_sample', type=int, default=700)
    parser.add_argument('--val_sample', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--testing', action="store_true", help="Testing mode")
    parser.add_argument('--first_n', type=int, default=3,
                        help="Only consider first N layers")
    args = parser.parse_args()
    print(args)

    # Keep track of ratios to be used
    train_dirs = [
        os.path.join(BASE_MODELS_DIR, "victim/%s/" % z) for z in SUPPORTED_RATIOS]
    test_dirs = [
        os.path.join(BASE_MODELS_DIR, "adv/%s/" % z) for z in SUPPORTED_RATIOS]

    if args.testing:
        num_train, num_val = 3, 2
    else:
        num_train, num_val = args.train_sample, args.val_sample

    X_train, X_test = [], []
    X_val, Y_val = [], []
    Y_train, Y_test = [], []
    for ratio in SUPPORTED_RATIOS:
        # Load model weights, convert to features
        train_dir = os.path.join(BASE_MODELS_DIR, "victim/%s/" % ratio)
        test_dir = os.path.join(BASE_MODELS_DIR, "victim/%s/" % ratio)
        dims, vecs_train = load_stuff(train_dir, args)
        _, vecs_test = load_stuff(test_dir, args)

        # Create train/val split
        shuffled = np.random.permutation(len(vecs_train))
        vecs_train_use = vecs_train[shuffled[:num_train]]
        vecs_val = vecs_train[
            shuffled[num_train:num_train+num_val]]

        # Keep collecting data...
        X_train.append(vecs_train_use)
        X_val.append(vecs_val)
        X_test.append(vecs_test)

        # ...and labels
        Y_train += [float(ratio)] * len(vecs_train_use)
        Y_val += [float(ratio)] * len(vecs_val)
        Y_test += [float(ratio)] * len(vecs_test)

    # Prepare for PIM
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)

    Y_train = ch.from_numpy(np.array(Y_train)).float()
    Y_val = ch.from_numpy(np.array(Y_val)).float()
    Y_test = ch.from_numpy(np.array(Y_test)).float()

    # Train meta-classifier model
    metamodel = PermInvModel(dims)
    metamodel = metamodel.cuda()

    batch_size = 10 if args.testing else args.batch_size
    _, tloss = train_meta_model(
                     metamodel,
                     (X_train, Y_train),
                     (X_test, Y_test),
                     epochs=args.epochs, binary=True,
                     lr=0.001, batch_size=batch_size,
                     val_data=(X_val, Y_val),
                     regression=True,
                     eval_every=10, gpu=True)
    print("Test loss %.4f" % (tloss))

    # Save meta-model
    ch.save(metamodel.state_dict(), "./metamodel_%d_%.3f.pth" %
            (args.first_n, tloss))
    
    # Train: 0.00048, 
    # Test: 0.00386 , 