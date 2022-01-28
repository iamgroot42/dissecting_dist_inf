"""
    Meta-classifier experiment using activation trends.
"""
import utils
from data_utils import SUPPORTED_PROPERTIES, CensusWrapper
from model_utils import get_models_path, get_models, layer_output
from data_utils import CensusWrapper
import argparse
from tqdm import tqdm
import numpy as np
import torch as ch
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def epoch_strategy(tg, args):
    return args.epochs


def get_model_activation_representations(models, data, label):
    w = []
    for model in tqdm(models):
        activations = layer_output(data, model, layer=4, get_all=True)
        w.append([ch.from_numpy(act).float() for act in activations])
    labels = np.array([label] * len(w))
    labels = ch.from_numpy(labels)

    # Make numpy object (to support sequence-based indexing)
    w = np.array(w, dtype=object)

    # Get dimensions of feature representations
    dims = [x.shape[1] for x in w[0]]

    return w, labels, dims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample', type=int, default=800,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=1000)
    # Sex: 1000 epochs, 1e-3
    # Race: 500* epochs, 1e-3
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Number of epochs to train meta-classifier")
    parser.add_argument('--ntimes', type=int, default=10,
                        help='number of repetitions for multimode')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--d_0', default="0.5", help='ratios to use for D_0')
    parser.add_argument('--trg', default=None, help='target ratios')
    parser.add_argument('--no_alignment', action="store_true",
                        help='do not align features across models')
    args = parser.parse_args()
    utils.flash_utils(args)

    d_0 = args.d_0
    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    # Only look at multiples of 0.10
    # targets = filter(lambda x: x != d_0 and int(float(x) * 10) ==
    #                 float(x) * 10, os.listdir(get_models_path(args.filter, "adv")))
    if args.trg is None:
        targets = sorted(['0.2,0.5', '0.5,0.2', '0.1,0.5'])
    else:
        lst = eval(args.trg)
        targets = []
        for i in lst:
            if type(i) is list:
                i = [str(x) for x in i]
                targets.append(','.join(i))

            else:
                targets.append(str(i))
        targets = sorted(targets)

    # targets = sorted(list(targets))

    # Fetch some data from both one of the ratios, get its activations
    ds = CensusWrapper(
            filter_prop=args.filter,
            ratio=0.5, split="adv")
    _, (data_for_features, _), _ = ds.load_data(custom_limit=64)
    n_samples = len(data_for_features)
    print(f"Using {n_samples} samples to compute activations")

    # Load up positive-label models
    pos_models = get_models(get_models_path(
        "adv", args.filter, d_0), n_models=2000, shuffle=True)
    pos_models_test = get_models(get_models_path(
        "victim", args.filter, d_0), shuffle=False)
    # Load up positive-label test data
    pos_w, pos_labels, _ = get_model_activation_representations(
        pos_models, data_for_features, 1)
    pos_w_test, pos_labels_test, dims = get_model_activation_representations(
        pos_models_test, data_for_features, 1)

    if not args.no_alignment:
        # Use first adversary model as reference for alignment
        alignment_ref = pos_w[0]

        # Align the rest of features
        pos_w[1:] = utils.align_all_features(alignment_ref, pos_w[1:])
        pos_w_test = utils.align_all_features(alignment_ref, pos_w_test)

    # Batch up data
    pos_w_test = utils.prepare_batched_data(pos_w_test)

    reduction_dims = [8, 4, 2, 1]

    data = []
    for tg in targets:
        tgt_data = []
        # Load up positive-label models
        neg_models = get_models(get_models_path(
            "adv", args.filter, tg), n_models=2000, shuffle=True)
        net_models_test = get_models(get_models_path(
            "victim", args.filter, tg), shuffle=False)
        # Load up negative-label train, test data
        neg_w, neg_labels, _ = get_model_activation_representations(
            neg_models, data_for_features, 0)
        neg_w_test, neg_labels_test, _ = get_model_activation_representations(
            net_models_test, data_for_features, 0)

        if not args.no_alignment:
            # Align features
            neg_w = utils.align_all_features(alignment_ref, neg_w)
            neg_w_test = utils.align_all_features(alignment_ref, neg_w_test)

        # Generate test set
        neg_w_test = utils.prepare_batched_data(neg_w_test)
        X_te = [ch.cat((x, y), 0) for x, y in zip(pos_w_test, neg_w_test)]
        Y_te = ch.cat((pos_labels_test, neg_labels_test)).float()

        for i in range(args.ntimes):
            # Random shuffles
            shuffled_1 = np.random.permutation(len(pos_labels))
            pp_x = pos_w[shuffled_1[:args.train_sample]]
            pp_y = pos_labels[shuffled_1[:args.train_sample]]

            shuffled_2 = np.random.permutation(len(neg_labels))
            np_x = neg_w[shuffled_2[:args.train_sample]]
            np_y = neg_labels[shuffled_2[:args.train_sample]]

            # Combine them together
            pp_x = utils.prepare_batched_data(pp_x)
            np_x = utils.prepare_batched_data(np_x)
            X_tr = [ch.cat((x, y), 0) for x, y in zip(pp_x, np_x)]
            Y_tr = ch.cat((pp_y, np_y)).float()

            val_data = None
            if args.val_sample > 0:
                pp_val_x = pos_w[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_x = neg_w[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                pp_val_y = pos_labels[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_y = neg_labels[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                # Combine them together
                pp_val_x = utils.prepare_batched_data(pp_val_x)
                np_val_x = utils.prepare_batched_data(np_val_x)
                X_val = [ch.cat((x, y), 0) for x, y in zip(pp_val_x, np_val_x)]
                Y_val = ch.cat((pp_val_y, np_val_y)).cuda().float()

                val_data = (X_val, Y_val)

            metamodel = utils.ActivationMetaClassifier(
                n_samples, dims,
                reduction_dims=reduction_dims)
            metamodel = metamodel.cuda()
            metamodel = ch.nn.DataParallel(metamodel)

            # Train PIM - retry with different init if fails to converge
            clf, tacc = utils.train_meta_model(
                metamodel,
                (X_tr, Y_tr), (X_te, Y_te),
                epochs=epoch_strategy(tg, args),
                binary=True, lr=1e-3,
                regression=False,
                batch_size=args.batch_size,
                val_data=val_data, combined=True,
                eval_every=10, gpu=True)
            if tacc < 60:
                print("Bad seed- retrying training")
                metamodel = utils.ActivationMetaClassifier(
                    n_samples, dims,
                    reduction_dims=reduction_dims)
                metamodel = metamodel.cuda()
                metamodel = ch.nn.DataParallel(metamodel)
                clf, tacc = utils.train_meta_model(
                    metamodel,
                    (X_tr, Y_tr), (X_te, Y_te),
                    epochs=epoch_strategy(tg, args),
                    binary=True, lr=1e-3,
                    regression=False,
                    batch_size=args.batch_size,
                    val_data=val_data, combined=True,
                    eval_every=10, gpu=True)

            tgt_data.append(tacc)
            print("Test accuracy: %.3f" % tacc)
        data.append(tgt_data)

    # Print data
    log_path = os.path.join("log/meta/", args.filter, "meta_act_result")
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, "-".join([args.filter, args.d_0])), "a") as wr:
        for i, tup in enumerate(data):
            print(targets[i], tup)
            wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
