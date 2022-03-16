"""
    Joint meta-classifier approach that uses weight matrices along with
    affinity scores across different layers (for activations on given data)
    to perform property inference.
"""
import utils
from data_utils import SUPPORTED_PROPERTIES, CensusWrapper
from model_utils import get_models_path, get_model_representations, save_model, convert_to_torch, make_activation_data
import argparse
import numpy as np
import torch as ch
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample', type=int, default=800,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200,
                        help="Number of epochs to train meta-classifier")
    parser.add_argument('--start_n', type=int, default=0,
                        help="Only consider starting from this layer")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Use only first N layers' parameters")
    parser.add_argument('--ntimes', type=int, default=10,
                        help='number of repetitions for multimode')
    parser.add_argument('--n_samples', default=80,
                        type=int,
                        help='number of examples to use for activations')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--d_0', default="0.5", help='ratios to use for D_0')
    parser.add_argument('--trg', default=None, help='target ratios')
    parser.add_argument('--testing', action="store_true",
                        help='Testing mode')
    parser.add_argument('--save', action="store_false",
                        help='save model or not')
    parser.add_argument('--use_logit', action="store_true",
                        help='Also use logits as features')
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.testing:
        num_train, num_test = 60, 10
    else:
        num_train, num_test = 1000, 1000

    num_layers = 4 if args.use_logit else 3
    num_features = int((args.n_samples) * (args.n_samples - 1) / 2)
    print("Number of features for activation metaclf: %d" % num_features)

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

    # Load up positive-label test, test data
    pos_w, pos_labels, _, pos_w_ch = get_model_representations(
        get_models_path("adv", args.filter, d_0),
        1, args.first_n, n_models=num_train,
        fetch_models=True)
    pos_w_test, pos_labels_test, dims, pos_w_test_ch = get_model_representations(
        get_models_path("victim", args.filter, d_0),
        1, args.first_n, n_models=num_test,
        fetch_models=True)
    # Convert to torch models
    pos_w_ch = convert_to_torch(pos_w_ch)
    pos_w_test_ch = convert_to_torch(pos_w_test_ch)

    data = []
    for tg in targets:

        # Function to get seed data
        def get_seed_data():
            # Prepare data wrappers
            ds_1 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.d_0), split="adv")
            ds_2 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(tg), split="adv")
            # Fetch test data from ratio, convert to Tensor
            _, (x_te_1, _), _ = ds_1.load_data(
                custom_limit=int(args.n_samples // 8))
            _, (x_te_2, _), _ = ds_2.load_data(
                custom_limit=int(args.n_samples // 8))
            X_te = np.concatenate((x_te_1, x_te_2), axis=0)
            return ch.from_numpy(X_te).float().cuda()

        tgt_data = []
        # Load up negative-label train, test data
        neg_w, neg_labels, _, neg_w_ch = get_model_representations(
            get_models_path("adv", args.filter, tg),
            0, args.first_n, n_models=num_train,
            fetch_models=True)
        neg_w_test, neg_labels_test, _, neg_w_test_ch = get_model_representations(
            get_models_path("victim", args.filter, tg),
            0, args.first_n, n_models=num_test,
            fetch_models=True)
        # Convert to torch models
        neg_w_ch = convert_to_torch(neg_w_ch)
        neg_w_test_ch = convert_to_torch(neg_w_test_ch)

        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))
        Y_te = ch.cat((pos_labels_test, neg_labels_test)).cuda()

        # Batch layer-wise inputs (weights)
        print("Batching data: hold on")
        X_te = utils.prepare_batched_data(X_te)

        for i in range(args.ntimes):
            # Generate seed data
            seed_data = get_seed_data()

            # Random shuffles
            shuffled_1 = np.random.permutation(len(pos_labels))
            pp_x = pos_w[shuffled_1[:args.train_sample]]
            pp_y = pos_labels[shuffled_1[:args.train_sample]]
            pp_x_ch = pos_w_ch[shuffled_1[:args.train_sample]]

            shuffled_2 = np.random.permutation(len(neg_labels))
            np_x = neg_w[shuffled_2[:args.train_sample]]
            np_y = neg_labels[shuffled_2[:args.train_sample]]
            np_x_ch = neg_w_ch[shuffled_2[:args.train_sample]]

            # Combine them together
            X_tr = np.concatenate((pp_x, np_x))
            Y_tr = ch.cat((pp_y, np_y))

            val_data, X_val = None, None
            if args.val_sample > 0:
                pp_val_x = pos_w[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_x = neg_w[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]
                pp_val_x_ch = pos_w_ch[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_x_ch = neg_w_ch[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                pp_val_y = pos_labels[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_y = neg_labels[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                # Combine them together
                X_val = np.concatenate((pp_val_x, np_val_x))
                Y_val = ch.cat((pp_val_y, np_val_y))

                # Batch layer-wise inputs (weights)
                print("Batching data: hold on")
                X_val = utils.prepare_batched_data(X_val)
                Y_val = Y_val.float()

                # Get val data ready
                val_data = (X_val, Y_val)
                X_val, _ = make_activation_data(
                    pp_val_x_ch, np_val_x_ch, seed_data, use_logit=args.use_logit)

            # Get activations ready for meta-classifier
            X_act_train, _ = make_activation_data(
                pp_x_ch, np_x_ch, seed_data, use_logit=args.use_logit)
            X_act_test, _ = make_activation_data(
                pos_w_test_ch, neg_w_test_ch, seed_data, use_logit=args.use_logit)

            metamodel = utils.WeightAndActMeta(dims, num_features, num_layers)
            metamodel = metamodel.cuda()

            # Float data
            Y_tr = Y_tr.float()
            Y_te = Y_te.float()

            # Batch layer-wise inputs (weights)
            print("Batching data: hold on")
            X_tr = utils.prepare_batched_data(X_tr)

            # Train PIM
            clf, tacc = utils.train_meta_model(
                metamodel,
                (X_tr, Y_tr), (X_te, Y_te),
                epochs=args.epochs,
                binary=True, lr=1e-3,
                regression=False,
                batch_size=args.batch_size,
                val_data=val_data, combined=True,
                eval_every=10, gpu=True,
                train_acts=X_act_train,
                val_acts=X_val,
                test_acts=X_act_test)
            # if args.save:
                # save_path = os.path.join("log/meta/", args.filter, "meta_model_joint", "-".join(
                    # [args.d_0, str(args.start_n), str(args.first_n)]), tg)
                # if not os.path.isdir(save_path):
                    # os.makedirs(save_path)
                # save_model(clf, os.path.join(save_path, str(i) +
                                            #  "_%.2f" % tacc))
            tgt_data.append(tacc)
            print("Test accuracy: %.3f" % tacc)
        data.append(tgt_data)

    # Print data
    # log_path = os.path.join("log/meta/", args.filter, "meta_result")
    # if not os.path.isdir(log_path):
        # os.makedirs(log_path)
    # with open(os.path.join(log_path, "-".join([args.filter, args.d_0, str(args.start_n), str(args.first_n)])), "a") as wr:
        # for i, tup in enumerate(data):
        #     print(targets[i], tup)
            # wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
    for i, tup in enumerate(data):
        print(targets[i], tup)
