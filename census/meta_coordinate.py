"""
    Meta-classifier experiment using coordinate descent to optimize for input data
    as well as generated data. Work in Progress.
"""
import utils
from data_utils import SUPPORTED_PROPERTIES
from typing import List
from model_utils import get_models_path, get_models, convert_to_torch, ACTIVATION_DIMS, PortedMLPClassifier
from data_utils import CensusWrapper
import argparse
from tqdm import tqdm
import numpy as np
import torch as ch


def get_model_activation_representations(
        models: List[PortedMLPClassifier],
        data, label, detach: bool = True,
        verbose: bool = True):
    w = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator)
    for model in iterator:
        activations = model(data, get_all=True,
                            detach_before_return=detach)

        w.append([act.float() for act in activations])
    labels = np.array([label] * len(w))
    labels = ch.from_numpy(labels)

    # Make numpy object (to support sequence-based indexing)
    w = np.array(w, dtype=object)

    # Get dimensions of feature representations
    dims = [x.shape[1] for x in w[0]]

    return w, labels, dims


def optimal_data_for_meta(meta, models_1, models_2,
                          data, get_activation_fn, steps,
                          step_size):
    """
        Generate data that leads to maximum accuracy
        for given meta-classifier, using current models
        as activation generators.
    """
    # Set meta-classifier to inference mode
    meta.eval()

    # Set up loss function
    loss_fn = ch.nn.BCEWithLogitsLoss()

    # Creat copy for gradient descent
    data_copy = data.clone().detach().cuda().requires_grad_(True)

    iterator = tqdm(range(steps))
    for i in iterator:
        # Zero out meta-classifier's gradients
        meta.zero_grad()

        # Get activation representations for each model
        X_all, y_all = utils.wrap_data_for_act_meta_clf(
                            models_1, models_2,
                            data_copy, get_activation_fn,
                            detach=False)

        # Sample 50% of the model at random
        ratio = 0.5
        random_indices = np.random.choice(
            len(X_all[0]), int(len(X_all[0]) * ratio), replace=False)
        # Also shift to GPU
        X_all = [x[random_indices].cuda() for x in X_all]
        y_all = y_all[random_indices].cuda()

        # Get meta-classifier's outputs for sampled data
        y_pred = meta(X_all)[:, 0]

        # Compute loss
        loss = loss_fn(y_pred, y_all)

        # Compute gradients
        loss.backward()

        # Updata data
        data_copy.data += step_size * data_copy.grad.data

        # Log loss
        iterator.set_description(f'Generating Data | Loss: {loss.item():.4f}')

    return data_copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample', type=int, default=800,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200,
                        help="Number of epochs to train meta-classifier")
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--d_0', default="0.5", help='ratios to use for D_0')
    parser.add_argument('--trg', default=None, help='target ratios')
    parser.add_argument('--n_samples', default=80,
                        type=int,
                        help='number of examples to use for activations')
    parser.add_argument('--ntimes', type=int, default=1,
                        help='number of repetitions for experiments')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=1e-1)
    args = parser.parse_args()
    utils.flash_utils(args)

    d_0 = args.d_0
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

    # Fetch some data from both one of the ratios, get its activations
    print(f"Using {args.n_samples} samples to compute activations")

    # Load up positive-label models
    pos_models = get_models(get_models_path(
        "adv", args.filter, d_0), n_models=2000, shuffle=True)
    pos_models_test = get_models(get_models_path(
        "victim", args.filter, d_0), shuffle=False)
    # Convert to PyTorch models
    pos_models = convert_to_torch(pos_models)
    pos_models_test = convert_to_torch(pos_models_test)

    # Layer-wise dimensions for meta-classifier
    reduction_dims = [8, 4, 2, 1]

    # Will generate from both distributions- make half
    args.n_samples = int(args.n_samples // 2)
    # Pseudo-attributes (args) for perf_gen
    args.use_normal = None
    args.use_natural = None
    args.start_natural = None
    args.latent_focus = None
    args.use_best = False
    args.constrained = False
    args.r = 1.0
    args.r2 = 1.0

    # Wrapper for data generation
    def data_gen_wrapper(meta, models_1, models_2,
                         data, get_activation_fn):
        return optimal_data_for_meta(meta, models_1, models_2,
                                     data, get_activation_fn,
                                     args.steps, args.step_size)

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
                custom_limit=int(args.n_samples // 4))
            _, (x_te_2, _), _ = ds_2.load_data(
                custom_limit=int(args.n_samples // 4))
            X_te = np.concatenate((x_te_1, x_te_2), axis=0)
            return ch.from_numpy(X_te).float().cuda()

        tgt_data = []
        # Load up positive-label models
        neg_models = get_models(get_models_path(
            "adv", args.filter, tg), n_models=2000, shuffle=True)
        neg_models_test = get_models(get_models_path(
            "victim", args.filter, tg), shuffle=False)
        # Convert to PyTorch models
        neg_models = convert_to_torch(neg_models)
        neg_models_test = convert_to_torch(neg_models_test)

        for i in range(args.ntimes):
            # Get some seed data for the meta-classifier
            seed_data = get_seed_data()

            # Random shuffles
            shuffled_1 = np.random.permutation(len(pos_models))
            pos_models_train = pos_models[shuffled_1[:args.train_sample]]

            shuffled_2 = np.random.permutation(len(neg_models))
            neg_models_train = neg_models[shuffled_2[:args.train_sample]]

            val_data = None
            if args.val_sample > 0:
                pos_models_val = pos_models[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                neg_models_val = neg_models[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]
                val_data = (neg_models_val, pos_models_val)

            # Coordinate descent training
            (best_tacc, best_clf), (tacc, clf), all_accs = utils.coordinate_descent(
                models_train=(neg_models_train, pos_models_train),
                models_val=val_data,
                models_test=(neg_models_test, pos_models_test),
                dims=ACTIVATION_DIMS,
                reduction_dims=reduction_dims,
                get_activation_fn=get_model_activation_representations,
                n_samples=args.n_samples * 2,
                meta_train_args={"epochs": args.epochs,
                                 "batch_size": args.batch_size},
                gen_optimal_fn=data_gen_wrapper,
                seed_data=seed_data,
                n_times=10
            )
            print("Accuracies across all rounds", all_accs)

            tgt_data.append(tacc)
            print("Test accuracy: %.3f" % tacc)
        data.append(tgt_data)

    # Print data
    # log_path = os.path.join("log/meta/", args.filter, "meta_act_result")
    # if not os.path.isdir(log_path):
    #     os.makedirs(log_path)
    # with open(os.path.join(log_path, "-".join([args.filter, args.d_0])), "a") as wr:
    #     for i, tup in enumerate(data):
    #         print(targets[i], tup)
    #         wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
