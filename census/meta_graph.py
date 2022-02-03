"""
    Graph-based meta-classifier approach that uses affinity scores
    across different layers to perform property inference.
"""
import utils
from data_utils import SUPPORTED_PROPERTIES
from model_utils import get_models_path, get_models, convert_to_torch
from data_utils import CensusWrapper
import argparse
from tqdm import tqdm
import numpy as np
import torch as ch


def make_activation_data(models_pos, models_neg, seed_data, detach=True, verbose=True):
    # Construct affinity graphs
    pos_model_scores = utils.make_affinity_features(
        models_pos, seed_data,
        detach=detach, verbose=verbose)
    neg_model_scores = utils.make_affinity_features(
        models_neg, seed_data,
        detach=detach, verbose=verbose)
    # Convert all this data to loaders
    X = ch.cat((pos_model_scores, neg_model_scores), 0)
    Y = ch.cat((ch.ones(len(pos_model_scores)),
                ch.zeros(len(neg_model_scores))))
    return X, Y


def convert_to_loaders_wrapper(models_pos, models_neg, seed_data,
                               batch_size=64, shuffle=True,
                               verbose=True, detach=True):
    X, Y = make_activation_data(
        models_pos, models_neg, seed_data, verbose=verbose, detach=detach)
    dataset = ch.utils.data.TensorDataset(X, Y)
    loader = ch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,)
    return loader


def optimal_data_for_meta(meta, models_pos, models_neg,
                          data, steps, step_size):
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
        X_all, y_all = make_activation_data(
            models_pos, models_neg, data_copy, detach=False,
            verbose=False)

        # Sample 90% of the models at random
        ratio = 0.9
        random_indices = np.random.choice(
            len(X_all), int(len(X_all) * ratio), replace=False)
        # Also shift to GPU
        X_all = X_all[random_indices].cuda()
        y_all = y_all[random_indices].cuda()

        # Get meta-classifier's outputs for sampled data
        y_pred = meta(X_all)[:, 0]

        # Compute loss
        loss = loss_fn(y_pred, y_all)

        # Compute gradients
        loss.backward()

        # Updata data
        data_copy.data -= step_size * data_copy.grad.data

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
    parser.add_argument('--restart_meta', action="store_true",
                        help='Re-init meta-classifier after every rum')
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
    parser.add_argument('--coordinate', action="store_true",
                        help='Use coordinate descent')
    parser.add_argument('--testing', action="store_true",
                        help='Testing mode')
    parser.add_argument('--coordinate_steps', type=int, default=10)
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.testing:
        num_train, num_test = 60, 10
    else:
        num_train, num_test = 2000, 1000

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
        "adv", args.filter, d_0), n_models=num_train, shuffle=True)
    pos_models_test = get_models(get_models_path(
        "victim", args.filter, d_0), n_models=num_test, shuffle=False)
    # Convert to PyTorch models
    pos_models = convert_to_torch(pos_models)
    pos_models_test = convert_to_torch(pos_models_test)

    # Wrapper for optimal data
    def optimal_data_wrapper(meta, models_pos, models_neg, data):
        return optimal_data_for_meta(meta, models_pos, models_neg,
                                     data, args.steps, args.step_size)

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
        # Load up positive-label models
        neg_models = get_models(get_models_path(
            "adv", args.filter, tg), n_models=num_train, shuffle=True)
        neg_models_test = get_models(get_models_path(
            "victim", args.filter, tg), n_models=num_test, shuffle=False)
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

            # Make a basic meta-classifier
            meta_clf = utils.AffinityMetaClassifier(3160, 3)
            meta_clf = meta_clf.cuda()

            # Coordinate descent training
            if args.coordinate:
                all_accs = utils.coordinate_descent_new(
                    (pos_models_train, neg_models_train),
                    (pos_models_test, neg_models_test),
                    num_features=3160, num_layers=3,
                    get_features=convert_to_loaders_wrapper,
                    meta_train_args={"epochs": args.epochs,
                                     "batch_size": args.batch_size},
                    gen_optimal_fn=optimal_data_wrapper,
                    seed_data=seed_data,
                    n_times=args.coordinate_steps,
                    restart_meta=args.restart_meta,
                    )
                tgt_data.append(all_accs)
                print("Test accuracy", all_accs)
            else:
                # Construct affinity graphs
                train_loader = convert_to_loaders_wrapper(
                    pos_models_train, neg_models_train, seed_data)
                test_loader = convert_to_loaders_wrapper(
                    pos_models_test, neg_models_test, seed_data)

                # Train meta_clf
                _, acc = utils.train(meta_clf, (train_loader, test_loader),
                                     epoch_num=100, expect_extra=False,
                                     verbose=False)
                tgt_data.append(acc)
                print("Test accuracy: %.3f" % acc)

        data.append(tgt_data)

    # Print data
    # log_path = os.path.join("log/meta/", args.filter, "meta_act_graph_result")
    # if not os.path.isdir(log_path):
    #     os.makedirs(log_path)
    # with open(os.path.join(log_path, "-".join([args.filter, args.d_0])), "a") as wr:
        # for i, tup in enumerate(data):
        #     print(targets[i], tup)
            # wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
    for i, tup in enumerate(data):
        print(targets[i], tup)
