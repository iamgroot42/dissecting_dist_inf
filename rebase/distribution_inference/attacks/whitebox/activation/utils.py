import numpy as np


def wrap_data(models_neg, models_pos,
              data, get_activation_fn,
              detach: bool = True):
    """
        Given models from two different distributions, get their
        activations on given data and activation-extraction function, and
        combine them into data-label format for a meta-classifier.
    """
    neg_w, neg_labels, _ = get_activation_fn(
        models_pos, data, 1, detach, verbose=False)
    pos_w, pos_labels, _ = get_activation_fn(
        models_neg, data, 0, detach, verbose=False)
    pp_x = prepare_batched_data(pos_w, verbose=False)
    np_x = prepare_batched_data(neg_w, verbose=False)
    X = [ch.cat((x, y), 0) for x, y in zip(pp_x, np_x)]
    Y = ch.cat((pos_labels, neg_labels))
    return X, Y.cuda().float()


def align_all_features(reference_point, features):
    """
        Perform layer-wise alignment of given features, using
        reference point. Return aligned features.
    """
    aligned_features = []
    for feature in tqdm(features, desc="Aligning features"):
        inside_feature = []
        for (ref_i, x_i) in zip(reference_point, feature):
            aligned_feature = x_i @ op_solution(x_i, ref_i)
            inside_feature.append(aligned_feature)
        aligned_features.append(inside_feature)
    return np.array(aligned_features, dtype=object)


def op_solution(x, y):
    """
        Return the optimal rotation to apply to x so that it aligns with y.
    """
    u, s, vh = np.linalg.svd(x.T @ y)
    optimal_x_to_y = u @ vh
    return optimal_x_to_y


def coordinate_descent(models_train, models_val,
                       models_test, dims, reduction_dims,
                       get_activation_fn,
                       n_samples, meta_train_args,
                       gen_optimal_fn, seed_data,
                       n_times: int = 10,
                       restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        dims: Dimensions of feature activations.
        reduction_dims: Dimensions for meta-classifier internal models.
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = ActivationMetaClassifier(
        n_samples, dims,
        reduction_dims=reduction_dims)
    metamodel = metamodel.cuda()

    best_clf, best_tacc = None, 0
    val_data = None
    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        X_tr, Y_tr = wrap_data_for_act_meta_clf(
            models_train[0], models_train[1], seed_data, get_activation_fn)
        X_te, Y_te = wrap_data_for_act_meta_clf(
            models_test[0], models_test[1], seed_data, get_activation_fn)
        if models_val is not None:
            val_data = wrap_data_for_act_meta_clf(
                models_val[0], models_val[1], seed_data, get_activation_fn)

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = ActivationMetaClassifier(
                n_samples, dims,
                reduction_dims=reduction_dims)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        metamodel.train()
        clf, tacc = train_meta_model(
            metamodel,
            (X_tr, Y_tr), (X_te, Y_te),
            epochs=meta_train_args['epochs'],
            binary=True, lr=1e-3,
            regression=False,
            batch_size=meta_train_args['batch_size'],
            val_data=val_data, combined=True,
            eval_every=10, gpu=True)
        all_accs.append(tacc)

        # Keep track of best model and latest model
        if tacc > best_tacc:
            best_tacc = tacc
            best_clf = clf

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data, get_activation_fn)

    # Return best and latest models
    return (best_tacc, best_clf), (tacc, clf), all_accs
