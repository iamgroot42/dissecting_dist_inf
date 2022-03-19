import torch as ch
import torch.nn as nn
import tqdm as tqdm

from distribution_inference.attacks.whitebox.affinity.affinity import AffinityMetaClassifier
from distribution_inference.training.core import train


def make_affinity_feature(model, data, use_logit: bool = False, detach: bool = True, verbose: bool = True):
    """
         Construct affinity matrix per layer based on affinity scores
         for a given model. Model them in a way that does not
         require graph-based models.
    """
    # Build affinity graph for given model and data
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Start with getting layer-wise model features
    model_features = model(data, get_all=True, detach_before_return=detach)
    layerwise_features = []
    for i, feature in enumerate(model_features):
        scores = []
        # Pair-wise iteration of all data
        for i in range(len(data)-1):
            others = feature[i+1:]
            scores += cos(ch.unsqueeze(feature[i], 0), others)
        layerwise_features.append(ch.stack(scores, 0))

    # If asked to use logits, convert them to probability scores
    # And then consider them as-it-is (instead of pair-wise comparison)
    if use_logit:
        logits = model_features[-1]
        probs = ch.sigmoid(logits)
        layerwise_features.append(probs)

    concatenated_features = ch.stack(layerwise_features, 0)
    return concatenated_features


def make_affinity_features(models, data, use_logit: bool = False,
                           detach: bool = True, verbose: bool = True):
    all_features = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Building affinity matrix")
    for model in iterator:
        affinity_feature = make_affinity_feature(
            model, data, use_logit=use_logit,
            detach=detach, verbose=verbose)
        all_features.append(affinity_feature)
    return ch.stack(all_features, 0)


def coordinate_descent(models_train, models_val,
                           num_features, num_layers,
                           get_features,
                           meta_train_args,
                           gen_optimal_fn, seed_data,
                           n_times: int = 10,
                           restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        num_layers: Number of layers of models used for activations
        get_features: Function that takes (models, data) as input and returns features
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = AffinityMetaClassifier(num_features, num_layers)
    metamodel = metamodel.cuda()

    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        train_loader = get_features(
            models_train[0], models_train[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])
        val_loader = get_features(
            models_val[0], models_val[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = AffinityMetaClassifier(num_features, num_layers)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        _, val_acc = train(metamodel, (train_loader, val_loader),
                           epoch_num=meta_train_args['epochs'],
                           expect_extra=False,
                           verbose=False)
        all_accs.append(val_acc)

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data)

    # Return all accuracies
    return all_accs
