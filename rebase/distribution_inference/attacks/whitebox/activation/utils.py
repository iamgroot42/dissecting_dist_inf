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
