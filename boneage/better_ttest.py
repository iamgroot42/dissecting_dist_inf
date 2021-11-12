from model_utils import load_model, get_model_folder_path
from data_utils import BoneWrapper, get_df, get_features
import torch.nn as nn
import torch as ch
import numpy as np
import utils
from tqdm import tqdm
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_models(folder_path, n_models=1000):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_accs(val_loader, models):
    accs = []

    criterion = nn.BCEWithLogitsLoss().cuda()
    for model in tqdm(models):
        model = model.cuda()

        _, vacc = utils.validate_epoch(
            val_loader, model, criterion, verbose=False)
        accs.append(vacc)

    return np.array(accs)


def get_predictions(val_loader, models):
    # Step 1: Get predictions for all points on all models
    all_outputs = []
    points = []
    for datum in tqdm(val_loader):
        images, _, _ = datum
        points.append(images)

        outputs = np.array([(model(images)[:, 0] >= 0).cpu().numpy()
                   for model in models])
        all_outputs.append(outputs)

    points = ch.cat(points, 0)
    all_outputs = np.concatenate(all_outputs, 1).T
    print(points.shape, all_outputs.shape)

    return points, all_outputs


def get_acc(preds_1, preds_2, rules, threshold=None):
    feat_vec_1, feat_vec_2 = [], []
    feat_vec_1.append(preds_1[rules])
    feat_vec_1.append(1 - preds_1[~rules])
    feat_vec_1 = np.concatenate(feat_vec_1)
    feat_vec_2.append(preds_2[rules])
    feat_vec_2.append(1 - preds_2[~rules])
    feat_vec_2 = np.concatenate(feat_vec_2)

    # Try thresholds if not specified
    best_acc, best_threshold = 0, 0
    if threshold is None:
        thresholds = np.arange(0.1, 1, 0.01)
        for threshold in thresholds:
            acc_0 = np.mean(feat_vec_1 < threshold)
            acc_1 = np.mean(feat_vec_1 >= threshold)
            acc = (acc_0 + acc_1) / 2
            if acc > best_acc:
                best_acc, best_threshold = acc, threshold
    else:
        best_threshold = 0.5

    acc = (np.mean(feat_vec_1 < best_threshold) + np.mean(feat_vec_1 >= best_threshold)) / 2

    if threshold is None:
        return acc, best_threshold
    return best_acc


def picked_points(val_loader, models_1, models_2, top_ratio=0.25):
    # Step 1: Get predictions for all points on all models
    points, preds_1 = get_predictions(val_loader, models_1)
    _, preds_2 = get_predictions(val_loader, models_2)

    # Step 2: Check which of the prediction rules, (0,1) or (1,0) is more valid for the models
    rule_1 = np.sum(preds_1 == 0, 1) + np.sum(preds_2 == 1, 1)
    rule_2 = np.sum(preds_1 == 1, 1) + np.sum(preds_2 == 0, 1)
    # Compare to get point-wise rules
    wanted_rule = (rule_1 >= rule_2)
    scores = np.zeros_like(rule_1)
    scores[wanted_rule] = rule_1[wanted_rule]
    scores[~wanted_rule] = rule_2[~wanted_rule]

    # Only retain top_ratio of the points, based on observed performance
    top_indices = np.argsort(scores)[-int(top_ratio*len(scores)):]
    points = points[top_indices]
    wanted_rule = wanted_rule[top_indices]

    # Create pseudo iterator of points for compatibility with get_predictions
    points_iterator = [(x.unsqueeze(0), None, None) for x in points]

    # Step 3: Return points with rules
    return points_iterator, wanted_rule


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256*32)
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2")
    parser.add_argument('--top_ratio', type=float,
                        help="ratio of top points to pick", default=0.2)
    parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()
    utils.flash_utils(args)

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    df_1 = utils.heuristic(
        df_val, filter, float(args.ratio_1),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    df_2 = utils.heuristic(
        df_val, filter, float(args.ratio_2),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    # Prepare data loaders
    ds_1 = BoneWrapper(
        df_1, df_1, features=features)
    ds_2 = BoneWrapper(
        df_2, df_2, features=features)
    loaders = [
        ds_1.get_loaders(args.batch_size, shuffle=False)[1],
        ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]

    # Load victim models
    models_victim_1 = get_models(get_model_folder_path("victim", args.ratio_1))
    models_victim_2 = get_models(get_model_folder_path("victim", args.ratio_2))

    # Load adv models
    total_models = 100
    models_1 = get_models(get_model_folder_path(
        "adv", args.ratio_1), total_models // 2)
    models_2 = get_models(get_model_folder_path(
        "adv", args.ratio_2), total_models // 2)

    allaccs_1, allaccs_2 = [], []
    vic_accs, adv_accs = [], []
    for loader in loaders:

        # Get picked points and corresponding rules
        points_iter, rules = picked_points(
            loader, models_1, models_2, args.top_ratio)

        # Get accuracy on adversary's models
        preds_1 = get_predictions(points_iter, models_1)[1]
        preds_2 = get_predictions(points_iter, models_2)[1]
        tracc = get_acc(preds_1, preds_2, rules)

        print("[Adversary] Accuracy: %.2f" % (100 * tracc))
        adv_accs.append(tracc)

        # Compute accuracies on this data for victim
        preds_victim_1 = get_predictions(points_iter, models_victim_1)[1]
        preds_victim_2 = get_predictions(points_iter, models_victim_2)[1]

        specific_acc = get_acc(preds_victim_1, preds_victim_2, rules)

        # Test based on adv models
        print("[Victim] Accuracy: %.2f" % (100 * specific_acc))
        vic_accs.append(specific_acc)

    adv_accs = np.array(adv_accs)
    vic_accs = np.array(vic_accs)

    # Look at model performance on test sets from both G_b
    # and pick the better one
    print("New-Test accuracy: %.3f" %
          (100 * vic_accs[np.argmax(adv_accs)]))
