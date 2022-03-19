"""
    Large collection of utility functions and classes that are
    shared across all datasets. In the long run, we would have a common outer
    structure for all datasets, with dataset-specific configuration files.
"""
import torch as ch
import numpy as np
from os import environ
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR
from robustness.tools import folder
from robustness.tools.misc import log_statement

from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List
import os


def scaled_values(val, mean, std, eps=1e-10):
    return (val - np.repeat(np.expand_dims(mean, 1), val.shape[1], axis=1)) / (np.expand_dims(std, 1) + eps)


def load_all_loader_data(data_loader):
    images, labels = [], []
    for (image, label) in data_loader:
        images.append(image)
        labels.append(label)
    images = ch.cat(images)
    labels = ch.cat(labels)
    return (images, labels)


def load_all_data(ds):
    batch_size = 512
    _, test_loader = ds.make_loaders(
        batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
    return load_all_loader_data(test_loader)


def get_sensitivities(path, numpy=False):
    log_statement("==> Loading Delta Values")
    # Directly load, if numpy array
    if numpy:
        return np.load(path)
    # Process, if text file
    features = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            values = np.array([float(x) for x in line.rstrip('\n').split(',')])
            features.append(values)
    return np.array(features)


def best_target_image(mat, which=0):
    sum_m = []
    for i in range(mat.shape[1]):
        mat_interest = mat[mat[:, i] != np.inf, i]
        sum_m.append(np.average(np.abs(mat_interest)))
    best = np.argsort(sum_m)
    return best[which]


def get_statistics(diff):
    l1_norms = ch.sum(ch.abs(diff), dim=1)
    l2_norms = ch.norm(diff, dim=1)
    linf_norms = ch.max(ch.abs(diff), dim=1)[0]
    return (l1_norms, l2_norms, linf_norms)


def get_stats(base_path):
    mean = np.load(os.path.join(base_path, "feature_mean.npy"))
    std = np.load(os.path.join(base_path, "feature_std.npy"))
    return mean, std


def get_logits_layer_name(arch):
    if "vgg" in arch:
        return "module.model.classifier.weight"
    elif "resnet" in arch:
        return "module.model.fc.weight"
    elif "densenet" in arch:
        return "module.model.linear.weight"
    return None


class SpecificLayerModel(ch.nn.Module):
    def __init__(self, model, layer_index):
        super(SpecificLayerModel, self).__init__()
        self.model = model
        self.layer_index = layer_index

    def forward(self, x):
        logits, _ = self.model(x, this_layer_input=self.layer_index)
        return logits


class MadryToNormal:
    def __init__(self, model, fake_relu=False):
        self.model = model
        self.fake_relu = fake_relu
        self.model.eval()

    def __call__(self, x):
        logits, _ = self.model(x, fake_relu=self.fake_relu)
        return logits

    def eval(self):
        return self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()


def classwise_pixelwise_stats(loader, num_classes=10, classwise=False):
    images, labels = load_all_loader_data(loader)
    if not classwise:
        return ch.mean(images, 0), ch.std(images, 0)
    means, stds = [], []
    for i in range(num_classes):
        specific_images = images[labels == i]
        means.append(ch.mean(specific_images, 0))
        stds.append(ch.std(specific_images, 0))
    return means, stds


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Input size: [batch, n_features]
        # Output size: [batch, 3, 32, 32]
        # Expects 48, 4, 4
        self.dnn = nn.Sequential(
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU())
        self.decoder = nn.Sequential(
            # [batch, 24, 8, 8]
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # [batch, 12, 16, 16]
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # [batch, 3, 32, 32]
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ = self.dnn(x)
        x_ = x_.view(x_.shape[0], 48, 4, 4)
        return self.decoder(x_)


class BasicDataset(ch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def compute_delta_values(logits, weights, actual_label=None):
    # Iterate through all possible classes, calculate flip probabilities
    actual_label = ch.argmax(logits)
    numerator = (logits[actual_label] - logits).unsqueeze(1)
    denominator = weights - weights[actual_label]
    numerator = numerator.repeat(1, denominator.shape[1])
    delta_values = ch.div(numerator, denominator)
    delta_values[actual_label] = np.inf
    return delta_values


def get_these_params(model, identifier):
    for name, param in model.state_dict().items():
        if name == identifier:
            return param
    return None


class MNISTFlatModel(nn.Module):
    def __init__(self):
        super(MNISTFlatModel, self).__init__()
        n_feat = 28 * 28
        self.dnn = nn.Sequential(
            nn.Linear(n_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10))

    def forward(self, x):
        x_ = x.view(x.shape[0], -1)
        return self.forward(x_)


def get_cropped_faces(cropmodel, x):
    def renormalize(z): return (z * 0.5) + 0.5

    images = [Image.fromarray(
        (255 * np.transpose(renormalize(x_.numpy()), (1, 2, 0))).astype('uint8')) for x_ in x]
    crops = cropmodel(images)

    x_cropped = []
    indices = []
    for j, cr in enumerate(crops):
        if cr is not None:
            x_cropped.append(cr)
            indices.append(j)

    return ch.stack(x_cropped, 0), indices


class CustomBertModel(nn.Module):
    def __init__(self, base_model):
        super(CustomBertModel, self).__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        outputs = self.bert(**x)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def get_outputs(model, X, no_grad=False):

    with ch.set_grad_enabled(not no_grad):
        outputs = model(X)

    return outputs[:, 0]


def get_threshold_pred(X, Y, threshold, rule,
                       get_pred: bool = False,
                       confidence: bool = False):
    if X.shape[1] != Y.shape[0]:
        raise ValueError('Dimension mismatch between X and Y: %d and %d should match' % (X.shape[1], Y.shape[0]))
    if X.shape[0] != threshold.shape[0]:
        raise ValueError('Dimension mismatch between X and threshold: %d and %d should match' % (X.shape[0], threshold.shape[0]))
    res = []
    for i in range(X.shape[1]):
        prob = np.average((X[:, i] <= threshold) == rule)
        if confidence:
            res.append(prob)
        else:
            res.append(prob >= 0.5)
    res = np.array(res)
    if confidence:
        acc = np.mean((res >= 0.5) == Y)
    else:    
        acc = np.mean(res == Y)
    if get_pred:
        return res, acc
    return acc


def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _perpoint_threshold_on_ratio(preds_1, preds_2, classes, threshold, rule):
    """
        Run perpoint threshold test (confidence)
        for a given "quartile" ratio
    """
    # Combine predictions into one vector
    combined = np.concatenate((preds_1, preds_2), axis=1)

    # Compute accuracy for given predictions, thresholds, and rules
    preds, acc = get_threshold_pred(
        combined, classes, threshold, rule, get_pred=True,
        confidence=True)

    return 100 * acc, preds


def perpoint_threshold_test_per_dist(preds_adv: List, preds_victim: List,
                                     ratios: List = [1.],
                                     granularity: float = 0.005):
    """
        Compute thresholds (based on probabilities) for each given datapoint,
        search for thresholds using given adv model's predictions.
        Compute accuracy and predictions using given data and predictions
        on victim model's predictions.
        Try this out with different values of "quartiles", where points
        are ranked according to some utility estimate.
    """
    # Predictions by adversary's models
    p1, p2 = preds_adv
    # Predictions by victim's models
    pv1, pv2 = preds_victim

    # Optimal order of point
    order = order_points(p1, p2)

    # Order points according to computed utility
    p1 = np.transpose(p1)[order][::-1]
    p2 = np.transpose(p2)[order][::-1]
    pv1 = np.transpose(pv1)[order][::-1]
    pv2 = np.transpose(pv2)[order][::-1]

    # Get thresholds for all points
    _, thres, rs = find_threshold_pred(p1, p2, granularity=granularity)

    # Ground truth
    classes_adv = np.concatenate(
        (np.zeros(p1.shape[1]), np.ones(p2.shape[1])))
    classes_victim = np.concatenate(
        (np.zeros(pv1.shape[1]), np.ones(pv2.shape[1])))

    adv_accs, victim_accs, victim_preds, adv_preds = [], [], [], []
    for ratio in ratios:
        # Get first <ratio> percentile of points
        leng = int(ratio * p1.shape[0])
        p1_use, p2_use = p1[:leng], p2[:leng]
        pv1_use, pv2_use = pv1[:leng], pv2[:leng]
        thres_use, rs_use = thres[:leng], rs[:leng]

        # Compute accuracy for given data size on adversary's models
        adv_acc, adv_pred = _perpoint_threshold_on_ratio(
            p1_use, p2_use, classes_adv, thres_use, rs_use)
        adv_accs.append(adv_acc)
        # Compute accuracy for given data size on victim's models
        victim_acc, victim_pred = _perpoint_threshold_on_ratio(
            pv1_use, pv2_use, classes_victim, thres_use, rs_use)
        victim_accs.append(victim_acc)
        # Keep track of predictions on victim's models
        victim_preds.append(victim_pred)
        adv_preds.append(adv_pred)

    adv_accs = np.array(adv_accs)
    victim_accs = np.array(victim_accs)
    victim_preds = np.array(victim_preds, dtype=object)
    adv_preds = np.array(adv_preds, dtype=object)
    return adv_accs, adv_preds, victim_accs, victim_preds


def perpoint_threshold_test(preds_adv: List, preds_victim: List,
                            ratios: List = [1.],
                            granularity: float = 0.005):
    """
        Take predictions from both distributions and run attacks.
        Pick the one that works best on adversary's models
    """
    # Get data for first distribution
    adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1 = perpoint_threshold_test_per_dist(
        preds_adv[0], preds_victim[0], ratios, granularity)
    # Get data for second distribution
    adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2 = perpoint_threshold_test_per_dist(
        preds_adv[1], preds_victim[1], ratios, granularity)

    # Get best adv accuracies for both distributions and compare
    which_dist = 0
    if np.max(adv_accs_1) > np.max(adv_accs_2):
        adv_accs_use, adv_preds_use, victim_accs_use, victim_preds_use = adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1
    else:
        adv_accs_use, adv_preds_use, victim_accs_use, victim_preds_use = adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2
        which_dist = 1

    # Out of the best distribution, pick best ratio according to accuracy on adversary's models
    ind = np.argmax(adv_accs_use)
    victim_acc_use = victim_accs_use[ind]
    victim_pred_use = victim_preds_use[ind]
    adv_acc_use = adv_accs_use[ind]
    adv_pred_use = adv_preds_use[ind]

    return (victim_acc_use, victim_pred_use), (adv_acc_use, adv_pred_use), (which_dist, ind)


def get_ratio_info_for_reg_meta(metamodel, X, Y, num_per_dist, batch_size, combined: bool = True):
    """
        Get MSE and actual predictions for each
        ratio given in Y, using a trained metamodel.
        Returns MSE per ratio, actual predictions per ratio, and
        predictions for each ratio a v/s be using regression
        meta-classifier for binary classification.
    """
    # Evaluate
    metamodel = metamodel.cuda()
    loss_fn = ch.nn.MSELoss(reduction='none')
    _, losses, preds = test_meta(
        metamodel, loss_fn, X, Y.cuda(),
        batch_size, None,
        binary=True, regression=True, gpu=True,
        combined=combined, element_wise=True,
        get_preds=True)
    y_np = Y.numpy()
    losses = losses.numpy()
    # Get all unique ratios (sorted) in GT, and their average losses from model
    ratios = np.unique(y_np)
    losses_dict = {}
    ratio_wise_preds = {}
    for ratio in ratios:
        losses_dict[ratio] = np.mean(losses[y_np == ratio])
        ratio_wise_preds[ratio] = preds[y_np == ratio]
    # Conctruct a matrix where every (i, j) entry is the accuracy
    # for ratio[i] v/s ratio [j], where whichever ratio is closer to the
    # ratios is considered the "correct" one
    # Assume equal number of models per ratio, stored in order of
    # ratios
    acc_mat = np.zeros((len(ratios), len(ratios)))
    for i in range(acc_mat.shape[0]):
        for j in range(i + 1, acc_mat.shape[0]):
            # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
            gt_z = (y_np[num_per_dist * i:num_per_dist * (i + 1)]
                    == float(ratios[j]))
            gt_o = (y_np[num_per_dist * j:num_per_dist * (j + 1)]
                    == float(ratios[j]))
            # Get relevant preds
            pred_z = preds[num_per_dist * i:num_per_dist * (i + 1)]
            pred_o = preds[num_per_dist * j:num_per_dist * (j + 1)]
            pred_z = (pred_z >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            pred_o = (pred_o >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            # Compute accuracies and store
            acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate(
                (pred_z, pred_o), 0)
            acc_mat[i, j] = np.mean(acc)

    return losses_dict, acc_mat, ratio_wise_preds
