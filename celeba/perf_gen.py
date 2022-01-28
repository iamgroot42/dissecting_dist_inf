from model_utils import BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES
import numpy as np
from utils import get_threshold_pred, find_threshold_pred, flash_utils
from tqdm import tqdm
import torch as ch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from optimal_generation import get_all_models, ordered_samples
from kornia.geometry.transform import resize
import random
mpl.rcParams['figure.dpi'] = 200


def get_preds(l, m):
    ps = []
    for model in tqdm(m):

        ch.cuda.empty_cache()
        with ch.no_grad():
            ps.append(model(l.cuda()).detach()[:, 0])

    ps = ch.stack(ps, 0).to(ch.device('cpu')).numpy()
    ps = np.transpose(ps)
    return ps


def gen_optimal(m1, m2, sample_shape, n_samples,
                n_steps=100, step_size=1e2, latent_focus=None, upscale=1,
                use_normal=None, constrained=False, model_ratio=1.0,
                clamp=False):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    if upscale > 1:
        actual_shape = (sample_shape[0], sample_shape[1] //
                        upscale, sample_shape[2] // upscale)
        x_rand_data = ch.rand(*((n_samples,) + actual_shape)).cuda()
        x_eff = resize(x_rand_data, (sample_shape[1], sample_shape[2]))
    else:
        if use_normal is None:
            x_rand_data = ch.rand(*((n_samples,) + sample_shape)).cuda()
        else:
            x_rand_data = use_normal.clone().cuda()

    x_rand_data_start = x_rand_data.clone().detach()

    iterator = tqdm(range(n_steps))
    # Focus on latent=4 for now
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)

        if upscale > 1:
            x_use = resize(x_rand, (sample_shape[1], sample_shape[2]))
        else:
            x_use = x_rand
        l1 = list(range(len(m1)))

        l2 = list(range(len(m2)))
        if model_ratio != 1.0:
            random.shuffle(l1)
            random.shuffle(l2)
        # Get representations from all models
        reprs1 = ch.stack([m1[j](x_use, latent=latent_focus)
                           for j in l1[0:int(model_ratio*len(l1))]], 0)
        reprs2 = ch.stack([m2[j](x_use, latent=latent_focus)
                           for j in l2[0:int(model_ratio*len(l2))]], 0)
        reprs_z = ch.mean(reprs1, 2)
        reprs_o = ch.mean(reprs2, 2)
        # If latent_focus is None, simply maximize difference in prediction probs
        if latent_focus is None:
            reprs_o = ch.sigmoid(reprs_o)
            reprs_z = ch.sigmoid(reprs_z)
            # +0.1*ch.std(reprs_o)+0.1*ch.std(reprs_z)
            loss = 1 - ch.mean((reprs_o - reprs_z) ** 2)
        else:
            # const = 2.
            const = 1.
            const_neg = 0.5
            loss = ch.mean((const - reprs_z) ** 2) + \
                ch.mean((const_neg + reprs_o) ** 2)
            # loss = ch.mean((const_neg + reprs_z) ** 2) + ch.mean((const - reprs_o) ** 2)

        # Compute gradient
        grad = ch.autograd.grad(loss, [x_rand])

        with ch.no_grad():
            if latent_focus is None:

                preds_z = reprs1 > 0.5
                preds_o = reprs2 > 0.5
                # Count mismatch in predictions
                n_mismatch = ch.mean(1.0*preds_z)-ch.mean(1.0*preds_o)
                iterator.set_description(
                    "Loss: %.4f | Mean dif in pred: %.4f" % (loss.item(), n_mismatch))
            else:
                zero_acts = ch.sum(1. * (reprs1 > 0), 2)
                one_acts = ch.sum(1. * (reprs2 > 0), 2)
                l1 = ch.mean((const - reprs_z) ** 2)
                l2 = ch.mean((const_neg + reprs_o) ** 2)
                # l1 = ch.mean((const_neg + reprs_z) ** 2)
                # l2 = ch.mean((const - reprs_o) ** 2)
                iterator.set_description("Loss: %.3f | ZA: %.1f | OA: %.1f | Loss(1): %.3f | Loss(2): %.3f" % (
                    loss.item(), zero_acts.mean(), one_acts.mean(), l1, l2))

        with ch.no_grad():
            x_intermediate = x_rand_data - step_size * grad[0]
            if constrained:
                shape = x_rand_data.shape
                difference = (x_rand_data_start - x_intermediate)
                difference = difference.view(difference.shape[0], -1)
                eps = 0.5
                difference_norm = eps * \
                    ch.norm(difference, p=2, dim=0, keepdim=True)
                difference_norm = difference_norm.view(*shape)
                # difference = difference.renorm(p=2, dim=0, maxnorm=eps)
                x_rand_data = x_rand_data_start - difference_norm
            else:
                x_rand_data = x_intermediate
            if clamp:
                x_rand_data = ch.clamp(x_rand_data, -1, 1)

    if latent_focus is None:
        return x_rand.clone().detach(), loss.item()
    return x_rand.clone().detach(), (l1 + l2).item()


def generate_data(X_train_1, X_train_2, ratio, args, shuffle=True):
    ds = CelebaWrapper(args.filter, ratio, "adv")
    if args.use_normal:
        _, test_loader = ds.get_loaders(args.n_samples, eval_shuffle=shuffle)
        normal_data = next(iter(test_loader))[0]
    else:
        _, test_loader = ds.get_loaders(100, eval_shuffle=True)
        normal_data = next(iter(test_loader))[0].cuda()
    if args.use_natural:
        x_use = ordered_samples(X_train_1, X_train_2, test_loader, args)
    else:
        if args.start_natural:
            normal_data = ordered_samples(
                X_train_1, X_train_2, test_loader, args)
            print("Starting with natural data")

        x_opt, losses = [], []
        for i in range(args.n_samples):
            # Get optimal point based on local set
            if args.r2 == 1.0:
                x_opt_, loss_ = gen_optimal(
                    X_train_1, X_train_2,
                    (3, 218, 178), 1,
                    args.steps, args.step_size,
                    args.latent_focus, args.upscale,
                    use_normal=normal_data[i:i +
                                           1] if (args.use_normal or args.start_natural) else None,
                    constrained=args.constrained,
                    model_ratio=args.r,
                    clamp=args.clamp)
            else:
                random.shuffle(X_train_1)
                random.shuffle(X_train_2)
                x_opt_, loss_ = gen_optimal(
                    X_train_1[0:int(args.r2*len(X_train_1))
                              ], X_train_2[0:int(args.r2*len(X_train_2))],
                    (3, 218, 178), 1,
                    args.steps, args.step_size,
                    args.latent_focus, args.upscale,
                    use_normal=normal_data[i:i +
                                           1] if (args.use_normal or args.start_natural) else None,
                    constrained=args.constrained,
                    model_ratio=args.r,
                    clamp=args.clamp)
            x_opt.append(x_opt_)
            losses.append(loss_)

        if args.use_best:
            best_id = np.argmin(losses)
            x_opt = x_opt[best_id:best_id+1]

        x_opt = ch.cat(x_opt, 0)
        #x_opt = normal_data

        if args.upscale:
            x_use = resize(x_opt, (218, 178))
        else:
            x_use = x_opt

    x_use = x_use.cuda()
    return x_use.cpu()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--task', default="Smiling",
                        choices=SUPPORTED_PROPERTIES,
                        help='task to focus on')
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2")
    parser.add_argument('--testing', action='store_true',
                        help="testing script or not")
    parser.add_argument('--n_models', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--latent_focus', type=int, default=None)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--step_size', type=float, default=1e2)
    parser.add_argument('--use_normal', action="store_true",
                        help="Use normal data for init instead of noise")
    parser.add_argument('--use_best', action="store_true",
                        help="Use lowest-loss example instead of all of them")
    parser.add_argument('--upscale', type=int,
                        default=1, help="optimize and upscale")
    parser.add_argument('--clamp', 
                        action="store_true", help="clamp data when generating")
    parser.add_argument('--use_natural', action="store_true",
                        help="Pick from actual images")
    parser.add_argument('--constrained', action="store_true",
                        help="Constrain amount of noise added")
    parser.add_argument('--start_natural', action="store_true",
                        help="Start with natural images, but better criteria")
    parser.add_argument('--r', type=float,
                        default=1.0, help="step-random, ratio of model to use to generate samples")
    parser.add_argument('--r2', type=float,
                        default=1.0, help="step-fixed,ratio of model to use to generate samples each datum")
    args = parser.parse_args()
    flash_utils(args)


    if args.use_natural:
        latent_focus = None
        fake_relu = False
    else:
        latent_focus = args.latent_focus
        fake_relu = True
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.ratio_1))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.ratio_2))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.ratio_1))
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "victim/%s/%s/" %
                              (args.filter, args.ratio_2))
    print("Loading models")

    X_train_1 = get_all_models(
        train_dir_1, args.n_models, latent_focus, fake_relu,
        shuffle=True)
    X_train_2 = get_all_models(
        train_dir_2, args.n_models, latent_focus, fake_relu,
        shuffle=True)

    x_use_1 = generate_data(
        X_train_1, X_train_2, float(args.ratio_1), args)
    x_use_2 = generate_data(
        X_train_1, X_train_2, float(args.ratio_2), args)

    loaders = (x_use_1, x_use_2)
    p1 = [get_preds(loaders[0], X_train_1), get_preds(loaders[1], X_train_1)]
    p2 = [get_preds(loaders[0], X_train_2), get_preds(loaders[1], X_train_2)]

    # Free up memory
    del X_train_1
    del X_train_2
    ch.cuda.empty_cache()

    # Load test models
    if args.testing:
        n_test_models = 10
    else:
        n_test_models = 1000

    X_test_1 = get_all_models(
        test_dir_1, n_test_models, latent_focus, fake_relu)
    pv1 = [get_preds(loaders[0], X_test_1), get_preds(loaders[1], X_test_1)]

    # Free up memory
    del X_test_1
    ch.cuda.empty_cache()

    X_test_2 = get_all_models(
        test_dir_2, n_test_models, latent_focus, fake_relu)

    vic_accs, adv_accs = [], []
    total_models = args.n_models*2

    pv2 = [get_preds(loaders[0], X_test_2), get_preds(loaders[1], X_test_2)]

    # Free up memory
    del X_test_2
    ch.cuda.empty_cache()

    for j in range(2):
        adv_accs, threshold, rule = find_threshold_pred(
            # accs_1, accs_2, granularity=0.01)
            p1[j], p2[j], granularity=0.005)
        combined = np.concatenate((pv1[j], pv2[j]), axis=1)
        classes = np.concatenate((
            np.zeros(pv1[j].shape[1]), np.ones(pv2[j].shape[1])))
        specific_acc = get_threshold_pred(combined, classes, threshold, rule)

        vic_accs.append(specific_acc)

        # Collect all accuracies for basic baseline

    adv_accs = np.array(adv_accs)
    vic_accs = np.array(vic_accs)

    # Basic baseline: look at model performance on test sets from both G_b
    # Predict b for whichever b it is higher
    content = "Perpoint accuracy: {}".format(
        100 * vic_accs[np.argmax(adv_accs)])
    print(content)
    log_path = os.path.join(
        './log', 'gen', "perpoint_{}:{}".format(args.filter, args.ratio_1))
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    #with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        #wr.write(content)
