from kornia.geometry.transform import resize
import torch as ch
from tkinter import Y
from model_utils import get_models_path,  BASE_MODELS_DIR
from data_utils import CensusTwo, CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc,get_threshold_pred, find_threshold_pred, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from perf_quart import get_models
from perf_all import get_preds,order_points,cal_acc
mpl.rcParams['figure.dpi'] = 200

def gen_optimal(models, labels, sample_shape, n_samples,
                n_steps, step_size,  upscale,
                use_normal=None, constrained=False,):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    if upscale > 1:
        actual_shape = (sample_shape[0], sample_shape[1] //
                        upscale, sample_shape[2] // upscale)
        x_rand_data = ch.rand(*((n_samples,) + actual_shape)).cuda()
        x_eff = resize(x_rand_data, (sample_shape[1], sample_shape[2]))
        print(models[0](x_eff.numpy()).shape[1:])
    else:
        if use_normal is None:
            x_rand_data = ch.rand(*((n_samples,) + sample_shape)).cuda()
        else:
            x_rand_data = use_normal.clone().cuda()
        #print(models[0].predict_proba(x_rand_data.numpy).shape)

    x_rand_data_start = x_rand_data.clone().detach()

    iterator = tqdm(range(n_steps))
    # Focus on latent=4 for now
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)
            
        if upscale > 1:
            x_use = resize(x_rand, (sample_shape[1], sample_shape[2]))
        else:
            x_use = x_rand.clone().detach()

        # Get representations from all models
        reprs = ch.stack([ch.from_numpy(m.predict_proba(x_use.cpu().numpy())) for m in models],0)
        reprs_z = ch.mean(reprs[labels == 0],2)
        reprs_o = ch.mean(reprs[labels == 1],2)
        # const = 2.
        const = 1.
        const_neg = 0.5
        loss = ch.mean((const - reprs_z) ** 2) + \
            ch.mean((const_neg + reprs_o) ** 2)
        # loss = ch.mean((const_neg + reprs_z) ** 2) + ch.mean((const - reprs_o) ** 2)
        grad = ch.autograd.grad(loss, [x_rand])

        with ch.no_grad():
            zero_acts = ch.sum(1. * (reprs[labels == 0] > 0),2)
            one_acts = ch.sum(1. * (reprs[labels == 1] > 0),2)
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
            x_rand_data = ch.clamp(x_rand_data, -1, 1)

    return x_rand.clone().detach(), (l1 + l2).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--step_size', type=float, default=1e2)
    parser.add_argument('--upscale', type=int,
                        default=1, help="optimize and upscale")
    parser.add_argument('--constrained', action="store_true",
                        help="Constrain amount of noise added")
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()
    flash_utils(args)
    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))
    basics, thresholds, tq= [], [],[]
    for _ in tqdm(range(args.tries)):
        
        # Load adv models
        total_models = 100
        models_1 = get_models(get_models_path(
            args.filter, "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(get_models_path(
            args.filter, "adv", args.ratio_2), total_models // 2)
        Y_train = [0.] * len(models_1) + [1.] * len(models_2)
        Y_train = ch.from_numpy(np.array(Y_train)).cuda()
        if args.filter == "two_attr":
            ds_1 = CensusTwo()
            ds_2 = CensusTwo()
            [r11,r12] = args.ratio_1.split(',')
            r11,r12 = float(r11),float(r12)
            [r21,r22] = args.ratio_2.split(',')
            r21,r22 = float(r21),float(r22)
            _, (x_te_1, y_te_1), _ = ds_1.get_data('adv',r11,r12)
            _, (x_te_2, y_te_2), _ = ds_2.get_data('adv',r21,r22)
        else:
            # Prepare data wrappers
            ds_1 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_1), split="adv")
            ds_2 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_2), split="adv")

        # Fetch test data from both ratios
            _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
            _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
        #y_te_1 = y_te_1.ravel()
        #y_te_2 = y_te_2.ravel()
        x_opt1, x_opt2 = [], []
        
        for i in range(args.n_samples):
            # Get optimal point based on local set
            x_opt1_, loss_ = gen_optimal(
                models_1 + models_2, Y_train,
                (42), 1,
                args.steps, args.step_size,
                args.upscale,
                use_normal=ch.from_numpy(x_te_1[i:i + 1]),
                constrained=args.constrained)
            x_opt1.append(x_opt1_.numpy())
        for i in range(args.n_samples):
            # Get optimal point based on local set
            x_opt2_, loss_ = gen_optimal(
                models_1 + models_2, Y_train,
                (42), 1,
                args.steps, args.step_size,
                args.upscale,
                use_normal=ch.from_numpy(x_te_2[i:i + 1]),
                constrained=args.constrained)
            x_opt2.append(x_opt2_.numpy())

        yg = [y_te_1,y_te_2]
        p1 = [get_preds(x_opt1,models_1), get_preds(x_opt2,models_1)]
        p2 = [get_preds(x_opt1,models_2), get_preds(x_opt2,models_2)]
        pv1 = [get_preds(x_opt1,models_victim_1), get_preds(x_opt2,models_victim_1)]
        pv2 = [get_preds(x_opt1,models_victim_2), get_preds(x_opt2,models_victim_2)]
        for i in range(2):
            p1[i] = np.transpose(p1[i])
            p2[i] = np.transpose(p2[i])
            pv1[i] = np.transpose(pv1[i])
            pv2[i] = np.transpose(pv2[i])
            yg[i] = yg[i]
        thres, rs = [],[]
        for j in range(2):
            _,threshold, rule = find_threshold_pred(
                # accs_1, accs_2, granularity=0.01)
            p1[j], p2[j], granularity=0.005)
            
            thres.append(threshold)
            rs.append(rule)
        f_accs = []
        f_accsq = []    
        allaccs_1, allaccs_2 = [], []
        adv_accs = []
        adv_accsq = []
            
        
        
        for j in range(2):
            #get accuracies
                
            
                
            cm = np.concatenate((p1[j], p2[j]),axis=1)
            cl = np.concatenate((
            np.zeros(p1[j].shape[1]), np.ones(p2[j].shape[1])))
            adv_accs.append(100 * get_threshold_pred(cm, cl, thres[j], rs[j]))                
            accs_1 = cal_acc(p1[j],yg[j])
            accs_2 = cal_acc(p2[j],yg[j])

            # Look at [0, 100]
            accs_1 *= 100
            accs_2 *= 100
            tracc, threshold, rule = find_threshold_acc(
            # accs_1, accs_2, granularity=0.01)
            accs_1, accs_2,granularity=0.005)
            adv_accsq.append(100 * tracc)
            combined = np.concatenate((pv1[j], pv2[j]),axis=1)
            classes = np.concatenate((np.zeros(pv1[j].shape[1]), np.ones(pv2[j].shape[1])))
            specific_acc = get_threshold_pred(
            combined, classes, thres[j], rs[j])
            
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
            f_accs.append(100 * specific_acc)
            accs_victim_1 = cal_acc(pv1[j],yg[j])
            accs_victim_2 = cal_acc(pv2[j],yg[j])

            # Look at [0, 100]
            accs_victim_1 *= 100
            accs_victim_2 *= 100
            allaccs_1.append(accs_victim_1)
            allaccs_2.append(accs_victim_2)
            combined = np.concatenate((accs_victim_1, accs_victim_2))
            classes = np.concatenate(
            (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
            specific_acc = get_threshold_acc(
            combined, classes, threshold, rule)
                
               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
            f_accsq.append(100 * specific_acc)
        adv_accs = np.array(adv_accs)
        adv_accsq = np.array(adv_accsq)
        allaccs_1 = np.array(allaccs_1)
        allaccs_2 = np.array(allaccs_2)

        preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
        preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])

        basic_baseline_acc = (np.mean(preds_1) + np.mean(preds_2)) / 2
        basics.append((100 * basic_baseline_acc))
        thresholds.append(f_accs[np.argmax(adv_accs)])
        tq.append(f_accsq[np.argmax(adv_accsq)])
    content = 'Perpoint thresholds accuracy: {}'.format(thresholds)
    print(content)
    log_path = os.path.join('./log','gen',"perf_perpoint_{}:{}".format(args.filter,args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    log_path = os.path.join('./log','gen',"selective_loss_{}:{}".format(args.filter,args.ratio_1))
    cl = 'Baseline accuracy: {}'.format(np.mean(basics))
    print(cl)
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(cl)
    content = 'Thresholds accuracy: {}'.format(tq)
    print(content)
    log_path = os.path.join('./log','gen',"perf_quart_{}:{}".format(args.filter,args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
    
