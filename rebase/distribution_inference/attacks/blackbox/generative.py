import numpy as np
import torch as ch
from tqdm import tqdm
import random
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions,PredictionsOnOneDistribution
from distribution_inference.attacks.blackbox.utils import get_preds
def gen_optimal(m1,m2, sample_shape, n_samples,
                n_steps, step_size,loader,latent_focus,preload:bool=False,
                use_normal=None, constrained=False,model_ratio=1.0):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    
    
    if use_normal is None:
        x_rand_data = ch.rand((n_samples,sample_shape[0])).cuda()
    else:
        x_rand_data = use_normal.clone().cuda()

    x_rand_data_start = x_rand_data.clone().detach()
    
    iterator = tqdm(range(n_steps))
    
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)

        x_use = x_rand
        l1 = list(range(len(m1)))
        
        l2 = list(range(len(m2)))
        if model_ratio != 1.0:
            random.shuffle(l1)
            random.shuffle(l2)
        # Get representations from all models
        reprs1 = get_preds(loader,np.array(m1)[l1])
        reprs2 = ch.stack([m2[j](x_use, latent=latent_focus) for j in l2[0:int(model_ratio*len(l2))]], 0)
        reprs_z = ch.mean(reprs1, 2)
        reprs_o = ch.mean(reprs2, 2)
        # If latent_focus is None, simply maximize difference in prediction probs
        if latent_focus is None:
            reprs_o = ch.sigmoid(reprs_o)
            reprs_z = ch.sigmoid(reprs_z)
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
            #x_rand_data = ch.clamp(x_rand_data, -1, 1)

    if latent_focus is None:
        return x_rand.clone().detach(), loss.item()
    return x_rand.clone().detach(), (l1 + l2).item()

def generate_data(X_train_1, X_train_2, ratio, args,shuffle=True):
    df_ = heuristic(
        df_val, filter, float(ratio),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)
    ds =  BoneWrapper(
        df_, df_, features=features)
    if args.use_normal:
        _, test_loader = ds.get_loaders(args.n_samples, shuffle=shuffle)
        normal_data = next(iter(test_loader))[0]
    else:
        _, test_loader = ds.get_loaders(100,shuffle=True)
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
            print('Gradient ascend')
            # Get optimal point based on local set
            if args.r2 == 1.0:
                x_opt_, loss_ = gen_optimal(
                X_train_1,X_train_2,
                [1024], 1,
                args.steps, args.step_size,
                args.latent_focus,
                use_normal=normal_data[i:i +
                                       1].cuda() if (args.use_normal or args.start_natural) else None,
                constrained=args.constrained,
                model_ratio=args.r)
            else:
                random.shuffle(X_train_1)
                random.shuffle(X_train_2)
                x_opt_, loss_ = gen_optimal(
                X_train_1[0:int(args.r2*len(X_train_1))],X_train_2[0:int(args.r2*len(X_train_2))],
                [1024], 1,
                args.steps, args.step_size,
                args.latent_focus,
                use_normal=normal_data[i:i +
                                       1] if (args.use_normal or args.start_natural) else None,
                constrained=args.constrained,
                model_ratio=args.r)
            x_opt.append(x_opt_)
            losses.append(loss_)

        if args.use_best:
            best_id = np.argmin(losses)
            x_opt = x_opt[best_id:best_id+1]

        x_opt = ch.cat(x_opt, 0)
        #x_opt = normal_data

        
    x_use = x_opt
    x_use = x_use.cuda()
    return x_use.cpu()
