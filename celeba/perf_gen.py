from model_utils import get_model, BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES
import torch.nn as nn
import numpy as np
from utils import get_threshold_pred, find_threshold_pred,flash_utils
from tqdm import tqdm
import torch as ch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from optimal_generation import specific_case,get_all_models
mpl.rcParams['figure.dpi'] = 200

def get_preds(l,m):
    ps=[]
    for model in tqdm(m):
        
        ch.cuda.empty_cache()
        with ch.no_grad():
            ps.append(model(l.cuda()).detach()[:, 0])       
        
    ps = ch.stack(ps,0).to(ch.device('cpu')).numpy()
    ps = np.transpose(ps)
    return ps



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
    parser.add_argument('--use_natural', action="store_true",
                        help="Pick from actual images")
    parser.add_argument('--use_dt', action="store_true",
                        help="Train small decision tree based on activation values")
    parser.add_argument('--dt_layers', default="1,2",
                        help="layers to use features for (if and when training DT)")
    parser.add_argument('--constrained', action="store_true",
                        help="Constrain amount of noise added")
    parser.add_argument('--start_natural', action="store_true",
                        help="Start with natural images, but better criteria")
    parser.add_argument('--align', action="store_true",
                        help="Look at relative change in activation trends")
    parser.add_argument('--gpu', type=int,
                        default=0, help="device number")
    args = parser.parse_args()
    flash_utils(args)
    #ch.cuda.set_device(args.gpu)
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
    Y_train = [0.] * len(X_train_1) + [1.] * len(X_train_2)
    Y_train = ch.from_numpy(np.array(Y_train)).cuda()
    x_use_1, normal_data, threshold, train_acc_1, clf_1 = specific_case(
        X_train_1, X_train_2, Y_train, float(args.ratio_1), args)
    x_use_2, normal_data, threshold, train_acc_2, clf_2 = specific_case(
        X_train_1, X_train_2, Y_train, float(args.ratio_2), args)

    x_use_1=(x_use_1 + 1) / 2
    x_use_2=(x_use_2 + 1) / 2
    loaders = (x_use_1,x_use_2)
    p1 = [get_preds(loaders[0],X_train_1), get_preds(loaders[1],X_train_1)]
    p2 = [get_preds(loaders[0],X_train_2), get_preds(loaders[1],X_train_2)]
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
    pv1 = [get_preds(loaders[0],X_test_1), get_preds(loaders[1],X_test_1)]
    
    X_test_2 = get_all_models(
        test_dir_2, n_test_models, latent_focus, fake_relu)
    Y_test = [0.] * len(X_test_1) + [1.] * len(X_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()
    
    
    vic_accs, adv_accs = [], []
    total_models = args.n_models*2
    del X_test_1
    ch.cuda.empty_cache()
    pv2 = [get_preds(loaders[0],X_test_2), get_preds(loaders[1],X_test_2)]
    del X_test_2
    ch.cuda.empty_cache()
    for j in range(2):
        adv_accs,threshold, rule = find_threshold_pred(
                # accs_1, accs_2, granularity=0.01)
            p1[j], p2[j], granularity=0.005)
        combined = np.concatenate((pv1[j], pv2[j]),axis=1)
        classes = np.concatenate((
        np.zeros(pv1[j].shape[1]), np.ones(pv2[j].shape[1])))
        specific_acc = get_threshold_pred(combined, classes, threshold, rule)
        
        vic_accs.append(specific_acc)

        # Collect all accuracies for basic baseline
        

    adv_accs = np.array(adv_accs)
    vic_accs = np.array(vic_accs)

    # Basic baseline: look at model performance on test sets from both G_b
    # Predict b for whichever b it is higher
    content = "Perpoint accuracy: {}".format(100 * vic_accs[np.argmax(adv_accs)])
    print(content)
    log_path = os.path.join('./log','gen',"perpoint_{}:{}".format(args.filter,args.ratio_1))
    if not os.path.isdir(log_path):
         os.makedirs(log_path)
    with open(os.path.join(log_path,args.ratio_2),"w") as wr:
        wr.write(content)
