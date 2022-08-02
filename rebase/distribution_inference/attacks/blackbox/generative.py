import numpy as np
import torch as ch
from tqdm import tqdm
import gc
from distribution_inference.attacks.blackbox.core import PredictionsOnDistributions,PredictionsOnOneDistribution,Attack
from distribution_inference.config.core import GenerativeAttackConfig
from torch.utils.data import Dataset
from distribution_inference.attacks.blackbox.per_point import PerPointThresholdAttack
def get_differences(models, x_use, latent_focus, reduce=True):
    # View resulting activation distribution for current models
    reprs = ch.stack([m(x_use, latent=latent_focus).detach()
                      for m in models], 0)
    # Count number of neuron activations
    reprs = (1. * ch.sum(reprs > 0, 2))
    if reduce:
        reprs = ch.mean(reprs, 1)
    reprs = reprs.cpu().numpy()
    return reprs


def ordered_samples(models_0, models_1, loader, latent_focus,n_samples):
    diffs_0, diffs_1, inputs = [], [], []
    for tup in loader:
        x = tup[0]
        inputs.append(x)
        x = x.cuda()
        reprs_0 = get_differences(models_0, x, latent_focus, reduce=False)
        reprs_1 = get_differences(models_1, x, latent_focus, reduce=False)
        diffs_0.append(reprs_0)
        diffs_1.append(reprs_1)

    diffs_0 = np.concatenate(diffs_0, 1).T
    diffs_1 = np.concatenate(diffs_1, 1).T
    # diffs = (np.mean(diffs_1, 1) - np.mean(diffs_0, 1))
    diffs = (np.min(diffs_1, 1) - np.max(diffs_0, 1))
    # diffs = (np.min(diffs_0, 1) - np.max(diffs_1, 1))
    inputs = ch.cat(inputs)
    # Pick examples with maximum difference
    diff_ids = np.argsort(-np.abs(diffs))[:n_samples]
    print("Best samples had differences", diffs[diff_ids])
    return inputs[diff_ids].cuda()



def gen_optimal(m1,m2, sample_shape, n_samples,
                n_steps, step_size,latent_focus,
                use_normal=None, constrained=False,model_ratio=1.0,clamp=False):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    
    
    if use_normal is None:
        x_rand_data = ch.rand(*((n_samples,) + sample_shape)).cuda()
    else:
        x_rand_data = use_normal.clone().cuda()

    x_rand_data_start = x_rand_data.clone().detach()
    
    iterator = tqdm(range(n_steps))
    
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)

        x_use = x_rand
        # Get representations from all models
        reprs1 = ch.stack([m(x_use, latent=latent_focus)
                           for m in np.random.permutation(m1)[:int(model_ratio*len(m1))]], 0)
        reprs2 = ch.stack([m(x_use, latent=latent_focus)
                           for m in np.random.permutation(m2)[:int(model_ratio*len(m2))]], 0)
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
            if clamp:
                x_rand_data = ch.clamp(x_rand_data, -1, 1)

            #x_rand_data = ch.clamp(x_rand_data, -1, 1)

    if latent_focus is None:
        return x_rand.clone().detach(), loss.item()
    return x_rand.clone().detach(), (l1 + l2).item()

def generate_data(X_train_1, X_train_2, ds, batch_size:int, config:GenerativeAttackConfig,seed_data=None):
    if config.use_normal:
        _, test_loader = ds.get_loaders(batch_size)
        normal_data = next(iter(test_loader))[0]
    else:
        _, test_loader = ds.get_loaders(100,shuffle=True)
        normal_data = next(iter(test_loader))[0].cuda()
    shape=normal_data[0].shape
    
    if config.start_natural:
        normal_data = ordered_samples(
                X_train_1, X_train_2, test_loader, config.latent_focus,config.n_samples)
        print("Starting with natural data")
    elif seed_data is not None:
            # Use seed data as normal data
        normal_data = seed_data
    x_opt, losses = [], []
    for i in range(config.n_samples):
        print('Gradient ascend')
            # Get optimal point based on local set
            
        x_opt_, loss_ = gen_optimal(
                X_train_1,X_train_2,
                shape, 1,
                config.steps, config.step_size,
                config.latent_focus,
                use_normal=normal_data[i:i +
                                       1].cuda() if (config.use_normal or config.start_natural) else None,
                constrained=config.constrained,
                model_ratio=config.model_ratio,
                clamp=config.clamp)

        x_opt.append(x_opt_)
        losses.append(loss_)

    if config.use_best:
        best_id = np.argmin(losses)
        x_opt = x_opt[best_id:best_id+1]

    x_opt = ch.cat(x_opt, 0)
        
        
    x_use = x_opt
    return x_use.cpu()
class BasicDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        if self.Y is not None:
            
            assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is None:
            return self.X[idx]
        return self.X[idx], self.Y[idx]

class GenerativeAttack(Attack):
    def gen_data(self,m1,m2,ds1,ds2, batch_size:int, config:GenerativeAttackConfig):
        return (generate_data(m1,m2,ds1,batch_size,config),generate_data(m1,m2,ds2,batch_size,config))
    def _get_preds(self,m,x,multi_class:bool=False):
        assert not multi_class
        ps = []
        for model in tqdm(m):
            model = model.cuda()
        # Make sure model is in evaluation mode
            model.eval()
            ch.cuda.empty_cache()
            with ch.no_grad():
                ps.append(model(x.cuda()).detach()[:, 0])
            model = model.cpu()
            del model
            gc.collect()
            ch.cuda.empty_cache()
        ps = ch.stack(ps, 0).to(ch.device('cpu')).numpy()
        
        return ps
    def preds_wrapper(self,m1,m2,x1,x2,multi_class:bool=False):
        p1 = PredictionsOnOneDistribution(
            self._get_preds(m1,x1,multi_class),
            self._get_preds(m2,x1,multi_class))
        p2 = PredictionsOnOneDistribution(
            self._get_preds(m1,x2,multi_class),
            self._get_preds(m2,x2,multi_class))
        return PredictionsOnDistributions(p1,p2)
    def attack(self,preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions):
        self.attack_object = PerPointThresholdAttack(self.config)
        return self.attack_object.attack(preds_adv,preds_vic,ground_truth=(None,None))