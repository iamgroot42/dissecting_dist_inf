import os
import torch as ch
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent

from distribution_inference.config import AttackConfig

def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def extract_adv_params(
        eps: float, eps_iter, nb_iter: int, norm,
        random_restarts, clip_min, clip_max):
    adv_params = {}
    adv_params["eps"] = eps
    adv_params["eps_iter"] = eps_iter
    adv_params["nb_iter"] = nb_iter
    adv_params["norm"] = norm
    adv_params["clip_min"] = clip_min
    adv_params["clip_max"] = clip_max
    adv_params["random_restarts"] = random_restarts

    return adv_params


def save_model(model, path):
    ch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(ch.load(path))
    return model


def generate_adversarial_input(model, data, adv_config: AttackConfig):
    adv_data = projected_gradient_descent(
                model, data,
                eps=adv_config.epsilon,
                eps_iter=adv_config.epsilon_iter,
                nb_iter=adv_config.iters,
                norm=adv_config.norm,
                clip_min=adv_config.clip_min,
                clip_max=adv_config.clip_max,
                random_restarts=adv_config.random_restarts,
                binary_sigmoid=True)
    return adv_data
