
import torch as ch
import numpy as np
from tqdm import tqdm

from distribution_inference.config import UnlearningConfig


class Unlearning:
    def __init__(self, config: UnlearningConfig):
        self.config = config
        self.stop_tol = self.config.stop_tol
        self.flip_tol = self.config.flip_tol
        self.flip_weight_ratio = self.config.flip_weight_ratio
        self.max_iters = self.config.max_iters
        self.k = self.config.k
        self.lr = self.config.learning_rate
        self.min_lr = self.config.min_lr

    def _wrap_pred(self, pred):
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            return ch.softmax(pred)
        pred_ = ch.sigmoid(pred)
        pred_neg_ = 1 - pred_
        return ch.stack([pred_neg_, pred_], dim=1)

    def defend(self, attacker_obj, victim_model,
               process_fn, verbose: bool = False):
        """
            Uses given meta-model, along with given victim model
            to unlearn 'property' features from data
        """
        # Make sure meta-classifier is on GPU
        attacker_obj.to_gpu()

        lr_ = self.lr
        victim_model_ = victim_model.cuda()

        # Take note of initial predicted class
        victim_features = process_fn(victim_model_)
        y_initial = self._wrap_pred(attacker_obj.get_pred(victim_features))
        y_initial = y_initial.detach().cpu().numpy()

        # Code below is annoted with line numbers corresponding to
        # Algorithm 1 in paper

        def _get_adv_utility(x):
            return ch.max(ch.abs(x - 1 / self.k))  # Line 23

        def _flip_parameters(x):
            for _, param in x.named_parameters():
                if param.requires_grad:
                    mask = ch.ones(param.numel()).cuda()
                    num_elems_to_set = int(
                        param.numel() * self.flip_weight_ratio)
                    mask[ch.randperm(len(mask))[num_elems_to_set]] = -1
                    # Reshape it to match param size
                    mask = mask.view(param.size())
                    # Set param with updated mask
                    param.data.mul_(mask)
            return x

        iterator = range(self.max_iters)
        if verbose:
            iterator = tqdm(iterator)

        prev_utility = np.inf
        for _ in iterator:
            victim_features = process_fn(victim_model_)
            y_i = self._wrap_pred(attacker_obj.get_pred(victim_features))  # Line 3

            if prev_utility <= self.stop_tol:  # Line 5
                break

            # If perfect prediction
            while ch.max(y_i) >= 1 - self.flip_tol:  # Line 6
                # Randomly flip weights of model  ## Line 7
                victim_model_ = _flip_parameters(victim_model_)
                victim_features = process_fn(victim_model_)
                y_i = self._wrap_pred(
                    attacker_obj.get_pred(victim_features))  # Line 8

            # Get gradients that would make all predictions approach 1/k
            loss = ch.mean((y_i - 1 / self.k) ** 2)
            grads = ch.autograd.grad(
                loss, victim_model_.parameters())  # Line 10

            # Apply gradients with given lr  # Line 11
            for param, grad in zip(victim_model_.parameters(), grads):
                param.data.sub_(lr_ * grad)

            victim_features = process_fn(victim_model_)
            y_i_updated = self._wrap_pred(
                attacker_obj.get_pred(victim_features))  # Line 12

            updated_utility = _get_adv_utility(y_i_updated)
            if updated_utility > prev_utility:  # Line 13
                # Undo weight param changes
                for param, grad in zip(victim_model_.parameters(), grads):
                    param.data.add_(lr_ * grad)
                lr_ /= 2  # Line 16
                # Since model not changed, utility remains the same
                updated_utility = prev_utility

            if lr_ < self.min_lr:
                break

            # Keep track of previous utility
            prev_utility = updated_utility

            if verbose:
                iterator.set_description(
                    "Loss: %4f | Utility: %.3f | Lr (log): %.4f" %
                        (loss, updated_utility.item(), np.log(lr_)))

        # Take note of final predicted class
        victim_features = process_fn(victim_model_)
        y_final = self._wrap_pred(attacker_obj.get_pred(victim_features))
        y_final = y_final.detach().cpu().numpy()

        # Shift new model to CPU
        victim_model_.cpu()

        return victim_model_, (y_initial[0], y_final[0])
