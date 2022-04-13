from typing import List, Tuple
import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
import warnings

from distribution_inference.attacks.whitebox.core import Attack
from distribution_inference.attacks.whitebox.permutation.models import PermInvModel, FullPermInvModel, PermInvConvModel
from distribution_inference.config import WhiteBoxAttackConfig, DatasetConfig
from distribution_inference.utils import log, get_save_path, warning_string, ensure_dir_exists


class PINAttack(Attack):
    def __init__(self,
                 dims: List[int],
                 config: WhiteBoxAttackConfig):
        super().__init__(config)
        self.dims = dims

    def _prepare_model(self):
        if isinstance(self.dims, tuple):
            if self.config.permutation_config.focus == "all":
                # If dims is tuple, need joint model
                self.model = FullPermInvModel(self.dims, dropout=0.5)
            elif self.config.permutation_config.focus == "conv":
                # Focus wanted only on conv layers
                self.model = PermInvConvModel(self.dims[0][:2], dropout=0.5)
            elif self.config.permutation_config.focus == "fc":
                self.model = PermInvModel(self.dims[1], dropout=0.5)
            else:
                raise NotImplementedError(
                    f"Focus mode {self.config.permutation_config.focus} not supported")
        else:
            # Focus must be on FC layers
            if self.config.permutation_config.focus != "fc":
                raise AssertionError("Mode must be FC if model has only FC layers")
            # Define meta-classifier
            self.model = PermInvModel(self.dims, dropout=0.5)
        if self.config.gpu:
            self.model = self.model.cuda()

    def _acc_fn(self, x, y):
        if self.config.binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))

    def save_model(self,
                   data_config: DatasetConfig,
                   attack_specific_info_string: str):
        if not self.trained_model:
            warnings.warn(warning_string(
                "\nAttack being saved without training."))
        if self.config.regression_config:
            model_dir = "pin/regression"
        else:
            model_dir = "pin/classification"
        save_path = os.path.join(
            get_save_path(),
            model_dir,
            data_config.name,
            data_config.prop,
            self.config.permutation_config.focus)
        if self.config.regression_config is None:
            save_path = os.path.join(save_path, str(data_config.value))

        # Make sure folder exists
        ensure_dir_exists(save_path)

        model_save_path = os.path.join(
            save_path,
            f"{attack_specific_info_string}.ch")
        ch.save(self.model.state_dict(), model_save_path)

    def _eval_attack(self, test_loader, epochwise_version: bool = False):
        """
            Evaluate attack on test data
        """
        regression = (self.config.regression_config is not None)
        if regression:
            loss_fn = nn.MSELoss()
        else:
            if self.config.binary:
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()

        if epochwise_version:
            evals = []
            # One loader per epoch
            for test_datum_loader in tqdm(test_loader):
                evals.append(
                    self._test(self.model, loss_fn,
                               loader=test_datum_loader,
                               verbose=False)[0])
            return evals
        else:
            return self._test(self.model, loss_fn,
                              loader=test_loader, verbose=True)[0]

    def execute_attack(self,
                       train_loader: Tuple[List, List],
                       test_loader: Tuple[List, List],
                       val_loader: Tuple[List, List] = None,
                       **kwargs):
        """
            Define and train meta-classifier
        """
        # Prepare model
        self._prepare_model()

        # Train PIM
        self.model, chosen_accuracy = self._train(
            model=self.model,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader)
        self.trained_model = True

        return chosen_accuracy

    def _train(self,
               model: nn.Module,
               train_loader: ch.utils.data.DataLoader,
               test_loader: ch.utils.data.DataLoader,
               val_loader: ch.utils.data.DataLoader = None):
        """
            Train meta-classifier.
        """

        # Define optimizer
        optimizer = ch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)

        regression = (self.config.regression_config is not None)

        if regression:
            loss_fn = nn.MSELoss()
        else:
            if self.config.binary:
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()

        # Reserve some data for validation, use this to pick best model
        if val_loader is not None:
            best_loss, best_model = np.inf, None

        # For each epoch
        iterator = tqdm(range(self.config.epochs))
        for e in iterator:
            # Training
            model.train()

            # Iterate through data batches
            running_acc, loss, num_samples = 0, 0, 0
            for param_batch, y_batch in train_loader:
                # Shift to GPU if requested
                if self.config.gpu:
                    y_batch = y_batch.cuda(0)
                    param_batch = [a.cuda() for a in param_batch]

                # Model features stored as list of objects
                outputs = []

                model_output = model(param_batch)

                # Handle binary and regression cases
                if self.config.binary or regression:
                    outputs.append(model_output[:, 0])
                else:
                    outputs.append(model_output)

                # Concatenate outputs from model
                outputs = ch.cat(outputs, 0)

                # Clear accumulated gradients
                optimizer.zero_grad()

                # Compute loss
                loss = loss_fn(outputs, y_batch)

                # Compute gradients
                loss.backward()

                # Take gradient step
                optimizer.step()

                # Keep track of total loss, samples processed so far
                num_samples += outputs.shape[0]
                loss += loss.item() * outputs.shape[0]

                print_acc = ""
                # Track accuracy, if not regression case
                if not regression:
                    running_acc += self._acc_fn(outputs, y_batch)
                    print_acc = ", Accuracy: %.2f" % (
                        100 * running_acc / num_samples)

                iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (
                    e+1, loss / num_samples, print_acc))

            # Evaluate on validation data, if present
            if val_loader is not None:
                v_acc, val_loss = self._test(
                    model, loss_fn, val_loader)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = deepcopy(model)

            # Evaluate on test data now
            if (e+1) % self.config.eval_every == 0:
                if val_loader is not None:
                    print_acc = ""
                    if not regression:
                        print_acc = ", Accuracy: %.2f" % (v_acc)

                    log("[Validation] Loss: %.5f%s" % (val_loss, print_acc))

                # Also log test-data metrics
                t_acc, t_loss = self._test(
                    model, loss_fn, test_loader)
                print_acc = ""
                if not regression:
                    print_acc = ", Accuracy: %.2f" % (t_acc)

                log("[Test] Loss: %.5f%s" % (t_loss, print_acc))
                print()

        # Pick best model (according to validation), if requested
        # And compute test accuracy on this model
        if val_loader is not None:
            t_acc, t_loss = self._test(
                best_model, loss_fn, val_loader)
            model = deepcopy(best_model)

        # Make sure model is in evaluation mode
        model.eval()

        # Compute test accuracy on this model
        t_acc, t_loss = self._test(
            model, loss_fn, test_loader)

        if regression:
            return model, t_loss
        return model, t_acc

    @ch.no_grad()
    def _test(self,
              model, loss_fn,
              loader: ch.utils.data.DataLoader,
              element_wise: bool = False,
              get_preds: bool = False,
              verbose: bool = False):
        # Set model to evaluation mode
        model.eval()
        regression = (self.config.regression_config is not None)

        # Batch data to fit on GPU
        num_samples, running_acc = 0, 0
        loss = [] if element_wise else 0
        all_outputs = []

        iterator = loader
        if verbose:
            iterator = tqdm(loader)
        for param_batch, y_batch in iterator:
            outputs = []
            if self.config.gpu:
                y_batch = y_batch.cuda(0)
                param_batch = [a.cuda() for a in param_batch]

            model_output = model(param_batch)
            # Handle binary and regression cases
            if self.config.binary or regression:
                outputs.append(model_output[:, 0])
            else:
                outputs.append(model_output)

            # Concatenate outputs from model
            outputs = ch.cat(outputs, 0)
            if get_preds:
                # Track model predictions as well, if requested
                all_outputs.append(outputs.cpu().detach().numpy())

            num_samples += outputs.shape[0]
            losses = loss_fn(outputs, y_batch)
            # Return element-wise loss ,if requested
            if element_wise:
                loss.append(losses.detach().cpu())
            else:
                loss += losses.item() * num_samples

            # Track accuracy, if not regression case
            if not regression:
                running_acc += self._acc_fn(outputs, y_batch).item()

        if element_wise:
            loss = ch.cat(loss, 0)
        else:
            loss /= num_samples

        if get_preds:
            all_outputs = np.concatenate(all_outputs, axis=0)
            return 100 * running_acc / num_samples, loss, all_outputs

        return 100 * running_acc / num_samples, loss
