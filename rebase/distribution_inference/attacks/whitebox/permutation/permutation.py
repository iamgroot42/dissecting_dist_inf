from typing import List, Tuple
import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from distribution_inference.attacks.whitebox.core import Attack
from distribution_inference.attacks.whitebox.permutation.models import PermInvModel
from distribution_inference.config import WhiteBoxAttackConfig
from distribution_inference.utils import log


class PINAttack(Attack):
    def __init__(self,
                 dims: List[int],
                 config: WhiteBoxAttackConfig):
        super().__init__(config)
        self.dims = dims

    def _prepare_model(self):
        # Define meta-classifier
        self.model = PermInvModel(self.dims, dropout=0.5)
        if self.config.gpu:
            self.model = self.model.cuda()

    def _acc_fn(self, x, y):
        if self.config.binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))

    def execute_attack(self,
                       train_data: Tuple[List, List],
                       test_data: Tuple[List, List],
                       val_data: Tuple[List, List] = None,
                       **kwargs):
        """
            Define and train meta-classifier
        """
        # Prepare model
        self._prepare_model()

        # Extract aux data, if given
        train_aux = kwargs.get("train_aux", None)
        test_aux = kwargs.get("test_aux", None)
        val_aux = kwargs.get("val_aux", None)

        # Train PIM
        self.model, chosen_accuracy = self._train(
            model=self.model,
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            train_aux=train_aux,
            test_aux=test_aux,
            val_aux=val_aux)

        return chosen_accuracy

    # Function to train meta-classifier
    def _train(self,
               model: nn.Module,
               train_data: Tuple[List, List],
               test_data: Tuple[List, List],
               val_data: Tuple[List, List] = None,
               train_aux: List = None,
               val_aux: List = None,
               test_aux: List = None):
        """
            Train meta-classifier.
        """

        # Define optimizer
        optimizer = ch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)

        # Make sure both weights and activations available if val requested
        assert (val_data is not None or val_aux is None), "Weights or activations for validation data must be provided"

        use_aux = (train_aux is not None)
        regression = (self.config.regression_config is not None)

        if regression:
            loss_fn = nn.MSELoss()
        else:
            if self.config.binary:
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()

        params, y = train_data
        params_test, y_test = test_data

        # Reserve some data for validation, use this to pick best model
        if val_data is not None:
            params_val, y_val = val_data
            best_loss, best_model = np.inf, None

        # Shift to GPU, if requested
        if self.config.gpu:
            y = y.cuda()
            y_test = y_test.cuda()
            y_val = y_val.cuda() if val_data is not None else None

        # For each epoch
        iterator = tqdm(range(self.config.epochs))
        for e in iterator:
            # Training
            model.train()

            # Shuffle train data
            if self.config.shuffle:
                shuffle_indices = np.random.permutation(y.shape[0])
                y = y[shuffle_indices]
                params = [x[shuffle_indices] for x in params]
                if use_aux:
                    train_aux = train_aux[shuffle_indices]

            # Batch data to fit on GPU
            running_acc, loss, num_samples = 0, 0, 0
            i = 0

            n_samples = len(params[0])

            # For each batch
            while i < n_samples:

                # Model features stored as list of objects
                outputs = []
                # Create batch of model parameters
                param_batch = [x[i:i+self.config.batch_size] for x in params]
                # Create batch of auxiliary data too, if requested
                if use_aux:
                    aux_batch = train_aux[i:i+self.config.batch_size]
                if self.config.gpu:
                    param_batch = [a.cuda() for a in param_batch]
                    aux_batch = aux_batch.cuda() if use_aux else None

                # Pass auxiliary data to model, if available
                if use_aux:
                    model_output = model(param_batch, aux_batch)
                else:
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
                loss = loss_fn(outputs, y[i:i+self.config.batch_size])

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
                    running_acc += self._acc_fn(outputs, y[i:i+self.config.batch_size])
                    print_acc = ", Accuracy: %.2f" % (
                        100 * running_acc / num_samples)

                iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (
                    e+1, loss / num_samples, print_acc))

                # Next batch
                i += self.config.batch_size

            # Evaluate on validation data, if present
            if val_data is not None:
                v_acc, val_loss = self._test(
                    model, loss_fn, params_val,
                    y_val, X_acts=val_aux)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = deepcopy(model)

            # Evaluate on test data now
            if (e+1) % self.config.eval_every == 0:
                if val_data is not None:
                    print_acc = ""
                    if not regression:
                        print_acc = ", Accuracy: %.2f" % (v_acc)

                    log("[Validation] Loss: %.5f%s" % (val_loss, print_acc))

                # Also log test-data metrics
                t_acc, t_loss = self._test(
                    model, loss_fn, params_test,
                    y_test, X_acts=test_aux)
                print_acc = ""
                if not regression:
                    print_acc = ", Accuracy: %.2f" % (t_acc)

                log("[Test] Loss: %.5f%s" % (t_loss, print_acc))
                print()

        # Pick best model (according to validation), if requested
        # And compute test accuracy on this model
        if val_data is not None:
            t_acc, t_loss = self._test(
                best_model, loss_fn, params_test,
                y_test, X_acts=test_aux)
            model = deepcopy(best_model)

        # Make sure model is in evaluation mode
        model.eval()

        if regression:
            return model, t_loss
        return model, t_acc

    @ch.no_grad()
    def _test(self,
              model, loss_fn, X, Y,
              X_acts: List = None,
              element_wise: bool = False,
              get_preds: bool = False):
        # Set model to evaluation mode
        model.eval()
        use_aux = (X_acts is not None)
        regression = (self.config.regression_config is not None)

        # Batch data to fit on GPU
        num_samples, running_acc = 0, 0
        loss = [] if element_wise else 0
        all_outputs = []

        i = 0
        n_samples = len(X[0])

        while i < n_samples:
            outputs = []
            param_batch = [x[i:i+self.config.batch_size] for x in X]
            if use_aux:
                aux_batch = X_acts[i:i+self.config.batch_size]
            if self.config.gpu:
                param_batch = [a.cuda() for a in param_batch]

            # Pass auxiliary data to model, if available
            if use_aux:
                model_output = model(param_batch, aux_batch)
            else:
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
            losses = loss_fn(outputs, Y[i:i+self.config.batch_size])
            # Return element-wise loss ,if requested
            if element_wise:
                loss.append(losses.detach().cpu())
            else:
                loss += losses.item() * num_samples

            # Track accuracy, if not regression case
            if not regression:
                running_acc += self._acc_fn(outputs, Y[i:i+self.config.batch_size]).item()

            # Next batch
            i += self.config.batch_size

        if element_wise:
            loss = ch.cat(loss, 0)
        else:
            loss /= num_samples

        if get_preds:
            all_outputs = np.concatenate(all_outputs, axis=0)
            return 100 * running_acc / num_samples, loss, all_outputs

        return 100 * running_acc / num_samples, loss
