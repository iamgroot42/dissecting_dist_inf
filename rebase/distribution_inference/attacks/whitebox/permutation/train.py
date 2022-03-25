"""
    Core code for training permutation-invariant meta-classifiers.
"""
import torch as ch
import torch.nn as nn


@ch.no_grad()
def test(model, loss_fn, X, Y, batch_size, accuracy,
              binary: bool = True, regression=False, gpu: bool = False,
              combined: bool = False, X_acts=None,
              element_wise=False, get_preds: bool = False):
    model.eval()
    use_acts = (X_acts is not None)
    # Activations must be provided if not combined
    assert (not use_acts) or (
        combined), "Activations must be provided if not combined"

    # Batch data to fit on GPU
    num_samples, running_acc = 0, 0
    loss = [] if element_wise else 0
    all_outputs = []

    i = 0

    if combined:
        n_samples = len(X[0])
    else:
        n_samples = len(X)

    while i < n_samples:
        # Model features stored as list of objects
        outputs = []
        if not combined:
            for param in X[i:i+batch_size]:
                # Shift to GPU, if requested
                if gpu:
                    param = [a.cuda() for a in param]

                if binary or regression:
                    outputs.append(model(param)[:, 0])
                else:
                    outputs.append(model(param))
        # Model features stored as normal list
        else:
            param_batch = [x[i:i+batch_size] for x in X]
            if use_acts:
                acts_batch = X_acts[i:i+batch_size]
            if gpu:
                param_batch = [a.cuda() for a in param_batch]

            if binary or regression:
                if use_acts:
                    outputs.append(model(param_batch, acts_batch)[:, 0])
                else:
                    outputs.append(model(param_batch)[:, 0])
            else:
                if use_acts:
                    outputs.append(model(param_batch, acts_batch))
                else:
                    outputs.append(model(param_batch))

        outputs = ch.cat(outputs, 0)
        if get_preds:
            all_outputs.append(outputs.cpu().detach().numpy())

        num_samples += outputs.shape[0]
        if element_wise:
            loss.append(loss_fn(outputs, Y[i:i+batch_size]).detach().cpu())
        else:
            loss += loss_fn(outputs,
                            Y[i:i+batch_size]).item() * num_samples
        if not regression:
            running_acc += accuracy(outputs, Y[i:i+batch_size]).item()

        # Next batch
        i += batch_size

    if element_wise:
        loss = ch.cat(loss, 0)
    else:
        loss /= num_samples

    if get_preds:
        all_outputs = np.concatenate(all_outputs, axis=0)
        return 100 * running_acc / num_samples, loss, all_outputs

    return 100 * running_acc / num_samples, loss


# Function to train meta-classifier
def train(model, train_data, test_data,
          epochs, lr, eval_every: int = 5,
          binary: bool = True, regression: bool = False,
          val_data=None, batch_size: int = 1000,
          gpu: bool = False, combined: bool = False,
          shuffle: bool = True, train_acts = None,
          test_acts = None, val_acts = None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # Make sure both weights and activations available if val requested
    assert (val_data is not None or val_acts is None), "Weights or activations for validation data must be provided"

    use_acts = (train_acts is not None)
    # Activations must be provided if not combined
    assert (not use_acts) or (
        combined), "Activations must be provided if not combined"

    if regression:
        loss_fn = nn.MSELoss()
    else:
        if binary:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

    params, y = train_data
    params_test, y_test = test_data

    # Shift to GPU, if requested
    if gpu:
        y = y.cuda()
        y_test = y_test.cuda()

    # Reserve some data for validation, use this to pick best model
    if val_data is not None:
        params_val, y_val = val_data
        best_loss, best_model = np.inf, None
        if gpu:
            y_val = y_val.cuda()

    def acc_fn(x, y):
        if binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))

    iterator = tqdm(range(epochs))
    for e in iterator:
        # Training
        model.train()

        # Shuffle train data
        if shuffle:
            rp_tr = np.random.permutation(y.shape[0])
            if not combined:
                params, y = params[rp_tr], y[rp_tr]
            else:
                y = y[rp_tr]
                params = [x[rp_tr] for x in params]
            if use_acts:
                train_acts = train_acts[rp_tr]

        # Batch data to fit on GPU
        running_acc, loss, num_samples = 0, 0, 0
        i = 0

        if combined:
            n_samples = len(params[0])
        else:
            n_samples = len(params)

        while i < n_samples:

            # Model features stored as list of objects
            outputs = []
            if not combined:
                for param in params[i:i+batch_size]:
                    # Shift to GPU, if requested
                    if gpu:
                        param = [a.cuda() for a in param]

                    if binary or regression:
                        outputs.append(model(param)[:, 0])
                    else:
                        outputs.append(model(param))
            # Model features stored as normal list
            else:
                param_batch = [x[i:i+batch_size] for x in params]
                if use_acts:
                    acts_batch = train_acts[i:i+batch_size]
                if gpu:
                    param_batch = [a.cuda() for a in param_batch]

                if binary or regression:
                    if use_acts:
                        outputs.append(model(param_batch, acts_batch)[:, 0])
                    else:
                        outputs.append(model(param_batch)[:, 0])
                else:
                    if use_acts:
                        outputs.append(model(param_batch, acts_batch))
                    else:
                        outputs.append(model(param_batch))

            outputs = ch.cat(outputs, 0)

            # Clear accumulated gradients
            optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(outputs, y[i:i+batch_size])

            # Compute gradients
            loss.backward()

            # Take gradient step
            optimizer.step()

            # Keep track of total loss, samples processed so far
            num_samples += outputs.shape[0]
            loss += loss.item() * outputs.shape[0]

            print_acc = ""
            if not regression:
                running_acc += acc_fn(outputs, y[i:i+batch_size])
                print_acc = ", Accuracy: %.2f" % (
                    100 * running_acc / num_samples)

            iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (
                e+1, loss / num_samples, print_acc))

            # Next batch
            i += batch_size

        # Evaluate on validation data, if present
        if val_data is not None:
            v_acc, val_loss = test_meta(model, loss_fn, params_val,
                                        y_val, batch_size, acc_fn,
                                        binary=binary, regression=regression,
                                        gpu=gpu, combined=combined,
                                        X_acts=val_acts)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(model)

        # Evaluate on test data now
        if (e+1) % eval_every == 0:
            if val_data is not None:
                print_acc = ""
                if not regression:
                    print_acc = ", Accuracy: %.2f" % (v_acc)

                utils.log("[Validation] Loss: %.5f%s" % (val_loss, print_acc))

            # Also log test-data metrics
            t_acc, t_loss = test_meta(model, loss_fn, params_test,
                                      y_test, batch_size, acc_fn,
                                      binary=binary, regression=regression,
                                      gpu=gpu, combined=combined,
                                      X_acts=test_acts)
            print_acc = ""
            if not regression:
                print_acc = ", Accuracy: %.2f" % (t_acc)

            utils.log("[Test] Loss: %.5f%s" % (t_loss, print_acc))
            print()

    # Pick best model (according to validation), if requested
    # And compute test accuracy on this model
    if val_data is not None:
        t_acc, t_loss = test_meta(best_model, loss_fn, params_test,
                                  y_test, batch_size, acc_fn,
                                  binary=binary, regression=regression,
                                  gpu=gpu, combined=combined,
                                  X_acts=test_acts)
        model = deepcopy(best_model)

    # Make sure model is in evaluation mode
    model.eval()

    if regression:
        return model, t_loss
    return model, t_acc


def coordinate_descent(models_train, models_val,
                       models_test, dims, reduction_dims,
                       get_activation_fn,
                       n_samples, meta_train_args,
                       gen_optimal_fn, seed_data,
                       n_times: int = 10,
                       restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        dims: Dimensions of feature activations.
        reduction_dims: Dimensions for meta-classifier internal models.
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = ActivationMetaClassifier(
                    n_samples, dims,
                    reduction_dims=reduction_dims)
    metamodel = metamodel.cuda()

    best_clf, best_tacc = None, 0
    val_data = None
    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        X_tr, Y_tr = wrap_data_for_act_meta_clf(
            models_train[0], models_train[1], seed_data, get_activation_fn)
        X_te, Y_te = wrap_data_for_act_meta_clf(
            models_test[0], models_test[1], seed_data, get_activation_fn)
        if models_val is not None:
            val_data = wrap_data_for_act_meta_clf(
                models_val[0], models_val[1], seed_data, get_activation_fn)

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = ActivationMetaClassifier(
                n_samples, dims,
                reduction_dims=reduction_dims)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        metamodel.train()
        clf, tacc = train_meta_model(
                    metamodel,
                    (X_tr, Y_tr), (X_te, Y_te),
                    epochs=meta_train_args['epochs'],
                    binary=True, lr=1e-3,
                    regression=False,
                    batch_size=meta_train_args['batch_size'],
                    val_data=val_data, combined=True,
                    eval_every=10, gpu=True)
        all_accs.append(tacc)

        # Keep track of best model and latest model
        if tacc > best_tacc:
            best_tacc = tacc
            best_clf = clf

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data, get_activation_fn)

    # Return best and latest models
    return (best_tacc, best_clf), (tacc, clf), all_accs
