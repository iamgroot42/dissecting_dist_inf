import torch as ch
from ogb.nodeproppred import Evaluator
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from distribution_inference.config import TrainConfig
from distribution_inference.utils import warning_string


@ch.no_grad()
def test_epoch(model, ds, train_idx, test_idx, evaluator, loss_fn):
    model.eval()

    X = ds.get_features()
    Y = ds.get_labels()

    out = model(ds.g, X)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': Y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': Y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    out = model(ds.g, X)[test_idx]
    test_loss = loss_fn(out, Y.squeeze(1)[test_idx])

    return train_acc, (test_acc, test_loss)


def train_epoch(model, ds, train_idx, optimizer, loss_fn):
    model.train()

    X = ds.get_features()
    Y = ds.get_labels()

    optimizer.zero_grad()
    out = model(ds.g, X)[train_idx]
    loss = loss_fn(out, Y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, loaders, train_config: TrainConfig,
              input_is_list: bool = False,
              extra_options: dict = None):
    evaluator = Evaluator(name='ogbn-arxiv')

    # Extract data from loaders
    ds, (train_idx, test_idx) = loaders

    metrics = {
        "train": [],
        "test": [],
        "test_loss": []
    }

    optimizer = ch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay)
    loss_fn = ch.nn.CrossEntropyLoss().cuda()
    iterator = tqdm(range(1, 1 + train_config.epochs))
    best_model, best_loss = None, np.inf

    for epoch in iterator:
        # Train epoch
        train_loss = train_epoch(
            model, ds, train_idx,
            optimizer, loss_fn)
        # Test epoch
        train_acc, (test_acc, test_loss) = test_epoch(
            model, ds, train_idx, test_idx,
            evaluator, loss_fn)

        iterator.set_description(f'Epoch: {epoch:02d}, '
                                 f'Loss: {train_loss:.4f}, '
                                 f'Train: {100 * train_acc:.2f}%, '
                                 f'Test: {100 * test_acc:.2f}%')
        
        if best_loss > test_loss:
            best_loss = test_loss
            best_model = deepcopy(model)

        # Keep track of train/test accuracies across runs
        metrics["train"].append(train_acc)
        metrics["test"].append(test_acc)
        metrics["test_loss"].append(test_loss)

    # Evaluate model
    if train_config.get_best:
        print(warning_string("\nUsing test-data to pick best-performing model\n"))
        return best_model, (test_loss, test_acc)
    return model, (test_loss, test_acc)
