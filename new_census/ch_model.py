import numpy as np
import torch.nn as nn
import torch as ch
import os
from tqdm import tqdm
from joblib import load, dump
from model_utils import BASE_MODELS_DIR
from opacus import PrivacyEngine
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

#  Ignore warnings from Opacus
import warnings
warnings.simplefilter("ignore")


BASE_MODELS_DIR = os.path.join(BASE_MODELS_DIR, "ch")


class MLP(nn.Module):
    def __init__(self, n_inp: int, num_classes: int = 1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inp, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def get_model(n_inp: int = 13):
    clf = MLP(n_inp=n_inp).cuda()
    return clf


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, property, split)
    return os.path.join(BASE_MODELS_DIR,  property, split, value)


def validate_model(model):
    """
        Check if model is compatible with Opacus
    """
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        print(str(errors[-5:]))
        raise ValueError("Model is not opacus compatible")


def opacus_stuff(model, train_loader, test_loader, args):
    """
        Train model with DP noise
    """
    # Get size of train dataset from loader
    train_size = len(train_loader.dataset)
    # Compute delta value corresponding to this size
    delta_computed = 1 / train_size
    print(f"Computed Delta {delta_computed} | Given  Delta {args.delta}")

    # Generally, it should be set to be less than the inverse of the size of the training dataset.
    assert args.delta < 1 / len(train_loader.dataset), "delta should be < the inverse of the size of the training dataset"

    device = ch.device("cuda")
    model = model.to(device)

    optimizer = ch.optim.Adam(
        model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    def accuracy(preds, labels):
        # Positive output corresponds to P[1] >= 0.5
        return (1. * ((preds >= 0) == labels)).mean().cpu()

    # Defaults to RDP
    privacy_engine = PrivacyEngine(accountant='rdp')
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        epsilon_tolerance=0.0001  # Lower tolerance gives tighter Sigma values
    )

    print(
        f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")

    def train_opacus(model, train_loader, optimizer, device):
        model.train()
        losses = []
        accuracies = []

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.physical_batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for (datum, target, _) in memory_safe_data_loader:
                optimizer.zero_grad()
                datum = datum.to(device)
                target = target.to(device)

                # compute output
                output = model(datum)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc = accuracy(output.detach(), target)

                losses.append(loss.item())
                accuracies.append(acc)

                loss.backward()
                optimizer.step()

        epsilon = privacy_engine.get_epsilon(args.delta)
        return np.mean(losses), np.mean(accuracies), epsilon

    def test_opacus(model, test_loader, device):
        model.eval()
        losses = []
        accuracies = []

        with ch.no_grad():
            for datum, target, _ in test_loader:
                datum = datum.to(device)
                target = target.to(device)

                output = model(datum)
                loss = criterion(output, target)
                acc = accuracy(output.detach(), target)

                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(accuracies)

    #  EPOCHS
    iterator = range(args.epochs)
    if args.verbose:
        iterator = tqdm(iterator, desc="Epoch", unit="epoch")
    for epoch in iterator:
        loss, acc, epsilon = train_opacus(
            model, train_loader, optimizer, device)
        if args.verbose:
            iterator.set_description(
                f"Epoch {epoch + 1} | Loss: {loss:.3f} | Accuracy: {acc:.3f} | ε={epsilon:.4f}")

    test_acc = test_opacus(model, test_loader, device)
    return test_acc
