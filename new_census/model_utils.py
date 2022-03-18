import numpy as np
import torch.nn as nn
import torch as ch
import os
from tqdm import tqdm
from opacus import PrivacyEngine
import numpy as np
from utils import check_if_inside_cluster
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

#  Ignore warnings from Opacus
import warnings
warnings.simplefilter("ignore")


if check_if_inside_cluster():
    BASE_MODELS_DIR = "/scratch/as9rw/models_new_census/60_40"
else:
    BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_new_census/60_40"


class MLP(nn.Module):
    def __init__(self, n_inp: int, num_classes: int = 1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inp, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def get_model(n_inp: int = 105):
    clf = MLP(n_inp=n_inp).cuda()
    return clf


def save_model(clf, path):
    state_dict = clf.state_dict()
    ch.save(state_dict, path)


def load_model(path):
    # TODO: Implement
    pass


def get_models_path(property, split, value=None, is_dp=None):
    if is_dp:
        base_path = os.path.join(BASE_MODELS_DIR, is_dp)
    else:
        base_path = os.path.join(BASE_MODELS_DIR, "normal")
    path = os.path.join(base_path, property, split)
    if value is not None:
        path = os.path.join(path, value)
    return path


def validate_model(model):
    """
        Check if model is compatible with Opacus
    """
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        print(str(errors[-5:]))
        raise ValueError("Model is not opacus compatible")


def train_model_with_dp(model, train_loader, test_loader, args):
    """
        Train model with DP noise
    """
    # Get size of train dataset from loader
    train_size = len(train_loader.dataset)
    # Compute delta value corresponding to this size
    delta_computed = 1 / train_size
    if args.verbose:
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
    # privacy_engine = PrivacyEngine(accountant='gdp')
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
    if args.verbose:
        print(f"Using sigma={optimizer.noise_multiplier}")

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

        return np.mean(losses), np.mean(accuracies)

    #  EPOCHS
    iterator = range(args.epochs)
    if args.verbose:
        iterator = tqdm(iterator, desc="Epoch", unit="epoch")
    for epoch in iterator:
        loss, acc, epsilon = train_opacus(
            model, train_loader, optimizer, device)
        test_loss, test_acc = test_opacus(model, test_loader, device)
        if args.verbose:
            iterator.set_description(
                f"Epoch {epoch + 1} | Train Loss: {loss:.4f} | Train Accuracy: {acc:.3f} | ε={epsilon:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    _, test_acc = test_opacus(model, test_loader, device)
    return test_acc
