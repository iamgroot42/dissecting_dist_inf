import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from opacus.privacy_engine import PrivacyEngine

from distribution_inference.config import TrainConfig

#  Ignore warnings from Opacus
import warnings
warnings.simplefilter("ignore")


def validate_model(model):
    """
        Check if model is compatible with Opacus
    """
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        print(str(errors[-5:]))
        raise ValueError("Model is not opacus compatible")


def train(model, loaders, train_config: TrainConfig):
    """
        Train model with DP noise
    """
    # Get data loaders
    train_loader, val_loader = loaders

    # Extract configuration specific to DP
    dp_config = train_config.dp_config
    if dp_config is None:
        raise ValueError("DP configuration is not specified")

    # Get size of train dataset from loader
    train_size = len(train_loader.dataset)
    # Compute delta value corresponding to this size
    delta_computed = 1 / train_size
    if train_config.verbose:
        print(
            f"Computed Delta {delta_computed} | Given  Delta {dp_config.delta}")

    # Generally, it should be set to be less than the inverse of the size of the training dataset.
    assert dp_config.delta < 1 / len(train_loader.dataset), "delta should be < the inverse of the size of the training dataset"

    device = ch.device("cuda")
    model = model.to(device)

    optimizer = ch.optim.Adam(
        model.parameters(), lr=train_config.learning_rate)
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
        epochs=train_config.epochs,
        target_epsilon=dp_config.epsilon,
        target_delta=dp_config.delta,
        max_grad_norm=dp_config.max_grad_norm,
        epsilon_tolerance=0.0001  # Lower tolerance gives tighter Sigma values
    )
    if train_config.verbose:
        print(f"Using sigma={optimizer.noise_multiplier}")

    def train_opacus(model, train_loader, optimizer, device):
        model.train()
        losses = []
        accuracies = []

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=dp_config.physical_batch_size,
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

        epsilon = privacy_engine.get_epsilon(dp_config.delta)
        return np.mean(losses), np.mean(accuracies), epsilon

    def test_opacus(model, val_loader, device):
        model.eval()
        losses = []
        accuracies = []

        with ch.no_grad():
            for datum, target, _ in val_loader:
                datum = datum.to(device)
                target = target.to(device)

                output = model(datum)
                loss = criterion(output, target)
                acc = accuracy(output.detach(), target)

                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

    #  EPOCHS
    iterator = range(train_config.epochs)
    iterator = tqdm(iterator, desc="Epoch", unit="epoch")
    for epoch in iterator:
        loss, acc, epsilon = train_opacus(
            model, train_loader, optimizer, device)
        test_loss, test_acc = test_opacus(model, val_loader, device)
        iterator.set_description(
            f"Epoch {epoch + 1} | Train Loss: {loss:.4f} | Train Accuracy: {acc:.3f} | ε={epsilon:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    test_loss, test_acc = test_opacus(model, val_loader, device)

    return model, (test_loss, test_acc)
