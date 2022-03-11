import numpy as np
import torch.nn as nn
import torch as ch
import os
from tqdm import tqdm
from joblib import load, dump
from model_utils import BASE_MODELS_DIR
from utils import AverageMeter
import utils 
from copy import deepcopy
from data_utils import CensusSet
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from opacus import PrivacyEngine
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager


from opacus.validators import ModuleValidator
#Constants
import warnings
warnings.simplefilter("ignore")


BASE_MODELS_DIR = os.path.join(BASE_MODELS_DIR, "ch")

class MLP(nn.Module):
    def __init__(self,n_inp:int, num_classes: int = 1):
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


def get_model(n_inp:int=13):
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


#Check if opacus compatible
def validate_model(model):

    #model = ModuleValidator.fix(model)
    #ModuleValidator.validate(model, strict=False)
    print("Validating model...")
    errors = ModuleValidator.validate(model, strict=False)
    print(str(errors[-5:]))
    if(errors == []):
        print("Validated")


def opacus_stuff(model, train_loader, test_loader, args):
    """
        Train model with DP noise
    """
    # The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.
    # Tuning MAX_GRAD_NORM is very important. Start with a low noise multiplier like .1, this should give 
    # comparable performance to a non-private model. Then do a grid search for the optimal MAX_GRAD_NORM value.
    # The grid can be in the range [.1, 10].
    MAX_GRAD_NORM = 1.2
    DELTA = 1e-5
    # Generally, it should be set to be less than the inverse of the size of the training dataset.
    assert DELTA < 1 / len(train_loader.dataset), "DELTA should be less than the inverse of the size of the training dataset"

    # Get size of train dataset from loader
    train_size = len(train_loader.dataset)
    # Compute delta value corresponding to this size

    # Peak memory is proportional to batch_size ** 2
    # This physical batch size should be set accordingly
    MAX_PHYSICAL_BATCH_SIZE = 128 
    device = ch.device("cpu")#"cuda" if ch.cuda.is_available() else "cpu")
    model = model.to(device)

    #optimizer and criterion taken from utils
    optimizer = ch.optim.Adam(
        model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    def accuracy(preds, labels):
        # Positive output corresponds to P[1] >= 0.5
        print(preds.shape, labels.shape)
        exit(0)
        return ((preds >=0) == labels).mean()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    def train_opacus(model, train_loader, optimizer, epoch, device):
        model.train()
        losses = []
        top1_acc = []
        
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for i, (datum, target, _) in enumerate(memory_safe_data_loader):   
                optimizer.zero_grad()
                datum = datum.to(device)
                target = target.to(device)
                target = target.type(ch.LongTensor)

                # compute output
                output = model(datum)
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

                loss.backward()
                optimizer.step()

                if (i+1) % 200 == 0:
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )

    def test_opacus(model, test_loader, device):
        model.eval()
        losses = []
        top1_acc = []

        with ch.no_grad():
            for datum, target,_ in test_loader:
                datum = datum.to(device)
                target = target.to(device)
                target = target.type(ch.LongTensor)

                output = model(datum)
                loss = criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

        top1_avg = np.mean(top1_acc)

        print(
            f"\tTest set:"
            f"Loss: {np.mean(losses):.6f} "
            f"Acc: {top1_avg * 100:.6f} "
        )
        return np.mean(top1_acc)
    
    #EPOCHS
    for epoch in tqdm(range(args.epochs), desc="Epoch", unit="epoch"):
        train_opacus(model, train_loader, optimizer, epoch + 1, device)

    top1_acc = test_opacus(model, test_loader, device)