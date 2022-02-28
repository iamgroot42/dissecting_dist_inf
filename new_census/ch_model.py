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
from torch.utils.data import   DataLoader
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent


from opacus.validators import ModuleValidator
#Constants
import warnings
warnings.simplefilter("ignore")


BM = os.path.join(BASE_MODELS_DIR,"ch")
class MLP(nn.Module):
    def __init__(self,n_inp:int,num_classes: int = 2):
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
        return os.path.join(BM, property, split)
    return os.path.join(BM,  property, split, value)
'''
def train(clf,loaders):
    (vloss,tacc, vacc) = _train(clf,(DataLoader(CensusSet(loaders[0][0],loaders[0][1]),batch_size=200),DataLoader(CensusSet(loaders[1][0],loaders[1][1]),batch_size=200)),
    epoch_num=40,
    verbose=False
    )
    return vloss,tacc,vacc'''
#method copied from utils, almost the same, might need modification for differential privacy
def train_epoch(train_loader, model, criterion, optimizer, epoch, verbose=True, adv_train=False):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    if verbose:
        iterator = tqdm(train_loader)
    for data in iterator:
        images, labels, _ = data
        images, labels = images.cuda(), labels.cuda()
        N = images.size(0)

        if adv_train is False:
            # Clear accumulated gradients
            optimizer.zero_grad()
            outputs = model(images)[:, 0]
        else:
            # Adversarial inputs
            adv_x = projected_gradient_descent(
                model, images, eps=adv_train['eps'],
                eps_iter=adv_train['eps_iter'],
                nb_iter=adv_train['nb_iter'],
                norm=adv_train['norm'],
                clip_min=adv_train['clip_min'],
                clip_max=adv_train['clip_max'],
                random_restarts=adv_train['random_restarts'],
                binary_sigmoid=True)
            # Important to zero grad after above call, else model gradients
            # get accumulated over attack too
            optimizer.zero_grad()
            outputs = model(adv_x)[:, 0]

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        prediction = (outputs >= 0)
        train_acc.update(prediction.eq(
            labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        if verbose:
            iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (
                epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg

def _train(model, loaders, lr=1e-3, epoch_num=10,
          weight_decay=0, verbose=True, get_best=False,
          adv_train=False):
    # Get data loaders
    train_loader, val_loader = loaders

    # Define optimizer, loss function
    optimizer = ch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().cuda()

    iterator = range(1, epoch_num+1)
    if not verbose:
        iterator = tqdm(iterator)

    best_model, best_loss = None, np.inf
    for epoch in iterator:
        _, tacc = train_epoch(train_loader, model,
                              criterion, optimizer, epoch,
                              verbose=verbose, adv_train=adv_train)

        vloss, vacc = validate_epoch(
            val_loader, model, criterion, verbose=verbose,
            adv_train=adv_train)
        if not verbose:
            if adv_train is False:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))
            else:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f | adv_val_acc: %.2f" % (tacc, vacc[0], vacc[1]))

        vloss_compare = vloss
        if adv_train is not False:
            vloss_compare = vloss[0]

        if get_best and vloss_compare < best_loss:
            best_loss = vloss_compare
            best_model = deepcopy(model)

    if get_best:
        return best_model, (vloss, tacc,vacc)
    return vloss, tacc,vacc
def validate_epoch(val_loader, model, criterion, verbose=True, adv_train=False):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    adv_val_loss = AverageMeter()
    adv_val_acc = AverageMeter()

    with ch.set_grad_enabled(adv_train is not False):
        for data in val_loader:
            images, labels, _ = data
            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)

            outputs = model(images)[:, 0]
            prediction = (outputs >= 0)

            if adv_train is not False:
                adv_x = projected_gradient_descent(
                    model, images, eps=adv_train['eps'],
                    eps_iter=adv_train['eps_iter'],
                    nb_iter=adv_train['nb_iter'],
                    norm=adv_train['norm'],
                    clip_min=adv_train['clip_min'],
                    clip_max=adv_train['clip_max'],
                    random_restarts=adv_train['random_restarts'],
                    binary_sigmoid=True)
                outputs_adv = model(adv_x)[:, 0]
                prediction_adv = (outputs_adv >= 0)

                adv_val_acc.update(prediction_adv.eq(
                    labels.view_as(prediction_adv)).sum().item()/N)

                adv_val_loss.update(
                    criterion(outputs_adv, labels.float()).item())

            val_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels.float()).item())

    if verbose:
        if adv_train is False:
            print('[Validation], Loss: %.5f, Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg))
        else:
            print('[Validation], Loss: %.5f, Accuracy: %.4f | Adv-Loss: %.5f, Adv-Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg,
                   adv_val_loss.avg, adv_val_acc.avg))
        print()

    if adv_train is False:
        return val_loss.avg, val_acc.avg
    return (val_loss.avg, adv_val_loss.avg), (val_acc.avg, adv_val_acc.avg)


#Check if opacus compatible
def validate_model(model):

    #model = ModuleValidator.fix(model)
    #ModuleValidator.validate(model, strict=False)
    print("Validating model...")
    errors = ModuleValidator.validate(model, strict=False)
    print(str(errors[-5:]))
    if(errors == []):
        print("Validated")


def opacus_stuff(model, loaders, lr=1e-3, weight_decay=0):
    MAX_GRAD_NORM = 1.2
    EPSILON = 50.0
    DELTA = 1e-5
    EPOCHS = 20

    LR = 1e-3
    #memory management
    BATCH_SIZE = 512
    MAX_PHYSICAL_BATCH_SIZE = 128
    device = ch.device("cpu")#"cuda" if ch.cuda.is_available() else "cpu")
    model = model.to(device)

    #optimizer and criterion taken from utils
    optimizer = ch.optim.Adam(
    model.parameters(), lr=lr,
    weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().cuda()


    def accuracy(preds, labels):
        return (preds == labels).mean()

    
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine()
    
    train_loader = DataLoader(CensusSet(loaders[0][0],loaders[0][1]),batch_size=200)
    test_loader = DataLoader(CensusSet(loaders[1][0],loaders[1][1]),batch_size=200)

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")



    import numpy as np
    from opacus.utils.batch_memory_manager import BatchMemoryManager


    def train_opacus(model, train_loader, optimizer, epoch, device):
        model.train()
        criterion = nn.CrossEntropyLoss()

        losses = []
        top1_acc = []
        
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for i, (images, target, _) in enumerate(memory_safe_data_loader):   
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)
                target = target.type(ch.LongTensor)

                # compute output
                output = model(images)
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
        criterion = nn.CrossEntropyLoss()
        losses = []
        top1_acc = []

        with ch.no_grad():
            for images, target,_ in test_loader:
                images = images.to(device)
                target = target.to(device)
                target = target.type(ch.LongTensor)

                output = model(images)
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
    for epoch in tqdm(range(1), desc="Epoch", unit="epoch"):
        train_opacus(model, train_loader, optimizer, epoch + 1, device)

    top1_acc = test_opacus(model, test_loader, device)