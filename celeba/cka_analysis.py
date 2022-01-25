"""
    Use CKA-based methods to identify "redundant" layers inside the model.
    Can also be extended to align representations between models with same (or even different) architectures.
"""

from torch_cka import CKA
from torch_cka.utils import plot_results
import torch as ch
import argparse
import os
import utils
from tqdm import tqdm
from model_utils import get_models, BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        default="Male",
                        choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--task', default="Smiling",
                        choices=SUPPORTED_PROPERTIES,
                        help='task to focus on')
    parser.add_argument('--ratio', help="ratio to focus on", default="0.5")
    parser.add_argument(
        '--n_models', help="number of models to use for cka matrix estimate",
        type=int, default=15)

    args = parser.parse_args()
    utils.flash_utils(args)

    # Pick models at random for preliminary analysis
    model_dir = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.ratio))
    models = get_models(model_dir, args.n_models)

    ds = CelebaWrapper(args.filter, float(
        args.ratio), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)

    dataloader = ds.get_loaders(args.batch_size, shuffle=False)[1]

    results = None
    wanted_layers = [f"features.{i}" for i in [
        2, 5, 7, 9, 11]] + [f"classifier.{i}" for i in [1, 3, 4]]
    for i in tqdm(range(len(models))):
        model = models[i]
        cka = CKA(model, model,
                  model1_name="First Model",
                  model2_name="Second Model",
                  model1_layers=wanted_layers,
                  model2_layers=wanted_layers,
                  device='cuda')

        result = cka.compare(dataloader, verbose=False)

        # Sum up CKA matrix
        if results is None:
            results = result
        else:
            results["CKA"] += result["CKA"]

        # Free up space : HAVE to delete model, else intermediate tensors
        # persist and cause OOM errors
        models[i] = None
        ch.cuda.empty_cache()

    # Average CKA matrix
    results["CKA"] /= len(models)

    plot_results(results, save_path="./plots/cka-experiment.png",
                 title=f"Average CKA across {args.n_models} models")
