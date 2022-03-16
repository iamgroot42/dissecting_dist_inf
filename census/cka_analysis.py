"""
    Use CKA-based methods to identify "redundant" layers inside the model.
    Can also be extended to align representations between models with same (or even different) architectures.
"""

from torch_cka import CKA_sklearn
from torch_cka.utils import plot_results
import argparse
import os
import utils
from tqdm import tqdm
from model_utils import get_models, BASE_MODELS_DIR
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        default="sex",
                        choices=SUPPORTED_PROPERTIES)
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

    # Prepare data wrappers
    ds = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.ratio), split="adv")

    # Fetch test data from both ratios
    _, (data, _), _ = ds.load_data(custom_limit=10000)
    # Use first 200 samples for testing
    data = data[:1000]

    results = None
    for i in tqdm(range(len(models))):
        model = models[i]
        cka = CKA_sklearn(
                  model, model,
                  model1_name="First Model",
                  model2_name="Second Model")

        result = cka.compare(data, verbose=False)

        # Sum up CKA matrix
        if results is None:
            results = result
        else:
            results["CKA"] += result["CKA"]

    # Average CKA matrix
    results["CKA"] /= len(models)

    plot_results(results, save_path="./plots/cka-experiment.png",
                 title=f"Average CKA across {args.n_models} models")
