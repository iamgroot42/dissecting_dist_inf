from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import DatasetConfig

# di = get_dataset_information("texas")()
# di.generate_victim_adversary_splits(num_tries=500, split_on_hospitals=False)
# ds = get_dataset_wrapper("texas")
# x = ds(None)
# z = x.ds.get_data("victim", 1.0, "ethnicity")

di = get_dataset_information("celeba")()
di._extract_pretrained_features()

# ds = get_dataset_wrapper("celeba")
# config = DatasetConfig(name="celeba", prop="Male", value=0.5, split="adv", classify="Smiling", processed_variant=True)
# x = ds(config)
# _ , loader = x.get_loaders(batch_size=512)
# x, y, z = next(iter(loader))
# print(x.shape)