from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information


# di = get_dataset_information("texas")()
# di.generate_victim_adversary_splits(num_tries=5)
ds = get_dataset_wrapper("texas")
x = ds(None)
z = x.ds.get_data("adv", 0.5, "sex")
print("Done")
# k = 5 + 2