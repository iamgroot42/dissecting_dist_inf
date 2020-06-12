import torch as ch
import utils
import numpy as np
from tqdm import tqdm


# Assuming single flipping point for prediction change for delta-addition to specific neuron's output
def delta_binary_search(model, image, injection_point, n_steps, injection_range, low, high, positive_range=True):
	injection_layer, injection_index = injection_point
	eps = 1e-10

	low  = np.zeros(image.shape[0]) + low
	high = np.zeros(image.shape[0]) + high

	with ch.no_grad():
		# Get intermediate outputs of model (caching)
		(logits, intermediate_outputs), _ = model(image, this_layer_output=injection_layer)
		prediction = ch.argmax(logits, 1)

		# Perform binary search
		for _ in range(n_steps):
			mid = (high + low) / 2

			# Construct delta vector (adding noise to output of specific layer's output)
			delta_vec                     = np.zeros((image.shape[0], injection_range), dtype=np.float32)
			delta_vec[:, injection_index] = mid
			delta_vec                     = ch.from_numpy(delta_vec).view_as(intermediate_outputs).cuda()
			perturbed_input = intermediate_outputs + delta_vec

			# Pass on modified input (intermediate output + noise) to rest of model
			perturbed_pred, _ = model(perturbed_input, this_layer_input=injection_layer + 1)
			perturbed_pred    = ch.argmax(perturbed_pred, 1)
			unchanged_indices = (perturbed_pred == prediction).cpu().numpy()

			# Proceed with binary search based on predictions
			if positive_range:
				low[unchanged_indices]   = mid[unchanged_indices]  + eps
				high[~unchanged_indices] = mid[~unchanged_indices] - eps
			else:
				high[unchanged_indices] = mid[unchanged_indices]   - eps
				low[~unchanged_indices] = mid[~unchanged_indices]  + eps
		
	return mid


def get_sensitivities(model, data_loader, injection_layer, injection_range, n_steps):
	sensitivities = {}
	upper_caps = (-1e10, 1e10)
	# Get batches of data
	for (im, label) in data_loader:
		for i in tqdm(range(injection_range)):
			# Delta value when adding
			delta_value = delta_binary_search(model, im.cuda(), (injection_layer, i), n_steps, injection_range, 0, upper_caps[1])
			# Replace maximum range with INF
			delta_value[delta_value == upper_caps[1]] = np.inf
			negative_delta_value = delta_binary_search(model, im.cuda(), (injection_layer, i), n_steps, injection_range, upper_caps[0], 0, positive_range=False)
			# Replace minimum range with -INF
			negative_delta_value[negative_delta_value == upper_caps[0]] = -np.inf
			delta_value = list(np.where(delta_value > abs(negative_delta_value), negative_delta_value, delta_value))
			sensitivities[i] = sensitivities.get(i, []) + delta_value
	return sensitivities


if __name__ == "__main__":
	import sys
	filename         = sys.argv[1]
	model_arch       = sys.argv[2]
	model_type       = sys.argv[3]
	dataset          = sys.argv[4]
	injection_layer  = int(sys.argv[5])

	if dataset == "cifar":
		dx = utils.CIFAR10()
	elif dataset == "imagenet":
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	batch_size = 10000
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	# VGG-19 specific parameters
	injection_range = 512 * 2 * 2
	n_steps = 100
	sensitivities = get_sensitivities(model, test_loader, injection_layer, injection_range, n_steps)

	with open("%s.txt" % filename, 'w') as f:
		for i in range(injection_range):
			floats_to_string = ",".join([str(x) for x in sensitivities[i]])
			f.write(floats_to_string + "\n")
