import torch as ch
import warnings
import numpy as np
from distribution_inference.config import ShuffleDefenseConfig
from distribution_inference.utils import warning_string


class ShuffleDefense:
    def __init__(self, config: ShuffleDefenseConfig):
        self.config = config
        self.desired_value = self.config.desired_value
        self.sample_ratio = self.config.sample_ratio

    def process(self, data, task_labels, prop_labels):
        """
            Process data and return copy of data
            that should be used instead.
        """
        final_samples = int(len(data) * self.sample_ratio)
        final_samples_one = int(final_samples * self.desired_value)
        final_samples_zero = final_samples - final_samples_one

        # Sample to achieve desired ratio
        one_label = ch.nonzero(prop_labels).squeeze(1)
        zero_label = ch.nonzero(1 - prop_labels).squeeze(1)
        oversampling = False
        # Bad batch- all prop labels same. Just let this one be
        if len(one_label) == 0 or len(zero_label) == 0:
            return data, task_labels, prop_labels

        if ch.sum(prop_labels) < final_samples_one:
            warnings.warn(warning_string(
                "Fewer samples available than requested"))

        if len(one_label) > final_samples_one and len(zero_label) > final_samples_zero:
            # Enough to sample for both
            num_sample_one = final_samples_one
            num_sample_zero = final_samples_zero
        elif len(one_label) > final_samples_one:
            # Enough to sample for one, but not zero
            num_sample_zero = len(zero_label)
            num_sample_one = int((self.desired_value) /
                                 (1 - self.desired_value) * num_sample_zero)
        elif len(zero_label) > final_samples_zero:
            # Enough to sample for zero, but not one
            num_sample_one = len(one_label)
            num_sample_zero = int((self.desired_value) / self.desired_value * num_sample_one)
        else:
            # If not enough for both, then oversampling
            oversampling = True
            num_sample_one = len(one_label)
            num_sample_zero = len(zero_label)

        if oversampling:
            one_sample = np.random.choice(one_label,size=num_sample_one)
            zero_sample = np.random.choice(zero_label,size=num_sample_zero)
            sample_index = ch.cat([ch.from_numpy(one_sample),ch.from_numpy(zero_sample)])
        else:
            one_sample = ch.randperm(len(one_label))[:num_sample_one]
            zero_sample = ch.randperm(len(zero_label))[:num_sample_zero]
            sample_index = ch.cat([one_label[one_sample], zero_label[zero_sample]])

        data_ = data[sample_index]
        task_labels_ = task_labels[sample_index]
        prop_labels_ = prop_labels[sample_index]

        return data_, task_labels_, prop_labels_
