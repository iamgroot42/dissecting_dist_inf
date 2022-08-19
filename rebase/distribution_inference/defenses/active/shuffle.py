from ...config.core import TrainConfig
import torch as ch
import warnings
import numpy as np
from distribution_inference.config import ShuffleDefenseConfig, TrainConfig
from distribution_inference.utils import warning_string


class ShuffleDefense:
    def __init__(self, config: ShuffleDefenseConfig):
        self.config = config
        self.desired_value = self.config.desired_value
        self.sample_ratio = self.config.sample_ratio
    
    def initialize(self, ds, train_config: TrainConfig):
        if self.config.data_level is False:
            # Nothing to do here- whatever happens will be at batch level
            return

        # Get loaders
        train_loader, _ = ds.get_loaders(
            batch_size=train_config.batch_size,
            shuffle=False)
        # Get mask
        selected_mask = self._get_mask_from_loader(train_loader)
        # Update data-wrapper
        ds.mask_data_selection(selected_mask)
    
    def _get_mask_from_loader(self, loader):
        """
            Collect data across all of the loader, process
            each batch with process function, and return
            version of loader with desired data.
        """
        all_labels = []
        for _, _, prop_labels in loader:
            all_labels.append(prop_labels)
        all_labels = ch.cat(all_labels)
        selected_indices = self.process(all_labels)
        return selected_indices
    
    def process_batch(self, batch):
        if self.config.data_level:
            # Already shuffled at data level- return s it is
            return batch

        x, y, prop_labels = batch
        selected_indices = self.process(prop_labels)
        x_ = x[selected_indices]
        y_ = y[selected_indices]
        prop_labels_ = prop_labels[selected_indices]
        return x_, y_, prop_labels_

    def process(self, prop_labels):
        """
            Process data and return copy of data
            that should be used instead.
        """
        final_samples = int(len(prop_labels) * self.sample_ratio)
        final_samples_one = int(final_samples * self.desired_value)
        final_samples_zero = final_samples - final_samples_one

        # Sample to achieve desired ratio
        one_label = ch.nonzero(prop_labels).squeeze(1)
        zero_label = ch.nonzero(1 - prop_labels).squeeze(1)
        oversampling = False
        # Bad batch- all prop labels same. Just let this one be
        if len(one_label) == 0 or len(zero_label) == 0:
            raise ValueError("All property labels are the same")

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
            num_sample_zero = int((self.desired_value) /
                                  self.desired_value * num_sample_one)
        else:
            # If not enough for both, then oversampling
            oversampling = True
            num_sample_one = final_samples_one
            num_sample_zero = final_samples_zero

        if oversampling:
            one_sample = np.random.choice(one_label, size=num_sample_one)
            zero_sample = np.random.choice(zero_label, size=num_sample_zero)
            sample_index = ch.cat(
                [ch.from_numpy(one_sample), ch.from_numpy(zero_sample)])
        else:
            one_sample = ch.randperm(len(one_label))[:num_sample_one]
            zero_sample = ch.randperm(len(zero_label))[:num_sample_zero]
            sample_index = ch.cat(
                [one_label[one_sample], zero_label[zero_sample]])
            
        return sample_index
