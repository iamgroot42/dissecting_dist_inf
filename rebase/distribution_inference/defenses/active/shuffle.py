import torch as ch
import numpy as np
from tqdm import tqdm
from distribution_inference.config import ShuffleDefenseConfig


class ShuffleDefense:
    def __init__(self, config: ShuffleDefenseConfig):
        self.config = config
        self.desired_value = self.config.desired_value
        self.sample_type = self.config.sample_type
    
    def initialize(self, train_loader):
        if self.config.data_level is False:
            # Nothing to do here- whatever happens will be at batch level
            return

        # Get mask
        selected_mask = self._get_mask_from_loader(train_loader)
        return selected_mask, None
    
    def _get_mask_from_loader(self, loader):
        """
            Collect data across all of the loader, process
            each batch with process function, and return
            version of loader with desired data.
        """
        all_labels = []
        for _, _, prop_labels in tqdm(loader, desc="Collecting data for ShuffleDefense"):
            all_labels.append(prop_labels)
        all_labels = ch.cat(all_labels)
        selected_indices = self.process(all_labels)
        return selected_indices
    
    def process_batch(self, batch):
        if self.config.data_level:
            # Already shuffled at data level- return as it is
            return batch

        x, y, prop_labels = batch
        selected_indices = self.process(prop_labels)
        x_ = x[selected_indices]
        y_ = y[selected_indices]
        prop_labels_ = prop_labels[selected_indices]
        return x_, y_, prop_labels_
    
    def _determine_one_zero_num_req(self, prop_labels):
        """
            Determine how many 0/1 labeled (prop) data
            should be there in batch to achieve
            desired ratio, and what kind of mode
            it would correspond to
        """
        num_one_label = ch.sum(prop_labels)
        num_zero_label = len(prop_labels) - num_one_label
        # Bad batch- all prop labels same. Just let this one be
        if num_one_label == 0 or num_zero_label == 0:
            raise ValueError("All property labels are the same")

        current_ratio = num_one_label / len(prop_labels)
        if self.sample_type == "over":
            oversampling = True
            # Over-sampling
            if self.desired_value > current_ratio:
                # Need to have more 1s than 0s
                # Over-sample from 1 class
                num_sample_zero = num_zero_label # Keep same
                num_sample_one = int((self.desired_value * num_sample_zero) / (1 - self.desired_value)) # Over-sample
            else:
                # Need to have more 0s than 1s
                # Over-sample from 0 class
                num_sample_one = num_one_label # Keep same
                num_sample_zero = int((self.desired_value * num_sample_one) / (1 - self.desired_value)) # Over-sample
        else:
            oversampling = False
            # Under-sampling
            if self.desired_value > current_ratio:
                # Need to have more 1s than 0s
                # Under-sample from 0 class
                num_sample_one = num_one_label
                num_sample_zero = int((self.desired_value * num_sample_one) / (1 - self.desired_value)) # Under-sample
            else:
                # Need to have more 0s than 1s
                # Under-sample from 1 class
                num_sample_zero = num_zero_label
                num_sample_one = int((self.desired_value * num_sample_zero) / (1 - self.desired_value)) # Under-sample
        
        return int(num_sample_zero), int(num_sample_one), oversampling

    def process(self, prop_labels):
        """
            Process data and return copy of data
            that should be used instead.
        """
        # Sample to achieve desired ratio
        num_sample_zero, num_sample_one, oversampling = self._determine_one_zero_num_req(prop_labels)
        one_label = ch.nonzero(prop_labels).squeeze(1)
        zero_label = ch.nonzero(1 - prop_labels).squeeze(1)

        oversample_zero = oversampling
        oversample_one = oversampling
        # if oversampling:
        #     # Need to oversample from only one of them
        #     if num_sample_one == len(one_label):
        #         oversample_one = False
        #     elif num_sample_zero == len(zero_label):
        #         oversample_zero = False
        #     else:
        #         raise ValueError("Should not be here")

        one_sample = np.random.choice(
            one_label, size=num_sample_one,
            replace=oversample_one)
        zero_sample = np.random.choice(
            zero_label, size=num_sample_zero,
            replace=oversample_zero)
        sample_index = ch.cat(
                [ch.from_numpy(one_sample), ch.from_numpy(zero_sample)])
            
        return sample_index
