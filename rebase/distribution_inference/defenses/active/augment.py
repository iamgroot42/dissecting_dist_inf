import torch as ch
import numpy as np
from torchvision import transforms
from distribution_inference.config import ShuffleDefenseConfig
from distribution_inference.defenses.active.shuffle import ShuffleDefense
from torch.distributions.beta import Beta
from tqdm import tqdm
from itertools import product


class AugmentDefense(ShuffleDefense):
    def __init__(self, config: ShuffleDefenseConfig):
        super().__init__(config)
        self.sample_type = "over"
    
    def initialize(self, train_loader):
        if self.config.data_level is False:
            # Nothing to do here- whatever happens will be at batch level
            return

        # Get extra data
        extra_data = self._get_data_from_loader(train_loader)
        return extra_data

    def _get_data_from_loader(self, loader):
        """
            Collect data across all of the loader, process
            each batch with process function, and return
            version of loader with desired data.
        """
        all_x, all_y, all_labels = [], [], []
        for x, y, prop_labels in tqdm(loader, desc="Collecting data for AugmentDefense"):
            all_x.append(x)
            all_y.append(y)
            all_labels.append(prop_labels)
        alL_x = ch.cat(all_x)
        all_y = ch.cat(all_y)
        all_labels = ch.cat(all_labels)
        processed_data = self.process((alL_x, all_y, all_labels))
        return processed_data
    
    def process_batch(self, batch):
        if self.config.data_level:
            # Already augmented at data level- return as it is
            return batch

        return self.process(batch)

    def process(self, data):
        """
            Process data and return copy of data
            that should be used instead.
        """
        x, y, prop_labels = data

        # Sample to achieve desired ratio
        num_sample_zero, num_sample_one, oversampling = self._determine_one_zero_num_req(prop_labels)
        one_label = ch.nonzero(prop_labels).squeeze(1)
        zero_label = ch.nonzero(1 - prop_labels).squeeze(1)
        assert oversampling is True, "This should be oversampling"
        
        if num_sample_one > len(one_label):
            # Oversample for label=1 property data
            num_oversample = num_sample_one - len(one_label)
            # Pick random data from one-label class 
            one_sample_idx = np.random.choice(
                one_label, size=num_oversample,
                replace=True)
            # Get augmented data
            data_extra = self.augment(
                (x[one_sample_idx],
                y[one_sample_idx],
                prop_labels[one_sample_idx]))
        elif num_sample_zero > len(zero_label):
            # Oversample for label=0 property data
            num_oversample = num_sample_zero - len(zero_label)
            zero_sample_idx = np.random.choice(
                zero_label, size=num_oversample,
                replace=True)
            # Get augmented data
            data_extra = self.augment(
                (x[zero_sample_idx],
                y[zero_sample_idx],
                prop_labels[zero_sample_idx]))
        else:
            raise AssertionError("This should not happen")
                
        # Return extra data
        return data_extra

    def augment(self, data):
        """
            Return augmented version of samples of X,
            based on random selection of augmentations for 
            every call.
        """
        x, y, prop_labels = data

        if self.config.use_mixup:
            raise ValueError("Mixup not implemented yet")

        # Transform data back to (0, 1) range from (-1, 1)
        x_ = (x.clone() + 1) / 2
        # Data not on GPU at this stage
        augment_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip()
        ])

        # Apply random transform per image
        # Clip data to [0, 1] (should  be already)
        x_ = ch.clamp(x_, 0, 1)
        x_ = transforms.Lambda(lambda x: ch.stack([augment_transforms(x_) for x_ in x]))(x)
        # Transform back to (-1, 1) range
        x_ = 2 * x_ - 1

        return (x_, y, prop_labels)

    def _mixup_data(self, X, Y, num_samples):
        # TODO: Mixup itself should not change prop_labels distribution
        # Select examples from y=0, y=1 class
        zero_idx = ch.nonzero(Y).squeeze(1).numpy()
        one_idx = ch.nonzero(1 - Y).squeeze(1).numpy()
        # Pick random pairs of indices from zero_idx and one_idx
        # We do not want duplicates pairs
        all_pairs = np.array(list(product(zero_idx, one_idx)))
        random_pairs = np.random.choice(all_pairs, size=num_samples, replace=True)
        return self._mixup_data(X[random_pairs[:, 0]], X[random_pairs[:, 1]])

    def _mixup_datum(self, data_0, data_1, alpha: float = 1.0):
        """
            Mixup augmentation
        """
        b = Beta(alpha, alpha)
        l = b.sample()
        x_0, y_0 = data_0
        x_1, y_1 = data_1
        x_combined = l * x_0 + (1 - l) * x_1
        y_combined = l * y_0 + (1 - l) * y_1
        return x_combined, y_combined
