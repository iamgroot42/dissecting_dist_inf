
import torch as ch
import numpy as np

from distribution_inference.config import WhiteBoxAttackConfig


class Attack:
    def __init__(self, config: WhiteBoxAttackConfig):
        self.config = config

    def _prepare_labels(self, X_1, X_2):
        Y = [0.] * len(X_1) + [1.] * len(X_2)
        Y = ch.from_numpy(np.array(Y)).cuda()
        return Y

    def prepare_attack(self):
        """
            Involves training meta-classifier, etc.
            After this method, attack should be ready to use
        """
        raise NotImplementedError("Must be implemented in subclass")

    def attack(self, X, Y):
        """
            Given model (features or in some form) and labels,
            return attack's predictions
        """
        raise NotImplementedError("Must be implemented in subclass")

    def _get_train_val_shuffle(self, X_1):
        train_sample = self.config.train_sample
        val_sample = self.config.val_sample

        # Create shuffles from both sets of models
        shuffled_indicec = np.random.permutation(len(X_1))

        # Extract model features corresponding to shuffled indices
        vecs_train = X_1[shuffled_indicec[:train_sample]]

        # If validation data requested
        vecs_val = None
        if val_sample > 0:
            vecs_val = X_1[
                shuffled_indicec[
                    train_sample:train_sample+val_sample]]

        # Ready train, val data
        return vecs_train, vecs_val
