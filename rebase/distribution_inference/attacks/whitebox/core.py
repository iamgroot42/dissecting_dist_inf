from distribution_inference.config import WhiteBoxAttackConfig, DatasetConfig


class Attack:
    def __init__(self,
                 config: WhiteBoxAttackConfig):
        self.config = config
        self.trained_model = False

    def _prepare_model(self):
        """
            Define and prepare model for attack.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def execute_attack(self, train_data, test_data, val_data=None, **kwargs):
        """
            Involves training meta-classifier, etc.
            After this method, attack should be ready to use.
            Return model performance (accuacy) on test data.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def save_model(self,
                   data_config: DatasetConfig,
                   attack_specific_info_string: str):
        """
            Save model to disk.
        """
        raise NotImplementedError("Must be implemented in subclass")
