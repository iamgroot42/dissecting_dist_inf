
from distribution_inference.config import WhiteBoxAttackConfig


class Attack:
    def __init__(self, config: WhiteBoxAttackConfig):
        self.config = config
