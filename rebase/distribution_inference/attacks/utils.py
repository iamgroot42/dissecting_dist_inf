from dataclasses import replace

from distribution_inference.config import DatasetConfig


def get_dfs_for_victim_and_adv(base_data_config: DatasetConfig):
    """
        Starting from given base data configuration, make two copies.
        One with the split as 'adv', the other as 'victim'
    """
    config_adv = replace(base_data_config)
    config_adv.split = "adv"
    config_victim = replace(base_data_config)
    config_victim.split = "victim"
    return config_adv, config_victim
