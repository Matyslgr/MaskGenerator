##
## EPITECH PROJECT, 2025
## root [SSH: pinkcc]
## File description:
## utils
##

import os
import random
import torch
import numpy as np
from omegaconf import OmegaConf

from mask_generator.config import Config

def set_deterministic_behavior(seed):
    """
    Function to set deterministic behavior for reproducibility.

    Args:
        seed (int): The seed value for random number generators.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)

def load_config_from_yaml(yaml_file: str) -> Config | None:
    try:
        run_cfg = OmegaConf.load(yaml_file)

        default_cfg = OmegaConf.structured(Config)

        cfg = OmegaConf.merge(default_cfg, run_cfg)
        OmegaConf.resolve(cfg)

        config: Config = OmegaConf.to_object(cfg)

    except Exception as e:
        return None
    return config