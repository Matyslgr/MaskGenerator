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

def set_deterministic_behavior(seed):
    """
    Function to set deterministic behavior for reproducibility.

    Args:
        seed (int): The seed value for random number generators.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
