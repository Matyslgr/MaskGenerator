##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## utils
##

import torch.nn as nn

from mask_generator.config import LossConfig
from mask_generator.loss.losses import DiceLoss, FocalLoss

class LossFactory:
    """
    Factory class to create loss functions based on a configuration dictionary.
    """

    _registry = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
    }

    @staticmethod
    def create(loss_config: LossConfig) -> nn.Module:
        """
        Create a loss function based on the provided configuration.
        Args:
            loss_config (LossConfig): Configuration for the loss function.
        Returns:
            nn.Module: The configured loss function.
        """
        assert isinstance(loss_config, LossConfig), "loss_config must be an instance of LossConfig"
        loss_class = LossFactory._registry.get(loss_config.name.lower())

        if loss_class is None:
            raise ValueError(f"Loss function '{loss_config.name}' is not registered.")
        # If the loss function requires parameters, pass them
        if loss_config.params:
            return loss_class(**loss_config.params)
        return loss_class()
