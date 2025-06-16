##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## losses
##

import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss between the predicted inputs and the target masks.
        Args:
            inputs (torch.Tensor): Predicted masks, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, C, H, W).
        Returns:
            torch.Tensor: Computed Dice loss.
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

class CompositeLoss(nn.Module):
    def __init__(self, losses_with_weights: List[Tuple[str, nn.Module, float]]):
        """
        Initialize the composite loss with a list of individual losses and their weights.
        Args:
            losses_with_weights (List[Tuple[str, nn.Module, float]]): List of tuples containing loss name, loss function, and weight.
        """
        super(CompositeLoss, self).__init__()
        self.losses = nn.ModuleDict(
            {name: loss for name, loss, _ in losses_with_weights}
        )
        self.weights = {
            name: weight for name, _, weight in losses_with_weights
        }

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the composite loss by summing individual losses weighted by their respective weights.
        Args:
            inputs (torch.Tensor): Predicted masks, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, C, H, W).
        Returns:
            torch.Tensor: Computed composite loss.
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            weight = self.weights[name]
            loss_value = loss_fn(inputs, targets)
            weighted_loss = weight * loss_value
            total_loss += weighted_loss
            loss_dict[name] = weighted_loss.item()

        return total_loss, loss_dict
