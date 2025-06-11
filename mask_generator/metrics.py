##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## epoch_metrics
##

from dataclasses import dataclass

@dataclass
class Metrics:
    loss: float
    dice: float
    iou: float
    acc: float
    precision: float
    recall: float
    f1: float
