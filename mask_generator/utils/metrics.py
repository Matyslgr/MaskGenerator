##
## EPITECH PROJECT, 2025
## root [SSH: pinkcc]
## File description:
## metrics
##

import torch

def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.view(-1).cpu()
    targets = targets.view(-1).cpu()

    TP = ((preds == 1) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }
