##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## losses
##

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss between the predicted logits and the target masks.
        Args:
            logits (torch.Tensor): Predicted masks, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, C, H, W).
        Returns:
            torch.Tensor: Computed Dice loss.
        """
        inputs = torch.sigmoid(logits)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss adapted for binary classification tasks.

        Args:
            alpha (float): Weighting factor for the class imbalance.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose from 'none', 'mean', or 'sum'.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal loss between the predicted logits and the target masks.
        Args:
            logits (torch.Tensor): Predicted masks, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, C, H, W).
        Returns:
            torch.Tensor: Computed Focal loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)

        # Compte pt: probabilities of the positive class
        pt = probs * targets + (1 - probs) * (1 - targets)

        # Apply the Focal Loss formula
        focal_term = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_term * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class BoundaryLoss(nn.Module):
    """
    Boundary Loss optimisée pour segmentation binaire (0: fond, 1: ligne).
    Basé sur : Boundary Loss for Remote Sensing Imagery Semantic Segmentation.
    """
    def __init__(self, theta0=3, theta=5):
        """
        Args:
            theta0 (int): Taille du noyau pour détecter les contours primaires.
            theta (int): Taille du noyau pour l'élargissement des bords.
        """
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred_logits, gt_mask, return_maps=False):
        """
        Args:
            pred_logits (torch.Tensor): Logits du modèle (N, 1, H, W).
            gt_mask (torch.Tensor): Masque binaire ground truth (N, H, W) avec 0 ou 1.

        Returns:
            torch.Tensor: Boundary loss moyenne.
        """
        assert pred_logits.shape[1] == 1, "BoundaryLoss : prédictions attendues avec 1 canal"

        pred = torch.sigmoid(pred_logits)
        pred = torch.cat([1 - pred, pred], dim=1)  # Shape : (N, 2, H, W)

        assert pred.shape[1] == 2, "BoundaryLoss : pred must have 2 channels (0: background, 1: line)"

        gt_one_hot = torch.stack([(gt_mask == 0).float(), (gt_mask == 1).float()], dim=1)  # (N, 2, H, W)

        # Détection des bords
        gt_b = F.max_pool2d(1 - gt_one_hot, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt_one_hot

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # Bords élargis
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        if return_maps:
            return gt_b, pred_b, gt_b_ext, pred_b_ext

        # Flatten pour simplifier le calcul
        gt_b = gt_b.view(gt_b.shape[0], 2, -1)
        pred_b = pred_b.view(pred_b.shape[0], 2, -1)
        gt_b_ext = gt_b_ext.view(gt_b_ext.shape[0], 2, -1)
        pred_b_ext = pred_b_ext.view(pred_b_ext.shape[0], 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        BF1 = 2 * P * R / (P + R + 1e-7)
        loss = torch.mean(1 - BF1[:, 1])

        return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    return scores[valid], labels[valid]

def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    return scores[valid], labels[valid]

def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels, per_image=True, ignore_index=None):
    """
    logits : [B, H, W] Logits non activés (avant Sigmoid)
    labels : [B, H, W] Masques binaires (0 ou 1)
    """
    if per_image:
        loss = torch.mean(torch.stack([lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore_index))
                                       for log, lab in zip(logits, labels)]))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore_index))
    return loss

class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for binary segmentation tasks.
    This loss is designed to handle class imbalance and is particularly useful for segmentation tasks.
    """
    def __init__(self, per_image=True, ignore_index=None):
        """
        Args:
            per_image (bool): If True, computes the loss per image in the batch.
            ignore_index (int or None): Class label to ignore during loss computation.
        """
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lovasz Hinge loss.
        Args:
            logits (torch.Tensor): Predicted logits, shape (N, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, H, W).
        Returns:
            torch.Tensor: Computed Lovasz Hinge loss.
        """
        return lovasz_hinge(logits, targets, self.per_image, self.ignore)

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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the composite loss by summing individual losses weighted by their respective weights.
        Args:
            logits (torch.Tensor): Predicted masks, shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks, shape (N, C, H, W).
        Returns:
            torch.Tensor: Computed composite loss.
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            weight = self.weights[name]
            loss_value = loss_fn(logits, targets)
            weighted_loss = weight * loss_value
            total_loss += weighted_loss
            loss_dict[name] = weighted_loss.item()

        return total_loss, loss_dict
