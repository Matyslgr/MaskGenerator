##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## base_trainer
##

import torch
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from model import MyUNet

class BaseTrainer:
    def __init__(self, args: dict, model_fn):
        self.args = args
        self.model_fn = model_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.dice_metric = DiceScore(num_classes=2).to(self.device)
        self.iou_metric = BinaryJaccardIndex(threshold=0.5).to(self.device)

    def create_model(self, args: dict):
        return self.model_fn(args).to(self.device)

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(self.device), masks.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, model, data_loader, criterion=None):
        model.eval()

        # Reset metrics
        self.dice_metric.reset()
        self.iou_metric.reset()

        total_loss = 0.0

        with torch.no_grad():
            for images, masks in data_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()

                self.dice_metric(outputs, masks)
                self.iou_metric(outputs, masks)

                if criterion:
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()

        dice_score = self.dice_metric.compute().item()
        iou_score = self.iou_metric.compute().item()
        avg_loss = total_loss / len(data_loader) if criterion else None

        return avg_loss, dice_score, iou_score
