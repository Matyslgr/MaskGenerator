##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## train
##

import os
import torch
import argparse
import numpy as np
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils import set_deterministic_behavior, get_all_pairs_path
from dataset import ImageMaskDataset
from model import MyUNet
from earlystopping import EarlyStopping

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def train_model(fold, model, train_loader, val_loader, optimizer, criterion, scheduler, early_stopping, num_epochs, verbose):

    for epoch in tqdm(range(num_epochs), desc=f"Training Fold {fold}", unit="epoch"):

        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model = early_stopping.load_best_model(model)
    return model

def evaluate_model(model: MyUNet, test_loader: DataLoader, dice_metric: DiceScore, iou_metric: BinaryJaccardIndex, verbose: bool):
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()

            # Calculate metrics
            dice_metric.update(outputs, labels)
            iou_metric.update(outputs, labels)

    dice_score = dice_metric.compute()
    iou_score = iou_metric.compute()

    print(f"Dice Score: {dice_score:.4f}")
    print(f"IOU Score: {iou_score:.4f}")
    return dice_score, iou_score

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train the model.')
    parser.add_argument('--model_path', type=str, required=True, help='The path to save the model')
    parser.add_argument('--dataset_root', type=str, default="Dataset", help='The root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs for training')
    parser.add_argument('--num_seeds', type=int, default=10, help='The number of seeds to use for training')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate for the optimizer')
    parser.add_argument('--step_size', type=int, default=10, help='The step size for the learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='The gamma value for the learning rate scheduler')
    parser.add_argument('--patience', type=int, default=30, help='The patience value for early stopping')
    parser.add_argument('--delta', type=float, default=0.0, help='The delta value for early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='The verbosity level for training')
    return parser.parse_args()

def main():
    args = parse_args()

    seeds = [42 + i for i in range(args.num_seeds)]

    pairs_path = get_all_pairs_path(args.dataset_root)

    pairs_path = pairs_path[:100]  # Limit to 100 pairs for testing

    print(f"Total number of pairs: {len(pairs_path)}")
    dice_metric = DiceScore(num_classes=1, average='macro').to(DEVICE)
    iou_metric = BinaryJaccardIndex().to(DEVICE)

    for seed in seeds:
        set_deterministic_behavior(seed)

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(kf.split(pairs_path)):
            train_pairs = pairs_path[train_index]
            test_pairs = pairs_path[test_index]

            val_size = int(len(train_pairs) * 0.2)
            val_pairs = train_pairs[:val_size]
            train_pairs = train_pairs[val_size:]

            train_dataset = ImageMaskDataset(train_pairs)
            val_dataset = ImageMaskDataset(val_pairs)
            test_dataset = ImageMaskDataset(test_pairs)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            model = MyUNet(in_channels=3, out_channels=1).to(DEVICE)

            criterion = torch.nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

            early_stopping = EarlyStopping(model_path=args.model_path, patience=args.patience, delta=args.delta, verbose=args.verbose)

            model = train_model(fold, model, train_loader, val_loader, optimizer, criterion, scheduler, early_stopping, args.num_epochs, args.verbose)

            dice_score, iou_score = evaluate_model(model, test_loader, dice_metric, iou_metric, args.verbose)

            print(f"Fold {fold} - Seed {seed} - Dice Score: {dice_score:.4f} - IOU Score: {iou_score:.4f}")

            # Reset metrics for the next fold
            dice_metric.reset()
            iou_metric.reset()


if __name__ == "__main__":
    main()
