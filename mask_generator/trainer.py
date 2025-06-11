##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## trainer
##

import torch
import time
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import torch.nn as nn

from mask_generator.earlystopping import EarlyStopping
from mask_generator.dataset import ImageMaskDataset
from mask_generator.transforms import AlbumentationsTrainTransform, KorniaInferTransform
from mask_generator.utils import Timer
from mask_generator.config import Config
import mask_generator.settings as settings
from mask_generator.utils import TrainingLogger

def compute_pos_weight(loader):
    total_pos = 0
    total_neg = 0
    for _, masks in loader:
        masks = masks.view(-1)
        total_pos += masks.sum().item()
        total_neg += (1 - masks).sum().item()

    print(f"Total positive pixels: {total_pos}, Total negative pixels: {total_neg}")
    return torch.tensor(total_neg / total_pos)

class Trainer():
    def __init__(self, config: Config, pad_divisor: int):
        self.config = config
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
        print(f"Using device: {self.device}")
        self.dice_metric = DiceScore(num_classes=2, average='macro').to(self.device)
        self.iou_metric = BinaryJaccardIndex(threshold=0.5).to(self.device)
        self.train_transform = AlbumentationsTrainTransform(
            seed=config.training.seed,
            pad_divisor=pad_divisor,
            image_size=config.training.train_image_size,
            augmentations_names=config.training.augmentations,
        )
        self.infer_transform = KorniaInferTransform(
            pad_divisor=pad_divisor,
            device=self.device_str
        )

        self.logger = TrainingLogger(config.other.run_dir)

        if config.training.use_amp:
            self.scaler = GradScaler()

    def _prepare_loaders(self, train_pairs, val_pairs, test_pairs):

        train_ds = ImageMaskDataset(train_pairs, transform=self.train_transform)
        val_ds = ImageMaskDataset(val_pairs, transform=self.infer_transform)
        test_ds = ImageMaskDataset(test_pairs, transform=self.infer_transform)

        kwargs = {
            "batch_size": self.config.training.batch_size,
            "num_workers": min(8, os.cpu_count() // 2),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "pin_memory_device": self.device_str if self.device_str == 'cuda' else ""
        }

        return (
            DataLoader(train_ds, shuffle=True, **kwargs),
            DataLoader(val_ds, shuffle=False, **kwargs),
            DataLoader(test_ds, shuffle=False, **kwargs)
        )

    def _train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        self.dice_metric.reset()
        self.iou_metric.reset()
        total_loss = 0.0

        for images, masks in tqdm(loader, desc="  Batch", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)

            optimizer.zero_grad()

            if self.config.training.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                # Use mixed precision training
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            self.dice_metric(preds, masks)
            self.iou_metric(preds, masks)

        return {
            "loss": total_loss / len(loader),
            "dice": self.dice_metric.compute().item(),
            "iou": self.iou_metric.compute().item()
        }

    def _evaluate(self, model: nn.Module, loader, criterion, with_conf_matrix: bool = False):
        model.eval()
        self.dice_metric.reset()
        self.iou_metric.reset()
        total_loss = 0.0

        if with_conf_matrix:
            all_preds = []
            all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(loader, desc="  Eval", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = model(images)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                self.dice_metric(preds, masks)
                self.iou_metric(preds, masks)

                loss = criterion(outputs, masks)
                total_loss += loss.item()

                if with_conf_matrix:
                    all_preds.append(preds.cpu().numpy().flatten())
                    all_targets.append(masks.cpu().numpy().flatten())

        result = {
            "loss": total_loss / len(loader),
            "dice": self.dice_metric.compute().item(),
            "iou": self.iou_metric.compute().item(),
        }
        if with_conf_matrix:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            cm = confusion_matrix(all_targets, all_preds)
            result["conf_matrix"] = cm

        return result

    def fit(self, model: nn.Module, train_pairs, test_pairs):
        model = model.to(self.device)
        val_size = int(len(test_pairs) * 0.2)
        val_pairs = test_pairs[:val_size]
        test_pairs = test_pairs[val_size:]

        train_loader, val_loader, test_loader = self._prepare_loaders(train_pairs, val_pairs, test_pairs)

        if self.config.training.weighted_loss:
            pos_weight = torch.tensor(31.568326950073242).to(self.device)
            # pos_weight = compute_pos_weight(train_loader).to(self.device)
            # print(f"Computed positive weight: {pos_weight.item()}")
        else:
            pos_weight = None
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.config.training.step_size,
            gamma=self.config.training.gamma
        )

        model_path = os.path.join(self.config.other.run_dir, settings.model_filename)
        if os.path.exists(model_path):
            print(f"Model {model_path} already exists. Skipping training.")
            return None

        early_stopping = EarlyStopping(
            model_path=model_path,
            patience=self.config.training.patience,
            delta=self.config.training.delta,
        )

        pbar = tqdm(range(self.config.training.num_epochs), desc="Train", unit="epoch", leave=True)
        with Timer() as timer:
            for epoch in pbar:
                start = time.time()
                train_metrics = self._train_epoch(model, train_loader, optimizer, criterion)
                val_metrics = self._evaluate(model, val_loader, criterion)
                end = time.time()
                elapsed = end - start

                lr = scheduler.get_last_lr()[0]
                self.logger.log_epoch(epoch, lr, elapsed, train_metrics, val_metrics)

                scheduler.step()
                early_stopping(epoch, val_metrics["loss"], model)

                pbar.set_postfix({
                    "Train Loss": f"{train_metrics['loss']:.4f}",
                    "Val Loss": f"{val_metrics['loss']:.4f}",
                    "Best": f"{early_stopping.best_loss:.4f}",
                })

                if epoch % 5 == 0:
                    self.logger.save_plots()

                if early_stopping.early_stop:
                    break

                torch.cuda.empty_cache()

            model = early_stopping.load_best_model(model)
            test_metrics = self._evaluate(model, test_loader, criterion, with_conf_matrix=True)

        elapsed_time = timer.elapsed
        best_epoch = early_stopping.best_epoch
        self.logger.save_all(
            test_metrics=test_metrics,
            elapsed_time=elapsed_time,
            best_epoch=best_epoch,
        )
