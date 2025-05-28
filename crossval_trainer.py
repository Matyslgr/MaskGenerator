##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## trainer
##

import torch
import time
import os
import csv
import numpy as np
from dataset import ImageMaskDataset
from model import MyUNet
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from base_trainer import BaseTrainer
from utils import CrossvalCSVLogger, Timer, plot_folds_histories
from config import MODELS_DIR, CROSSVAL_DIR, RESULTS_DIR, CROSSVAL_RESULTS_FILE_TEMPLATE

class CrossvalTrainer(BaseTrainer):
    def __init__(self, args, model_fn):
        super().__init__(args, model_fn)
        self.logger = CrossvalCSVLogger()

    def train_fold(self, fold, model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs):
        pbar = tqdm(range(num_epochs), desc=f"Training Fold {fold}", unit="epoch", leave=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "dice_score": [],
            "iou_score": [],
        }

        for _ in pbar:
            avg_train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            avg_val_loss, dice_score, iou_score = self.evaluate(model, val_loader, criterion=criterion)

            scheduler.step()
            early_stopping(avg_val_loss, model)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["dice_score"].append(dice_score)
            history["iou_score"].append(iou_score)

            pbar.set_postfix({
                "Train Loss": f"{avg_train_loss:.4f}",
                "Val Loss": f"{avg_val_loss:.4f}",
                "Best": f"{early_stopping.best_loss:.4f}"
            })
            if early_stopping.early_stop:
                break

        model = early_stopping.load_best_model(model)
        return model, history

    def train(self, pairs_path):

        model_dir = os.path.join(MODELS_DIR, CROSSVAL_DIR, self.args.hash).replace("\\", "/")
        os.makedirs(model_dir, exist_ok=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=self.args.seed)

        all_histories = {}

        # dice_scores = []
        # iou_scores = []
        # total_time = 0.0
        for fold, (train_idx, val_idx) in enumerate(kf.split(pairs_path)):
            fold += 1

            train_pairs = pairs_path[train_idx]
            test_pairs = pairs_path[val_idx]

            val_size = int(len(train_pairs) * 0.2)
            val_pairs = train_pairs[:val_size]
            train_pairs = train_pairs[val_size:]

            train_dataset = ImageMaskDataset(
                train_pairs,
                transform=self.transform_manager.get_train_transform(self.args.augmentations, self.args.train_image_size)
            )
            val_dataset = ImageMaskDataset(
                val_pairs,
                transform=self.transform_manager.get_val_transform()
            )
            test_dataset = ImageMaskDataset(
                test_pairs,
                transform=self.transform_manager.get_test_transform()
            )

            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            model = self.create_model(self.args)

            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

            model_name = self.args.model_name_template.format(hash=self.args.hash, seed=self.args.seed, fold=fold)

            model_path = os.path.join(model_dir, model_name).replace("\\", "/")

            if os.path.exists(model_path):
                print(f"Model {model_path} already exists. Skipping fold {fold}.")
                continue
            early_stopping = EarlyStopping(model_path=model_path, patience=self.args.patience, delta=self.args.delta, verbose=self.args.verbose)

            with Timer() as timer:
                model, history = self.train_fold(fold, model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, self.args.num_epochs)
                _, dice_score, iou_score = self.evaluate(model, test_loader)

            elapsed_time = timer.elapsed
            # total_time += elapsed_time
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f"Fold {fold} - Dice Score: {dice_score:.4f} - IOU Score: {iou_score:.4f}, Time: {elapsed_time}")

            results_dict = {
                "model_path": model_path,
                "fold": fold,
                "dice_score": dice_score,
                "iou_score": iou_score,
                "elapsed_time": elapsed_time,
            }

            crossval_result_file = CROSSVAL_RESULTS_FILE_TEMPLATE.format(experiment_name=self.args.experiment_name)
            results_path = os.path.join(RESULTS_DIR, crossval_result_file).replace("\\", "/")
            self.logger.log(results_path, results_dict, self.args)

            all_histories[fold] = history

        plot_name = self.args.model_name_template.replace("_fold{fold}", "_folds").replace(".pth", ".png").format(hash=self.args.hash, seed=self.args.seed)
        plot_path = os.path.join(model_dir, plot_name).replace("\\", "/")

        plot_folds_histories(all_histories, plot_path)
