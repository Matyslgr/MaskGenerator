##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## fulltrainer
##

import torch
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from earlystopping import EarlyStopping
from dataset import ImageMaskDataset
from base_trainer import BaseTrainer
from utils import FulltrainCSVLogger, Timer, plot_history
from config import FULLTRAIN_DIR, RESULTS_DIR, MODELS_DIR, FULLTRAIN_RESULTS_FILE_TEMPLATE


class FullTrainer(BaseTrainer):
    def __init__(self, args, model_fn):
        super().__init__(args, model_fn)
        self.logger = FulltrainCSVLogger()

    def train(self, train_pairs_path, test_pairs_path):
        val_size = int(len(test_pairs_path) * 0.2)

        val_pairs_path = test_pairs_path[:val_size]
        test_pairs_path = test_pairs_path[val_size:]

        train_transform = self.transform_manager.get_train_transform(self.args.augmentations, self.args.train_image_size)
        eval_transform = self.transform_manager.get_eval_transform()

        train_dataset = ImageMaskDataset(
            train_pairs_path,
            transform=train_transform
        )
        val_dataset = ImageMaskDataset(
            val_pairs_path,
            transform=eval_transform
        )
        test_dataset = ImageMaskDataset(
            test_pairs_path,
            transform=eval_transform
        )

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = self.create_model(self.args)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        model_name = self.args.model_name_template.format(hash=self.args.hash)
        model_dir = os.path.join(MODELS_DIR, FULLTRAIN_DIR).replace("\\", "/")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name).replace("\\", "/")

        if os.path.exists(model_path):
            print(f"Model {model_path} already exists. Skipping training.")
            return None

        early_stopping = EarlyStopping(model_path=model_path, patience=self.args.patience, delta=self.args.delta, verbose=self.args.verbose)
        pbar = tqdm(range(self.args.num_epochs), desc="Full Training", unit="epoch", leave=True)

        history = {
            "train_loss": [],
            "val_loss": [],
            "dice_score": [],
            "iou_score": [],
        }
        with Timer() as timer:
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
                    "Best": f"{early_stopping.best_loss:.4f}",
                })
                if early_stopping.early_stop:
                    break
                torch.cuda.empty_cache()

            model = early_stopping.load_best_model(model)
            _, dice_score, iou_score = self.evaluate(model, test_loader)

        elapsed_time = timer.elapsed
        elapsed_time_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        print(f"Final Dice Score: {dice_score:.4f} - IOU Score: {iou_score:.4f} - Time: {elapsed_time_fmt}")

        results_dict = {
            "model_path": model_path,
            "dice_score": dice_score,
            "iou_score": iou_score,
            "elapsed_time": elapsed_time_fmt,
        }

        fulltrain_result_file = FULLTRAIN_RESULTS_FILE_TEMPLATE.format(experiment_name=self.args.experiment_name)
        results_path = os.path.join(RESULTS_DIR, fulltrain_result_file).replace("\\", "/")
        self.logger.log(results_path, results_dict, self.args)


        plot_name = self.args.model_name_template.replace(".pth", ".png").format(hash=self.args.hash, seed=self.args.seed)
        plot_path = os.path.join(model_dir, plot_name).replace("\\", "/")

        plot_history(history, plot_path)