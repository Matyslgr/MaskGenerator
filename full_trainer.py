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

from dataset import ImageMaskDataset
from base_trainer import BaseTrainer
from utils import FulltrainCSVLogger, Timer
from config import FULLTRAIN_DIR, RESULTS_DIR, MODELS_DIR, FULLTRAIN_RESULTS_FILE


class FullTrainer(BaseTrainer):
    def __init__(self, args, model_fn):
        super().__init__(args, model_fn)
        self.logger = FulltrainCSVLogger()

    def train(self, pairs_path):
        dataset = ImageMaskDataset(pairs_path)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        model = self.create_model(self.args)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        model_name = self.args.model_name_template.format(hash=self.args.hash)
        model_dir = os.path.join(MODELS_DIR, FULLTRAIN_DIR).replace("\\", "/")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name).replace("\\", "/")

        pbar = tqdm(range(self.args.num_epochs), desc="Full Training", unit="epoch", leave=True)

        with Timer() as timer:
            for _ in pbar:
                avg_loss = self.train_epoch(model, data_loader, optimizer, criterion)
                scheduler.step()

                pbar.set_postfix({
                    "Train Loss": f"{avg_loss:.4f}",
                })

            _, dice_score, iou_score = self.evaluate(model, data_loader)

        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        elapsed_time = timer.elapsed
        elapsed_time_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        print(f"Final Dice Score: {dice_score:.4f} - IOU Score: {iou_score:.4f} - Time: {elapsed_time_fmt}")

        results_dict = {
            "model_path": model_path,
            "dice_score": dice_score,
            "iou_score": iou_score,
            "elapsed_time": elapsed_time_fmt,
        }

        results_path = os.path.join(RESULTS_DIR, FULLTRAIN_RESULTS_FILE).replace("\\", "/")
        self.logger.log(results_path, results_dict, self.args)
