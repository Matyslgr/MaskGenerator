##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## experiment_tracker
##

import os
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import numpy as np

class ExperimentTracker:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "train_iou": [],
            "val_iou": [],
            "lr": [],
            "epoch_times": []
        }

        self.history_file = os.path.join(self.output_dir, "history.json")
        self.summary_file = os.path.join(self.output_dir, "summary.json")

    def _save_json(self, data: Dict[str, List[float]], file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_epoch(self, epoch: int, lr: float, epoch_time: float, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        self.history["epochs"].append(epoch)
        self.history["lr"].append(lr)
        self.history["epoch_times"].append(epoch_time)

        self.history["train_loss"].append(train_metrics["loss"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["train_dice"].append(train_metrics["dice"])
        self.history["val_dice"].append(val_metrics["dice"])
        self.history["train_iou"].append(train_metrics["iou"])
        self.history["val_iou"].append(val_metrics["iou"])

        self._save_json(self.history, self.history_file)

    