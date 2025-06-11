##
## EPITECH PROJECT, 2025
## root [SSH: pinkcc]
## File description:
## training_monitor
##

import os
import csv
import yaml
import numpy as np
import time
from typing import Dict
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mask_generator.settings as settings

GREEN = "\033[92m"
RESET = "\033[0m"

class TrainingLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

        self.paths = {
            "results": os.path.join(output_dir, settings.results_filename),
            "metrics": os.path.join(output_dir, settings.metrics_filename),
            "results_yaml": os.path.join(output_dir, settings.results_filename),
            "curves": os.path.join(output_dir, settings.plots_dir, "curves.png"),
            "lr_schedule": os.path.join(output_dir, settings.plots_dir, "lr_schedule.png"),
            "conf_matrix": os.path.join(output_dir, settings.plots_dir, "confusion_matrix.png"),
        }

        os.makedirs(os.path.join(output_dir, settings.plots_dir), exist_ok=True)

        self.history = {
            "epochs": [],
            "train": [],
            "val": [],
            "lr": [],
            "epoch_times": [],
        }

        self._init_csv_metrics()

    def _init_csv_metrics(self):
        with open(self.paths["metrics"], mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["epoch", "lr", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

    def log_epoch(self, epoch: int, lr: float, epoch_time: float, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        self.history["lr"].append(lr)
        self.history["train"].append(train_metrics)
        self.history["val"].append(val_metrics)
        self.history["epoch_times"].append(epoch_time)

        with open(self.paths["metrics"], mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                epoch,
                lr,
                train_metrics.get("loss", 0.0),
                train_metrics.get("dice", 0.0),
                train_metrics.get("iou", 0.0),
                val_metrics.get("loss", 0.0),
                val_metrics.get("dice", 0.0),
                val_metrics.get("iou", 0.0),
            ])

    def save_results(self, test_metrics: Dict[str, float], elapsed_time: float, best_epoch: int):
        results = {
            "total_epochs": len(self.history["train"]),
            "best_epoch": best_epoch,
            "test": test_metrics,
            "time": {
                "seconds": elapsed_time,
                "formatted": time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
            }
        }

        with open(self.paths["results"], 'w') as f:
            yaml.dump(results, f, indent=4)

    def save_lr_plot(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.history["lr"], label='Learning Rate', color='blue')
        plt.title('Learning Rate per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.paths["lr_schedule"])
        plt.close()

    def save_curves(self):
        train_losses = [m["loss"] for m in self.history["train"]]
        val_losses = [m["loss"] for m in self.history["val"]]
        train_dices = [m["dice"] for m in self.history["train"]]
        val_dices = [m["dice"] for m in self.history["val"]]
        train_ious = [m["iou"] for m in self.history["train"]]
        val_ious = [m["iou"] for m in self.history["val"]]

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(train_losses, label='Train Loss', color='blue')
        axs[0].plot(val_losses, label='Val Loss', color='orange')
        axs[0].set_title('Loss per Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(train_dices, label='Train Dice', color='blue')
        axs[1].plot(val_dices, label='Val Dice', color='orange')
        axs[1].set_title('Dice Score per Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Dice Score')
        axs[1].legend()

        axs[2].plot(train_ious, label='Train IOU', color='blue')
        axs[2].plot(val_ious, label='Val IOU', color='orange')
        axs[2].set_title('IOU Score per Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('IOU Score')
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(self.paths["curves"])
        plt.close()

    def save_conf_matrix(self, cm : np.ndarray):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Global)")
        plt.tight_layout()
        plt.savefig(self.paths["conf_matrix"])
        plt.close()

    def save_epoch_times_plot(self):
        if not self.history["epoch_times"]:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.history["epoch_times"], label='Epoch Time (s)', color='green')
        plt.title('Epoch Duration per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.run_dir, settings.plots_dir, "epoch_times.png"))
        plt.close()

    def save_plots(self):
        self.save_lr_plot()
        self.save_epoch_times_plot()
        self.save_curves()

    def save_all(self, test_metrics: Dict[str, float], elapsed_time: float, best_epoch: int):
        cm_sklearn = test_metrics.get("conf_matrix", None)
        test_metrics.pop("conf_matrix", None)
        test_metrics["conf_matrix"] = cm_sklearn.tolist() if cm_sklearn is not None else []
        self.save_results(test_metrics, elapsed_time, best_epoch)
        self.save_plots()
        self.save_conf_matrix(cm_sklearn)
        print(f"{GREEN}All Results saved for run {self.run_dir}{RESET}")

    