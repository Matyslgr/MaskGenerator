##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## experiment_tracker
##

import os
import csv
import json
from typing import Dict, Any, List
import numpy as np
import time
from dataclasses import asdict

from mask_generator.metrics import Metrics

class ExperimentTracker:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics_path = os.path.join(self.output_dir, "metrics.csv")
        self.results_path = os.path.join(self.output_dir, "results.json")
        self.conf_matrix_path = os.path.join(self.output_dir, "conf_matrix.npy")

        self.fieldnames = ['epoch', 'lr', 'epoch_time']
        for loader in ['train', 'val']:
            self.fieldnames.extend([
                f'{loader}_loss', f'{loader}_dice', f'{loader}_iou', f'{loader}_acc', f'{loader}_precision', f'{loader}_recall', f'{loader}_f1'
            ])

        if not os.path.exists(self.metrics_path):
            self._init_csv()

    def _init_csv(self) -> None:
        with open(self.metrics_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
            writer.writeheader()

    def _append_csv(self, data: Dict[str, Any]) -> None:
        with open(self.metrics_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
            writer.writerow(data)

    def _save_json(self, data: Dict[str, List[float]], file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_epoch(self, epoch: int, lr: float, epoch_time: float, train_metrics: Metrics, val_metrics: Metrics) -> None:
        data = {
            'epoch': epoch,
            'lr': lr,
            'epoch_time': epoch_time
        }
        for loader, metrics in zip(['train', 'val'], [train_metrics, val_metrics]):
            for metric_name, value in ['loss', 'dice', 'iou', 'acc', 'precision', 'recall', 'f1']:
                data[f'{loader}_{metric_name}'] = getattr(metrics, metric_name, 0.0)

        self._append_csv(data)

    def save_results(self, best_epoch: int, elapsed_time: float, test_metrics: Metrics) -> None:
        results = {
            'best_epoch': best_epoch,
            'elapsed_time_sec': elapsed_time,
            'elapsed_time_str': time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
            'test_metrics': asdict(test_metrics)
        }

        self._save_json(results, self.results_path)

    def save_conf_matrix(self, conf_matrix: np.ndarray) -> None:
        np.save(self.conf_matrix_path, conf_matrix)
