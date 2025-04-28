##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## csv_logger
##

import os
import csv

class BaseCSVLogger:
    def __init__(self, excluded_keys=None):
        self.excluded_keys = excluded_keys or [
            "verbose", "mode", "model_name_template"
        ]
        self.base_fields = self.define_base_fields()

    def define_base_fields(self):
        """Doit être redéfinie dans les classes filles."""
        raise NotImplementedError("define_base_fields() must be implemented in subclasses.")

    def _is_empty(self, path):
        return not os.path.exists(path) or os.stat(path).st_size == 0

    def _build_args_dict(self, args):
        return {k: v for k, v in vars(args).items() if k not in self.excluded_keys}

    def _write_dict(self, path: str, row: dict, fieldnames: list):
        file_is_empty = self._is_empty(path)
        with open(path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if file_is_empty:
                writer.writeheader()
            writer.writerow(row)

    def log(self, path: str, result_dict: dict, args):
        args_dict = self._build_args_dict(args)
        full_dict = {**result_dict, **args_dict}

        additional_keys = sorted(k for k in full_dict if k not in self.base_fields)
        fieldnames = self.base_fields + additional_keys

        self._write_dict(path, full_dict, fieldnames)

class CrossvalCSVLogger(BaseCSVLogger):
    def define_base_fields(self):
        return [
            "hash", "fold", "dice_score", "iou_score", "elapsed_time",
            "dataset_version", "model_path", "seed"
        ]

class FulltrainCSVLogger(BaseCSVLogger):
    def define_base_fields(self):
        return [
            "hash", "dataset_version", "dice_score", "iou_score",
            "elapsed_time", "model_path", "seed"
        ]
