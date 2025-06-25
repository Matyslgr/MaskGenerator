##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## utils
##

# import numpy as np
# import os, glob
# from abc import ABC, abstractmethod

# class BaseDatasetLoader(ABC):
#     def __init__(self, dataset_root: str):
#         if not os.path.exists(dataset_root):
#             raise FileNotFoundError(f"Dataset directory {dataset_root} does not exist.")
#         self.dataset_root = dataset_root

#     def check_maps(self, image_map: dict, mask_map: dict):
#         image_ids = set(image_map.keys())
#         mask_ids = set(mask_map.keys())

#         missing_in_masks = image_ids - mask_ids
#         missing_in_images = mask_ids - image_ids

#         assert not missing_in_masks, f"[ERROR] Missing masks for image IDs: {sorted(missing_in_masks)}"
#         assert not missing_in_images, f"[ERROR] Missing images for mask IDs: {sorted(missing_in_images)}"
#         return sorted(image_ids & mask_ids)

#     @abstractmethod
#     def get_pairs(self) -> np.ndarray:
#         """Doit retourner une liste numpy d'image/mask path pairs."""
#         raise NotImplementedError("Subclasses should implement this method to return image-mask pairs.")

# class SimuDatasetLoader(BaseDatasetLoader):
#     def get_pairs(self) -> np.ndarray:
#         image_dir = os.path.join(self.dataset_root, "Images/")
#         mask_dir = os.path.join(self.dataset_root, "Masks/")

#         image_files = glob.glob(os.path.join(image_dir, "*.png"))
#         mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

#         # Création des dictionnaires : id -> chemin complet
#         image_map = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in image_files}
#         mask_map  = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in mask_files}

#         common_ids = self.check_maps(image_map, mask_map)

#         pairs_path = []

#         for id_ in common_ids:
#             image_path = image_map[id_]
#             mask_path = mask_map[id_]

#             pairs_path.append((image_path, mask_path))

#         return np.array(pairs_path)

# class MoLaneDatasetLoader(BaseDatasetLoader):
#     def get_pairs(self) -> np.ndarray:
#         pairs = []

#         for subfolder in os.listdir(self.dataset_root):
#             folder = os.path.join(self.dataset_root, subfolder)
#             image_files = glob.glob(os.path.join(folder, "*_image.jpg"))
#             mask_files = glob.glob(os.path.join(folder, "*_label.png"))

#             image_map = {os.path.splitext(os.path.basename(p))[0].split('_')[0]: p for p in image_files}
#             mask_map = {os.path.splitext(os.path.basename(p))[0].split('_')[0]: p for p in mask_files}

#             common_ids = self.check_maps(image_map, mask_map)

#             for id_ in common_ids:
#                 image_path = image_map[id_]
#                 mask_path = mask_map[id_]
#                 pairs.append((image_path, mask_path))
#         if not pairs:
#             raise ValueError(f"No valid pairs found in dataset directory {self.dataset_root}.")
#         return np.array(pairs)

# class DatasetLoaderFactory:
#     registry = {
#         "simu_v0": SimuDatasetLoader,
#         "molane": MoLaneDatasetLoader,
#     }

#     @staticmethod
#     def get_loader(dataset_root: str) -> BaseDatasetLoader:
#         dataset_name = os.path.basename(dataset_root).lower()
#         if dataset_name not in DatasetLoaderFactory.registry:
#             raise ValueError(f"[DatasetFactory] Unknown dataset: {dataset_name}")
#         return DatasetLoaderFactory.registry[dataset_name](dataset_root)

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple
from mask_generator.config import DatasetConfig, TrainDatasetConfig
from mask_generator.dataset import ImageMaskDataset
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

def load_datasets(train_dataset_configs: List[TrainDatasetConfig],
                  eval_dataset_configs: List[DatasetConfig],
                  seed: int, train_transform, eval_transform) -> Tuple[ImageMaskDataset, ImageMaskDataset, ImageMaskDataset]:
    """
    Load training and evaluation datasets from the provided configurations.
    Args:
        train_dataset_configs (List[TrainDatasetConfig]): List of training dataset configurations.
        eval_dataset_configs (List[EvalDatasetConfig]): List of evaluation dataset configurations.
        seed (int): Random seed for reproducibility.
        train_transform: Transformations to apply to the training dataset.
        eval_transform: Transformations to apply to the evaluation dataset.
    Returns:
        Tuple[ImageMaskDataset, ImageMaskDataset, ImageMaskDataset]: Training dataset, validation dataset, and test dataset.
    """

    train_datasets = []

    for cfg in train_dataset_configs:
        if not os.path.exists(cfg.csv):
            raise FileNotFoundError(f"Train CSV '{cfg.csv}' not found.")

        df = pd.read_csv(cfg.csv)
        pairs_path = df[['image_path', 'mask_path']].values

        if len(pairs_path) == 0:
            raise ValueError(f"No pairs found in train CSV '{cfg.csv}'.")

        logging.info(f"Loading {len(pairs_path)} pairs from train CSV '{cfg.csv}'.")

        dataset = ImageMaskDataset(pairs_path=pairs_path, transform=train_transform)
        train_datasets.append(dataset)

    if not train_datasets:
        raise ValueError("At least one training dataset must be provided.")

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    # ===== Eval : split val/test =====
    if not eval_dataset_configs:
        raise ValueError("At least one evaluation dataset must be provided.")

    all_eval_pairs = []

    for cfg in eval_dataset_configs:
        if not os.path.exists(cfg.csv):
            raise FileNotFoundError(f"Eval CSV '{cfg.csv}' not found.")

        df = pd.read_csv(cfg.csv)
        pairs_path = df[['image_path', 'mask_path']].values

        if len(pairs_path) == 0:
            raise ValueError(f"No pairs found in eval CSV '{cfg.csv}'.")

        logging.info(f"Loading {len(pairs_path)} pairs from eval CSV '{cfg.csv}'.")

        all_eval_pairs.append(pairs_path)

    all_eval_pairs = np.concatenate(all_eval_pairs, axis=0)

    val_pairs, test_pairs = train_test_split(all_eval_pairs, test_size=0.8, random_state=seed, shuffle=True)

    val_dataset = ImageMaskDataset(val_pairs, transform=eval_transform)
    test_dataset = ImageMaskDataset(test_pairs, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset
