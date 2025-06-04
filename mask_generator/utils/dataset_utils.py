##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## utils
##

import numpy as np
import os, glob
from abc import ABC, abstractmethod

class BaseDatasetLoader(ABC):
    def __init__(self, dataset_root: str):
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset directory {dataset_root} does not exist.")
        self.dataset_root = dataset_root

    def check_maps(self, image_map: dict, mask_map: dict):
        image_ids = set(image_map.keys())
        mask_ids = set(mask_map.keys())

        missing_in_masks = image_ids - mask_ids
        missing_in_images = mask_ids - image_ids

        assert not missing_in_masks, f"[ERROR] Missing masks for image IDs: {sorted(missing_in_masks)}"
        assert not missing_in_images, f"[ERROR] Missing images for mask IDs: {sorted(missing_in_images)}"
        return sorted(image_ids & mask_ids)

    @abstractmethod
    def get_pairs(self) -> np.ndarray:
        """Doit retourner une liste numpy d'image/mask path pairs."""
        pass


class SimuDatasetLoader(BaseDatasetLoader):
    def get_pairs(self) -> np.ndarray:
        image_dir = os.path.join(self.dataset_root, "Images/")
        mask_dir = os.path.join(self.dataset_root, "Masks/")

        image_files = glob.glob(os.path.join(image_dir, "*.png"))
        mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

        # Création des dictionnaires : id -> chemin complet
        image_map = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in image_files}
        mask_map  = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in mask_files}

        common_ids = self.check_maps(image_map, mask_map)

        pairs_path = []

        for id_ in common_ids:
            image_path = image_map[id_]
            mask_path = mask_map[id_]

            pairs_path.append((image_path, mask_path))

        return np.array(pairs_path)

class MoLaneDatasetLoader(BaseDatasetLoader):
    def get_pairs(self) -> np.ndarray:
        pairs = []

        for subfolder in os.listdir(self.dataset_root):
            folder = os.path.join(self.dataset_root, subfolder)
            image_files = glob.glob(os.path.join(folder, "*_image.jpg"))
            mask_files = glob.glob(os.path.join(folder, "*_label.png"))

            image_map = {os.path.splitext(os.path.basename(p))[0].split('_')[0]: p for p in image_files}
            mask_map = {os.path.splitext(os.path.basename(p))[0].split('_')[0]: p for p in mask_files}

            common_ids = self.check_maps(image_map, mask_map)

            for id_ in common_ids:
                image_path = image_map[id_]
                mask_path = mask_map[id_]
                pairs.append((image_path, mask_path))
        if not pairs:
            raise ValueError(f"No valid pairs found in dataset directory {self.dataset_root}.")
        return np.array(pairs)

class DatasetLoaderFactory:
    registry = {
        "simu_v0": SimuDatasetLoader,
        "molane": MoLaneDatasetLoader,
    }

    @staticmethod
    def get_loader(dataset_root: str) -> BaseDatasetLoader:
        dataset_name = os.path.basename(dataset_root).lower()
        if dataset_name not in DatasetLoaderFactory.registry:
            raise ValueError(f"[DatasetFactory] Unknown dataset: {dataset_name}")
        return DatasetLoaderFactory.registry[dataset_name](dataset_root)
