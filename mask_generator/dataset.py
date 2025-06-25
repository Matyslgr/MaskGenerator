##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## dataset
##

import os
import cv2
import torch
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from mask_generator.transforms import BaseTransform

class ImageMaskDataset(Dataset):
    def __init__(self, pairs_path: List[Tuple], transform: BaseTransform):
        if not isinstance(pairs_path, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs_path):
            raise TypeError(f"pairs_path must be a list of tuples, got {type(pairs_path)}")
        if not isinstance(transform, BaseTransform):
            raise TypeError(f"transform must be an instance of BaseTransform, got {type(transform)}")
        self.pairs_path = pairs_path
        self.transform = transform

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs_path[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' does not exist.")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file '{mask_path}' does not exist.")

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = (original_mask >= 1).astype(np.uint8)

        image, mask = self.transform(image=original_image, mask=original_mask)

        unique = torch.unique(mask)
        if len(unique) != 2 or not torch.all((unique == 0) | (unique == 1)):
            raise ValueError(f"Expected binary mask with classes 0 and 1, found: {unique}")
        return image, mask
