##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## dataset
##

import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from mask_generator.transforms import BaseTransform
import matplotlib.pyplot as plt

class ImageMaskDataset(Dataset):
    def __init__(self, pairs_path: np.ndarray, transform: BaseTransform):
        if not isinstance(pairs_path, np.ndarray):
            raise TypeError(f"pairs_path must be a numpy array, got {type(pairs_path)}")
        if not isinstance(transform, BaseTransform):
            raise TypeError(f"transform must be an instance of BaseTransform, got {type(transform)}")
        self.pairs_path = pairs_path
        self.transform = transform

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs_path[idx]

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = (original_mask >= 1).astype(np.uint8)

        image, mask = self.transform(image=original_image, mask=original_mask)

        unique = torch.unique(mask)
        if len(unique) != 2 or not torch.all((unique == 0) | (unique == 1)):
            raise ValueError(f"Expected binary mask with classes 0 and 1, found: {unique}")

        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title("Transformed Image")
        plt.axis('off')
        plt.savefig("transformed_image.png")
        return image, mask
