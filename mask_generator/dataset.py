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
from torchvision.transforms import ToTensor

class ImageMaskDataset(Dataset):
    def __init__(self, pairs_path, transform=None):
        # Stocker les paires prétraitées
        self.pairs_path = pairs_path
        self.transform = transform
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs_path[idx]

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = (original_mask >= 1).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=original_image, mask=original_mask)
            image_np = augmented['image']  # Tensor (3, H, W)
            mask_np = augmented['mask']    # Tensor (H, W)
        else:
            image_np = original_image
            mask_np = original_mask

        image = self.to_tensor(image_np)
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()  # Convert to (1, H, W)

        unique = torch.unique(mask)
        if len(unique) != 2 or not torch.all((unique == 0) | (unique == 1)):
            raise ValueError(f"Expected binary mask with classes 0 and 1, found: {unique}")

        return image, mask, original_image, original_mask
