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
from transform_manager import TransformManager
import matplotlib.pyplot as plt

# def show_sample(image, mask):

#     print("Image shape:", image.shape)
#     print("Mask shape:", mask.shape)
#     print("Image dtype:", image.dtype)
#     print("Mask dtype:", mask.dtype)

#     transform_manager = TransformManager(num_encoders=4)
#     image = transform_manager.denormalize(image)
#     image = image.permute(1, 2, 0).numpy()
#     mask = mask.squeeze().numpy()


#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(mask, cmap="gray")
#     plt.title("Mask")
#     plt.axis("off")
#     plt.show()

class ImageMaskDataset(Dataset):
    def __init__(self, pairs_path, transform=None):
        # Stocker les paires prétraitées
        self.pairs_path = pairs_path
        self.transform = transform

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs_path[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # Tensor (3, H, W)
            mask = augmented['mask']    # Tensor (H, W)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Convert to (1, H, W)
        mask = mask.float()

        # show_sample(image, mask)
        # exit(0)

        unique = torch.unique(mask)
        if len(unique) != 2 or not torch.all(unique == torch.tensor([0., 1.])):
            raise ValueError(f"Expected binary mask with classes 0 and 1, found: {unique}")
        return image, mask
