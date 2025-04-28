##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## dataset
##

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def preprocess_image_mask(image_path, mask_path, image_size=(256, 256)):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    if image.size != mask.size:
        raise ValueError(f"Image size {image.size} and mask size {mask.size} do not match!")

    image_tensor = transform(image)

    mask = mask.resize(image_size, Image.NEAREST)
    mask_array = np.array(mask, dtype=np.uint8)

    # 0-255 to 0-1
    mask_array = (mask_array > 127).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

    unique = torch.unique(mask_tensor)
    if len(unique) != 2:
        raise ValueError(f"Expected binary mask with 2 classes, but found {len(unique)} classes.")
    if unique[0] != 0 or unique[1] != 1:
        raise ValueError(f"Expected binary mask with classes 0 and 1, but found {unique}.")

    return image_tensor, mask_tensor

class ImageMaskDataset(Dataset):
    def __init__(self, pairs_path, image_size=(256, 256)):
        # Stocker les paires prétraitées
        self.pairs_path = pairs_path
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs_path[idx]
        return preprocess_image_mask(image_path, mask_path, self.image_size)
