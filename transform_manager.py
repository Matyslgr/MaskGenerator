##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## transform_manager
##

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
import torchvision.transforms.functional as F

from augmentation_factory import AugmentationFactory

class ResizeWithAspectRatio(DualTransform):
    def __init__(self, height, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def get_params_dependent_on_data(self, params, data):
        # Récupère les dimensions de l'image
        orig_height, orig_width = data["image"].shape[:2]

        # Calcul de la nouvelle largeur en respectant le ratio d'aspect
        new_width = int(self.height * orig_width / orig_height) if orig_height != 0 else orig_width

        return {"resized_shape": (self.height, new_width)}

    def apply(self, img: np.ndarray, resized_shape: tuple[int, int], **params):
        # Redimensionne l'image avec la nouvelle largeur et hauteur fixe
        return cv2.resize(img, (resized_shape[1], resized_shape[0]), interpolation=self.interpolation)

    def apply_to_mask(self, mask: np.ndarray, resized_shape: tuple[int, int], **params):
        # Redimensionne le masque avec la nouvelle largeur et hauteur fixe
        return cv2.resize(mask, (resized_shape[1], resized_shape[0]), interpolation=self.mask_interpolation)

class TransformManager:
    def __init__(self, seed, num_encoders):
        self.seed = seed
        self.divisor = 2 ** num_encoders
        self.transforms_after = [
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0),
            ToTensorV2()
        ]
        self.augmentation_factory = AugmentationFactory(seed)

    def get_train_transform(self, augmentation_names: list[str], image_size: tuple[int, int] = (256, 256)):
        return A.Compose([
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.divisor, pad_width_divisor=self.divisor),
            A.RandomCrop(height=image_size[0], width=image_size[1]),
            *self.augmentation_factory.build(augmentation_names),
            *self.transforms_after
        ], seed=self.seed)

    def get_eval_transform(self):
        return A.Compose([
            ResizeWithAspectRatio(height=256, p=1.0),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.divisor, pad_width_divisor=self.divisor),
            *self.transforms_after
        ], seed=self.seed, save_applied_params=True)

    @staticmethod
    def denormalize(tensor) -> torch.Tensor:
        # Inverse de la normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    @staticmethod
    def reverse_image(tensor: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:

        # [B, C, H, W] -> [C, H, W]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # CPU
        image = tensor.cpu()

        # Reverse the normalization and clamp the values to [0, 1]
        image = TransformManager.denormalize(image).clamp(0, 1)


        # Resize the image to the original shape
        image = F.center_crop(image, output_size=original_shape)

        # Convert CHW → HWC
        image = image.permute(1, 2, 0)

        # Convert to numpy array and scale to [0, 255]
        image = (image.numpy() * 255).astype(np.uint8)

        return image

    @staticmethod
    def reverse_mask(tensor: torch.Tensor, original_shape: tuple[int, int]) -> np.ndarray:

        # [B, C, H, W] -> [C, H, W]
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # [C, H, W] -> [H, W]
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)

        # [H, W] + CPU
        mask = tensor.cpu()

        mask = F.center_crop(mask, output_size=original_shape)

        # Convert to numpy array and scale to [0, 255]
        mask = (mask.numpy() * 255).astype(np.uint8)

        return mask