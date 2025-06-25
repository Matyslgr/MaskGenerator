##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## transform_manager
##

import cv2
import logging
import numpy as np
import kornia.geometry.transform as kt
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from abc import ABC, abstractmethod
from mask_generator.logger import setup_logging

logger = setup_logging(__file__, level=logging.DEBUG)

from mask_generator.augmentation_factory import AugmentationFactory

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        raise NotImplementedError("Subclasses should implement this method.")

class ResizeWithAspectRatio(DualTransform):
    def __init__(self, height: int = 256, p: float = 1.0):
        super().__init__(p=p)
        self.height = height

    def apply(self, img: np.ndarray, **params):
        h, w = img.shape[:2]
        new_w = int(self.height * w / h) if h != 0 else w
        resized_shape = (new_w, self.height)
        return cv2.resize(img, resized_shape, interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask: np.ndarray, **params):
        h, w = mask.shape[:2]
        new_w = int(self.height * w / h) if h != 0 else w
        resized_shape = (new_w, self.height)
        return cv2.resize(mask, resized_shape, interpolation=cv2.INTER_NEAREST)

    def get_transform_init_args_names(self):
        return ("height",)

class AlbumentationsTrainTransform(BaseTransform):
    def __init__(self, pad_divisor: int, image_size: tuple[int, int] = (256, 256), augmentations = None):
        self.pad_divisor = pad_divisor
        self.image_size = image_size
        self.augmentation_factory = AugmentationFactory()
        self.augmentations = augmentations or []
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        logger.debug(f"Using augmentations: {self.augmentations}")

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization of a tensor."""
        return tensor * self.std + self.mean

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        compose = [
            ResizeWithAspectRatio(height=self.image_size[0], p=1.0),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.pad_divisor, pad_width_divisor=self.pad_divisor),
            A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
            *self.augmentation_factory.build(self.augmentations),
            A.Normalize(mean=self.mean,
                        std=self.std,
                        max_pixel_value=255.0),
            ToTensorV2()
        ]

        transform = A.Compose(compose)
        if mask is None:
            return transform(image=image)['image']
        res = transform(image=image, mask=mask)
        return res['image'], res['mask'].unsqueeze(0).float()

class AlbumentationsInferTransform:
    def __init__(self, pad_divisor: int):
        self.pad_divisor = pad_divisor

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        compose = [
            ResizeWithAspectRatio(height=256, p=1.0),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.pad_divisor, pad_width_divisor=self.pad_divisor),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0),
            ToTensorV2()
        ]

        transform = A.Compose(compose)
        if mask is None:
            return transform(image=image)['image']
        res = transform(image=image, mask=mask)
        return res['image'], res['mask'].unsqueeze(0).float() # Ensure mask is [1, H, W] for consistency

class KorniaInferTransform(BaseTransform):
    def __init__(self, pad_divisor: int, target_height: int = 256, device: str = 'cpu'):
        self.pad_divisor = pad_divisor
        self.target_height = target_height
        self.device = device

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _resize_with_aspect_ratio(self, img: torch.Tensor, interpolation: str = 'bilinear') -> torch.Tensor:
        if interpolation not in ['bilinear', 'nearest']:
            raise ValueError("Interpolation must be either 'bilinear' or 'nearest'.")

        _, h, w = img.shape
        new_w = int(self.target_height * w / h) if h != 0 else w
        img = kt.resize(img, (self.target_height, new_w), interpolation=interpolation)
        return img

    def _pad_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        pad_h = (self.pad_divisor - h % self.pad_divisor) % self.pad_divisor
        pad_w = (self.pad_divisor - w % self.pad_divisor) % self.pad_divisor
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            img = F.pad(img, padding, mode='constant', value=0)
        return img

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize the image tensor."""
        return (img - self.mean) / self.std

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        assert isinstance(image, np.ndarray), "Image must be a numpy array."
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be HWC with 3 channels"

        img_tensor = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0 # [H, W, C] -> [C, H, W]
        img_tensor = self._resize_with_aspect_ratio(img_tensor, interpolation='bilinear')
        img_tensor = self._pad_if_needed(img_tensor)
        img_tensor = self._normalize(img_tensor)

        if mask is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a numpy array."
            assert mask.ndim == 2, "Mask must be grayscale with shape [H, W]"

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float().to(self.device) # [H, W] -> [1, H, W]
            mask_tensor = self._resize_with_aspect_ratio(mask_tensor, interpolation='nearest')
            mask_tensor = self._pad_if_needed(mask_tensor)
            return img_tensor, mask_tensor
        return img_tensor

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization of a tensor."""
        return tensor * self.std + self.mean

    def to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a numpy image."""

        if tensor.dim() == 4:
            tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
        image = self.denormalize(tensor).clamp(0, 1)
        image = image.permute(1, 2, 0).cpu().numpy() # [C,H,W] -> [H,W,C]
        return (image * 255).astype(np.uint8)

    def to_mask(self, tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0) # [C, H, W] -> [H, W]

        mask = tensor.cpu().numpy()
        return (mask > threshold).astype(np.uint8) * 255
