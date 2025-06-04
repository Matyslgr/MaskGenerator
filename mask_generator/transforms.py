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
from mask_generator.augmentation_factory import AugmentationFactory

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

class TrainTransform:
    def __init__(self, seed: int, pad_divisor: int, image_size: tuple[int, int] = (256, 256), to_tensor: bool = False, augmentations_names = None):
        self.seed = seed
        self.pad_divisor = pad_divisor
        self.image_size = image_size
        self.to_tensor = to_tensor
        self.augmentation_factory = AugmentationFactory(seed)
        self.augmentations = augmentations_names if augmentations_names is not None else []

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        compose = [
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.pad_divisor, pad_width_divisor=self.pad_divisor),
            A.RandomCrop(height=self.image_size[0], width=self.image_size[1]),
            *self.augmentation_factory.build(self.augmentations),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0)
        ]

        if self.to_tensor:
            compose.append(ToTensorV2())

        transform = A.Compose(compose, seed=self.seed)
        if mask is None:
            return transform(image=image)
        return transform(image=image, mask=mask)

class EvalTransform:
    def __init__(self, pad_divisor: int, to_tensor: bool = False):
        self.pad_divisor = pad_divisor
        self.to_tensor = to_tensor

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        compose = [
            ResizeWithAspectRatio(height=256, p=1.0),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.pad_divisor, pad_width_divisor=self.pad_divisor),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0)
        ]

        if self.to_tensor:
            compose.append(ToTensorV2())

        transform = A.Compose(compose)
        if mask is None:
            return transform(image=image)
        return transform(image=image, mask=mask)

class TensorDecoder:
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

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
