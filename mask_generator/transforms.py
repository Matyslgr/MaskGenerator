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
from scipy import ndimage
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from abc import ABC, abstractmethod
from mask_generator.logger import setup_logging

logger = setup_logging(__name__, level=logging.DEBUG)

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

class AspectAwareCropper:
    def __init__(self, target_ratio: float, crop_strategy: str = "adaptive", debug: bool = False):
        self.target_ratio = target_ratio
        assert crop_strategy in ["adaptive", "road_optimized", "center"], "Invalid crop_strategy"
        self.crop_strategy = crop_strategy
        self.debug = debug

    def compute_crop(self, img: np.ndarray, mask: np.ndarray = None):
        h, w = img.shape[:2]
        target_height = int(w / self.target_ratio)

        # Si l'image est déjà plus horizontale que le ratio cible, pas de crop
        if target_height >= h:
            logger.warning(f"Image height {h} is already greater than or equal to target height {target_height}. No crop applied.")
            return None

        if self.crop_strategy == "adaptive":
            if mask is None:
                logger.warning("Mask is None, falling back to road_optimized crop.")
                return self._road_optimized_crop(h, target_height)
            else:
                return self._adaptive_crop_based_on_lines(h, target_height, mask)
        elif self.crop_strategy == "road_optimized":
            return self._road_optimized_crop(h, target_height)
        else:
            return self._center_crop(h, target_height)

    def _adaptive_crop_based_on_lines(self, h: int, target_height: int, mask: np.ndarray):
        # Calculer la densité de lignes par ligne (row)
        line_density = np.sum(mask, axis=1).astype(float)

        # Lisser pour éviter le bruit
        if len(line_density) > 20:  # Éviter les erreurs sur petites images
            sigma = max(1, len(line_density) // 20)  # Sigma adaptatif
            smoothed_density = ndimage.gaussian_filter1d(line_density, sigma=sigma)
        else:
            smoothed_density = line_density

        # Éviter la division par zéro
        total_density = np.sum(smoothed_density)
        if total_density < 1e-6:
            # Pas de lignes détectées, utiliser crop road_optimized
            return self._road_optimized_crop(h, target_height)

        # Calculer le centre de masse des lignes
        indices = np.arange(len(smoothed_density))
        center_of_mass = np.average(indices, weights=smoothed_density)

        # Centrer le crop sur le centre de masse
        crop_start = int(center_of_mass - target_height / 2)
        crop_start = max(0, min(crop_start, h - target_height))
        crop_end = crop_start + target_height

        return (crop_start, crop_end)

    def _road_optimized_crop(self, h: int, target_height: int):
        # Pour les images de voiture, supprimer principalement la partie haute (ciel)
        # Garder plus de la partie basse (route proche) que de la partie haute (horizon)
        crop_start = int(h * 0.15)  # Commencer à 15% du haut
        crop_end = crop_start + target_height

        if crop_end > h:
            # Si on dépasse, privilégier la partie basse
            crop_end = h
            crop_start = h - target_height

        return (crop_start, crop_end)

    def _center_crop(self, h: int, target_height: int):
        crop_start = (h - target_height) // 2
        crop_end = crop_start + target_height
        return (crop_start, crop_end)

    def debug_visual(self, mask, crop_coords):
        import matplotlib.pyplot as plt
        y_start, y_end = crop_coords
        plt.imshow(mask, cmap='gray')
        plt.axhline(y=y_start, color='red')
        plt.axhline(y=y_end, color='red')
        plt.title("Crop Debug View")
        plt.show()

class AdaptiveCropToAspectRatio(DualTransform):
    """
    Crop the image to a target aspect ratio while preserving the content.
    This transform is designed to work with images that have a significant amount of content
    in the lower part (e.g., road images) and aims to keep the most relevant part of the image.
    It can adaptively crop based on the density of lines in the mask.
    """

    def __init__(self, target_ratio: float = 3.0, crop_strategy: str = "adaptive", debug: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.cropper = AspectAwareCropper(target_ratio, crop_strategy, debug)

    def apply(self, img: np.ndarray, crop_coords=None, **params):
        if crop_coords is None:
            return img

        y_start, y_end = crop_coords
        return img[y_start:y_end, :]

    def apply_to_mask(self, mask: np.ndarray, crop_coords=None, **params):
        if crop_coords is None:
            return mask

        y_start, y_end = crop_coords
        return mask[y_start:y_end, :]

    def get_params_dependent_on_data(self, params, data):
        img = data['image']
        mask = data.get('mask', None)

        crop_coords = self.cropper.compute_crop(img, mask)

        if self.cropper.debug and mask is not None and crop_coords is not None:
            self.cropper.debug_visual(mask, crop_coords)

        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self):
        return ("target_ratio", "crop_strategy", "debug")

class AlbumentationsTrainTransform(BaseTransform):
    def __init__(self, pad_divisor: int, image_size: tuple[int, int], augmentations = None, debug: bool = False):
        self.pad_divisor = pad_divisor
        self.image_size = image_size
        self.target_ratio = image_size[1] / image_size[0]  # width / height
        self.augmentation_factory = AugmentationFactory()
        self.augmentations = augmentations or []

        self.mean_list = [0.485, 0.456, 0.406]
        self.std_list = [0.229, 0.224, 0.225]

        self.mean_tensor = torch.tensor(self.mean_list, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std_list, dtype=torch.float32).view(3, 1, 1)
        self.max_pixel_value = 255.0
        logger.debug(f"Using augmentations: {self.augmentations}")

        compose = [
            AdaptiveCropToAspectRatio(target_ratio=self.target_ratio, crop_strategy="adaptive", debug=debug),
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=self.pad_divisor, pad_width_divisor=self.pad_divisor),
            # *self.augmentation_factory.build(self.augmentations),
            A.Normalize(mean=self.mean_list,
                        std=self.std_list,
                        max_pixel_value=self.max_pixel_value),
            ToTensorV2()
        ]

        self.transform = A.Compose(compose)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization to retrieve image in original scale [0, max_pixel_value]."""
        return (tensor * self.std_tensor * self.max_pixel_value) + (self.mean_tensor * self.max_pixel_value)

    def to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a numpy image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        image = self.denormalize(tensor).clamp(0, self.max_pixel_value)
        image = image.permute(1, 2, 0).cpu().numpy()
        return image.astype(np.uint8)

    def to_mask(self, tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Convert a tensor to a binary mask."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        mask = tensor.cpu().numpy()
        return (mask > threshold).astype(np.uint8) * 255  # Convert to binary mask

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        if mask is None:
            return self.transform(image=image)['image']
        res = self.transform(image=image, mask=mask)
        return res['image'], res['mask'].unsqueeze(0).float()

class KorniaInferTransform(BaseTransform):
    def __init__(self, pad_divisor: int, image_size: tuple[int, int], device: str = 'cpu', debug: bool = False):
        self.pad_divisor = pad_divisor
        self.image_size = image_size
        self.target_ratio = image_size[1] / image_size[0] # width / height
        self.device = device

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        self.max_pixel_value = 255.0
        self.cropper = AspectAwareCropper(target_ratio=self.target_ratio, crop_strategy="adaptive", debug=True)

    # def _resize_with_aspect_ratio(self, img: torch.Tensor, interpolation: str = 'bilinear') -> torch.Tensor:
    #     if interpolation not in ['bilinear', 'nearest']:
    #         raise ValueError("Interpolation must be either 'bilinear' or 'nearest'.")

    #     _, h, w = img.shape
    #     new_w = int(self.target_height * w / h) if h != 0 else w
    #     img = kt.resize(img, (self.target_height, new_w), interpolation=interpolation)
    #     return img

    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        """Resize the image tensor to the target size."""
        _, h, w = img.shape

        # if abs(aspect_ratio - self.target_ratio) > 0.01:
        #     logger.warning(f"Aspect ratio mismatch: {aspect_ratio} != {self.target_ratio}. Resizing to {(h, w)} to {self.image_size}.")

        target_height, target_width = self.image_size
        if h != target_height or w != target_width:
            img = kt.resize(img, (target_height, target_width), interpolation='bilinear')
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
        return (img - self.mean * self.max_pixel_value) / (self.std * self.max_pixel_value)

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        assert isinstance(image, np.ndarray), "Image must be a numpy array."
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be [H, W, C] with 3 channels"

        crop_coords = self.cropper.compute_crop(image, mask)
        if crop_coords is not None:
            y_start, y_end = crop_coords
            image = image[y_start:y_end, :, :]
        elif self.cropper.debug and mask is not None:
            self.cropper.debug_visual(mask, crop_coords)

        img_tensor = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() # [H, W, C] -> [C, H, W]
        img_tensor = self._resize(img_tensor)
        img_tensor = self._pad_if_needed(img_tensor)
        img_tensor = self._normalize(img_tensor)

        if mask is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a numpy array."
            assert mask.ndim == 2, "Mask must be grayscale with shape [H, W]"

            if crop_coords is not None:
                y_start, y_end = crop_coords
                mask = mask[y_start:y_end, :]
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float().to(self.device) # [H, W] -> [1, H, W]
            mask_tensor = self._pad_if_needed(mask_tensor)
            return img_tensor, mask_tensor
        return img_tensor

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization to retrieve image in original scale [0, max_pixel_value]."""
        return (tensor * self.std * self.max_pixel_value) + (self.mean * self.max_pixel_value)

    def to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a numpy image."""

        if tensor.dim() == 4:
            tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
        image = self.denormalize(tensor).clamp(0, self.max_pixel_value)
        image = image.permute(1, 2, 0).cpu().numpy() # [C,H,W] -> [H,W,C]
        return image.astype(np.uint8)

    def to_mask(self, tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0) # [B, C, H, W] -> [C, H, W]
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0) # [C, H, W] -> [H, W]

        mask = tensor.cpu().numpy()
        return (mask > threshold).astype(np.uint8) * 255
