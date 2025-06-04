##
## EPITECH PROJECT, 2025
## root [SSH: robocar-scaleway]
## File description:
## ray_distance
##

import os
import cv2
import torch
import random
import numpy as np
import os
import yaml
import torch.nn as nn

from mask_generator.config import ModelConfig
from mask_generator.models.utils import create_model
from mask_generator.transforms import EvalTransform, TensorDecoder

# -------------------------------------------
# Ray Tracing Core
# -------------------------------------------

# NumPy 2.x compatibility patch
def ensure_numpy_array(data):
    """Ensure data is a properly formatted NumPy array for ray calculations"""
    if isinstance(data, np.ndarray):
        # Make sure we're working with a 2D array
        if data.ndim == 2:
            return data
        elif data.ndim > 2:
            return data.squeeze()
    return np.array(data)

def generate_rays(mask, num_rays=50, fov_degrees=120, max_distance=None):
    """
    Generate rays from the bottom center of the mask and calculate distances to obstacles.
    Compatible with NumPy 2.x
    """
    # Ensure mask is properly formatted
    mask = ensure_numpy_array(mask)

    # Rest of the function remains the same
    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

    # Calculate the angle range
    fov_radians = np.radians(fov_degrees)
    half_fov = fov_radians / 2

    # Calculate the angles for each ray
    angles = np.linspace(-half_fov, half_fov, num_rays)

    ray_endpoints = []
    distances = {}

    if max_distance is None:
        max_distance = int(np.sqrt(width**2 + height**2))

    # Cast rays and find intersections
    for i, angle in enumerate(angles):
        # Direction vector
        dx = np.sin(angle)
        dy = -np.cos(angle)  # Negative because y-axis is inverted in images

        # Ray tracing
        found_obstacle = False
        for dist in range(1, max_distance):
            x = int(origin_x + dx * dist)
            y = int(origin_y + dy * dist)

            # Check if we're out of bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                ray_endpoints.append((x, y))
                distances[f"ray_{i}"] = dist
                found_obstacle = True
                break

            # Check if we hit an obstacle (white pixel)
            if mask[y, x] > 0:
                ray_endpoints.append((x, y))
                distances[f"ray_{i}"] = dist
                found_obstacle = True
                break

        # If no obstacle was found, add the maximum distance
        if not found_obstacle:
            x = int(origin_x + dx * max_distance)
            y = int(origin_y + dy * max_distance)
            ray_endpoints.append((x, y))
            distances[f"ray_{i}"] = max_distance

    return distances, ray_endpoints

def compute_ray_distances(image: np.ndarray, model: nn.Module, pad_divisor: int, num_rays: int = 50, fov_degrees: int = 120, max_distance: int = None) -> dict:
    """
    Compute mask ray distances using a segmentation model.
    Args:
        image (np.ndarray): Input image for segmentation.
        model (nn.Module): Pre-trained segmentation model.
        num_rays (int): Number of rays to cast.
        fov_degrees (int): Field of view in degrees.
        max_distance (int, optional): Maximum distance for ray tracing. Defaults to None.
    Returns:
        dict: Dictionary with ray distances
    """

    # Ensure the model is in evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = EvalTransform(pad_divisor=pad_divisor, to_tensor=True)
    transformed = transform(image=image)

    image_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)

    decoder = TensorDecoder()

    mask_np = decoder.to_mask(output.cpu())

    distances, _ = generate_rays(mask_np, num_rays=num_rays, fov_degrees=fov_degrees, max_distance=max_distance)
    return distances
