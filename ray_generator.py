##
## EPITECH PROJECT, 2024
## MaskGenerator
## File description:
## ray_generator.py
##

import numpy as np
import matplotlib.pyplot as plt
import math

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
                distances[i] = dist
                found_obstacle = True
                break
            
            # Check if we hit an obstacle (white pixel)
            if mask[y, x] > 0:
                ray_endpoints.append((x, y))
                distances[i] = dist
                found_obstacle = True
                break
        
        # If no obstacle was found, add the maximum distance
        if not found_obstacle:
            x = int(origin_x + dx * max_distance)
            y = int(origin_y + dy * max_distance)
            ray_endpoints.append((x, y))
            distances[i] = max_distance
    
    return distances, ray_endpoints

# Fonction pour afficher les rayons si jamais pour debug/voir la representation
def show_rays(mask, ray_endpoints):
    height, width = mask.shape
    origin_x = width // 2
    origin_y = height - 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray_r', origin='upper')

    for end_x, end_y in ray_endpoints:
        plt.plot([origin_x, end_x], [origin_y, end_y], color='cyan')

    plt.plot(origin_x, origin_y, "ro")
    plt.title("Rays visualization")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.show()

