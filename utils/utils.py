##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## utils
##

import torch
import numpy as np
import random
import os, glob

def set_deterministic_behavior(seed):
    """
    Function to set deterministic behavior for reproducibility.

    Args:
        seed (int): The seed value for random number generators.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)

def get_all_train_pairs_path(dataset_root):
    image_dir = os.path.join(dataset_root, "Images/")
    mask_dir = os.path.join(dataset_root, "Masks/")

    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

    # Création des dictionnaires : id -> chemin complet
    image_map = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in image_files}
    mask_map  = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in mask_files}

    image_ids = set(image_map.keys())
    mask_ids = set(mask_map.keys())

    # Vérifie s'il y a des ID manquants
    missing_in_masks = image_ids - mask_ids
    missing_in_images = mask_ids - image_ids

    assert not missing_in_masks, f"[ERROR] Missing masks for image IDs: {sorted(missing_in_masks)}"
    assert not missing_in_images, f"[ERROR] Missing images for mask IDs: {sorted(missing_in_images)}"

    # Construire les paires à partir des IDs communs
    common_ids = sorted(image_ids & mask_ids)
    pairs_path = []

    for id_ in common_ids:
        image_path = image_map[id_]
        mask_path = mask_map[id_]

        pairs_path.append((image_path, mask_path))

    return np.array(pairs_path)

def get_all_test_pairs_path(dataset_root):
    image_dir = os.path.join(dataset_root, "Images/")
    mask_dir = os.path.join(dataset_root, "Masks/")

    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

    # Création des dictionnaires : id -> chemin complet
    image_map = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in image_files}
    mask_map  = {os.path.splitext(os.path.basename(p))[0].split('_')[-1]: p for p in mask_files}

    image_ids = set(image_map.keys())
    mask_ids = set(mask_map.keys())

    # Vérifie s'il y a des ID manquants
    # missing_in_masks = image_ids - mask_ids
    # missing_in_images = mask_ids - image_ids

    # assert not missing_in_masks, f"[ERROR] Missing masks for image IDs: {sorted(missing_in_masks)}"
    # assert not missing_in_images, f"[ERROR] Missing images for mask IDs: {sorted(missing_in_images)}"

    # Construire les paires à partir des IDs communs
    common_ids = sorted(image_ids & mask_ids)
    pairs_path = []

    for id_ in common_ids:
        image_path = image_map[id_]
        mask_path = mask_map[id_]

        pairs_path.append((image_path, mask_path))

    return np.array(pairs_path)
