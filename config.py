##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## config
##

import os

DATASETS_DIR = "Datasets"
RESULTS_DIR = "Results"
MODELS_DIR = "Models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CROSSVAL_DIR = "crossval"
FULLTRAIN_DIR = "fulltrain"

CROSSVAL_RESULTS_FILE = "crossval_results.csv"
FULLTRAIN_RESULTS_FILE = "fulltrain_results.csv"

