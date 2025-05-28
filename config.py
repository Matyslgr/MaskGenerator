##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## config
##

import os

TRAIN_DATASETS_DIR = "TrainDatasets"
TEST_DATASET_DIR = "TestDataset"
RESULTS_DIR = "Results"
MODELS_DIR = "Models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CROSSVAL_DIR = "crossval"
FULLTRAIN_DIR = "fulltrain"

CROSSVAL_RESULTS_FILE_TEMPLATE = "crossval_results_{experiment_name}.csv"
FULLTRAIN_RESULTS_FILE_TEMPLATE = "fulltrain_results_{experiment_name}.csv"

