##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## settings
##

import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
experiment_dir = os.path.join(root, "experiments")
train_dataset_dir = os.path.join(root, "Datasets", "train")
test_dataset_dir = os.path.join(root, "Datasets", "test")
model_filename = "model.pth"
config_filename = "config.yaml"
metrics_filename = "metrics.csv"
results_filename = "results.yaml"
samples_dir = "samples"
plots_dir = "plots"

os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(train_dataset_dir, exist_ok=True)
os.makedirs(test_dataset_dir, exist_ok=True)
