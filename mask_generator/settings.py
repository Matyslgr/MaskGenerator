##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## settings
##

import os

# Directories
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
experiment_dir = os.path.join(root, "experiments")
train_dataset_dir = os.path.join(root, "Datasets", "train")
test_dataset_dir = os.path.join(root, "Datasets", "test")

os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(train_dataset_dir, exist_ok=True)
os.makedirs(test_dataset_dir, exist_ok=True)

# Filenames
model_filename = "model.pth"
onnx_filename = "model.onnx"
config_filename = "config.yaml"
metrics_filename = "metrics.csv"
results_filename = "results.json"
conf_matrix_filename = "conf_matrix.npy"
logging_filename = "experiment.log"

# Other
logger_name = "MaskGenerator"
