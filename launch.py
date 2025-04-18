##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## launch
##

import os
import subprocess

TRAIN_SCRIPT = "train.py"
MODELS_DIR = "Models"

os.makedirs(MODELS_DIR, exist_ok=True)

def run_training(args: dict):
    command = ["python", TRAIN_SCRIPT]
    args["model_path"] = f"{MODELS_DIR}/model.pth"

    # if os.path.exists(args["model_path"]):
    #     print(f"Model with hash {hash} already exists. Skipping...")
    #     return

    for arg, value in args.items():
        command.append(f"--{arg}")
        command.append(str(value))

    subprocess.run(command)


DATASET_ROOT = "Dataset/"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_SEEDS = 1
LR = 0.001
STEP_SIZE = 10
GAMMA = 0.1
PATIENCE = 30
DELTA = 0.0
VERBOSE = 0

args = {
    "dataset_root": DATASET_ROOT,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "num_seeds": NUM_SEEDS,
    "lr": LR,
    "step_size": STEP_SIZE,
    "gamma": GAMMA,
    "patience": PATIENCE,
    "delta": DELTA,
    "verbose": VERBOSE,
}
run_training(args)
