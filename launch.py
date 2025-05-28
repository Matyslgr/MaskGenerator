##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## launch
##

import os
import subprocess
import hashlib
import time
from config import MODELS_DIR, RESULTS_DIR

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_SCRIPT = "train.py"
CROSSVAL_DIR = "crossval"
FULLTRAIN_DIR = "fulltrain"

def generate_hash(model_args: dict, training_args: dict) -> str:
    """Generate a hash based on the arguments to ensure model uniqueness."""
    training_without_seed = training_args.copy()
    training_without_seed["seed"] = 0
    hash_input = str(model_args) + str(training_without_seed)
    return hashlib.md5(hash_input.encode()).hexdigest()[:10]

def model_exists(hash: str, mode: str, seed: int, name_template: str) -> bool:
    """Check if the model already exists."""
    if mode == "crossval":
        return all([os.path.exists(os.path.join(MODELS_DIR, CROSSVAL_DIR, hash, name_template.format(hash=hash, seed=seed, fold=i))) for i in range(1, 6)])
    elif mode == "fulltrain":
        return os.path.exists(os.path.join(MODELS_DIR, FULLTRAIN_DIR, name_template.format(hash=hash)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run_training(model_args: dict, training_args: dict, other_args: dict) -> None:
    command = ["python", TRAIN_SCRIPT]
    hash = generate_hash(model_args, training_args)

    if model_exists(hash, other_args["mode"], training_args["seed"], other_args["model_name_template"]):
        print(f"Model {hash} with seed {training_args['seed']} already exists.")
        return None

    args = {"hash": hash, **model_args, **training_args, **other_args}

    for arg, value in args.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{arg}")
            continue
        else:
            command.append(f"--{arg}")

        if isinstance(value, list) or isinstance(value, tuple):
            for v in value:
                command.append(str(v))
        else:
            command.append(str(value))

    print(f"{GREEN}Running model {hash} with seed {training_args['seed']}...{RESET}")
    subprocess.run(command)
    return hash

# Model arguments
n_convss = [2]
num_filterss = [[32, 64, 128, 256]]
DROPOUT = 0.0

# Training arguments
DATASET_VERSION = "v0"
seeds = range(1)
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 0.001
STEP_SIZE = 10
GAMMA = 0.1
PATIENCE = 30
DELTA = 0.0
TRAIN_IMAGE_SIZE = (256, 256)
# TEST_IMAGE_SIZE = (1080, 1920)
# List of augmentations to apply
# "geometry", "dropout", "color_invariance", "color_variation", "blur", "noise", "weather"
augmentationss = [["geometry", "dropout", "color_invariance", "color_variation", "blur", "noise", "weather"]]

# Other arguments
MODE = "fulltrain"
# MODE = "crossval"
VERBOSE = False
EXPERIMENT_NAME = "data_augmentation"

for seed in seeds:
    for num_filters in num_filterss:
        for n_convs in n_convss:
            for augmentations in augmentationss:
                print(f"{YELLOW}Augmentations: {augmentations}{RESET}")

                model_args = {
                    "n_convs": n_convs,
                    "num_filters": num_filters,
                    "dropout": DROPOUT,
                }

                training_args = {
                    "dataset_version": DATASET_VERSION,
                    "seed": seed + 42,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "lr": LR,
                    "step_size": STEP_SIZE,
                    "gamma": GAMMA,
                    "patience": PATIENCE,
                    "delta": DELTA,
                    "train_image_size": TRAIN_IMAGE_SIZE,
                    "augmentations": augmentations,
                }

                other_args = {
                    "mode": MODE,
                    "verbose": VERBOSE,
                    "experiment_name": EXPERIMENT_NAME,
                }

                # Create the name of the model
                if MODE == "crossval":
                    model_name_template = "model_{hash}_s{seed}_fold{fold}.pth"
                elif MODE == "fulltrain":
                    model_name_template = "model_{hash}.pth"
                else:
                    raise ValueError(f"Unknown mode: {MODE}")
                other_args["model_name_template"] = model_name_template

                start = time.time()
                hash = run_training(model_args, training_args, other_args)
                end = time.time()
                elapsed_time = end - start
                elapsed_time_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                if hash is not None:
                    print(f"{GREEN}Training completed{RESET} in {CYAN}{elapsed_time_fmt}{RESET} for hash {MAGENTA}{hash}{RESET} with seed {seed + 42}.")
