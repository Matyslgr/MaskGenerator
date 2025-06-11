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
import json
import logging
from omegaconf import OmegaConf
import itertools
import shutil
from datetime import timedelta
from git import Repo
from pathlib import Path

from mask_generator.config import Config
import mask_generator.settings as settings
from scripts.logger_manager import LoggerManager

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

CONFIG_FILE = "config.yaml"
TRAIN_MODULE = "mask_generator.train"

logger_manager = LoggerManager(__name__, level=logging.DEBUG)
logger = logger_manager.get_logger()

def get_git_metadata(repo_path: str = ".") -> dict:
    repo = Repo(Path(repo_path).resolve())
    head_commit = repo.head.commit

    return {
        "commit": head_commit.hexsha,
        "branch": repo.active_branch.name if not repo.head.is_detached else "detached",
        "dirty": repo.is_dirty(untracked_files=True),
    }

def launch_training(config_path: str) -> None:
    """Launch the training script with the given configuration."""
    logger.info(f"Launching training with config: {config_path}")
    command = ["python", "-m", TRAIN_MODULE, "--config", config_path]
    subprocess.run(command)

def make_hash(*dicts: dict) -> str:
    """Create MD5 hash from concatenated dictionaries."""
    combined = {}
    for d in dicts:
        combined.update(d)

    s = json.dumps(combined, sort_keys=True).encode()
    return hashlib.md5(s).hexdigest()[:10]

def dict_product(d: dict):
    """Generate all combinations of dictionary values."""
    keys, values = zip(*d.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def recursive_product(d: dict):
    groups = d.keys()
    groups_product = {g: list(dict_product(d[g])) for g in groups}

    for combined in itertools.product(*groups_product.values()):
        yield dict(zip(groups, combined))

def run_already_exists(run_hash: str) -> bool:
    """Check if a run with the given hash already exists and is fully completed."""
    if not os.path.exists(settings.experiment_dir):
        return False
    if not os.path.isdir(settings.experiment_dir):
        return False

    for name in os.listdir(settings.experiment_dir):
        run_dir_path = os.path.join(settings.experiment_dir, name)
        if not os.path.isdir(run_dir_path):
            continue
        if run_hash in name:
            # Check if all expected files exist
            expected_files = [
                settings.model_filename,
                settings.config_filename,
                settings.metrics_filename,
                settings.results_filename,
            ]
            if all(os.path.exists(os.path.join(run_dir_path, f)) for f in expected_files):
                return True
            else:
                logger.warning(f"Run {run_hash} is incomplete. Expected files are missing in {run_dir_path}.")
                shutil.rmtree(run_dir_path, ignore_errors=True)

    return False

def format_run_dir(run_hash: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{run_hash}"

def main():

    # ==== PARAMETERS ====
    POSSIBLES_AUGMENTATIONS = ["geometry", "dropout", "color_invariance", "color_variation", "blur", "noise", "weather"]

    augmentations = [
        [],
    ]

    model_args = {
        "n_convs": [2],
        "filters": [[32, 64, 128, 256]],
        "dropout": [0.0],
        "quantize": [True],
    }

    training_args = {
        "train_dataset_path": [os.path.join(settings.train_dataset_dir, "simu_v0")],
        "eval_dataset_path": [os.path.join(settings.test_dataset_dir, "MoLane")],
        "seed": [i + 42 for i in range(1)],
        "batch_size": [32],
        "num_epochs": [100],
        "lr": [0.001],
        "step_size": [10],
        "gamma": [0.1],
        "patience": [30],
        "delta": [0.0],
        "train_image_size": [[256, 256]],
        "augmentations": augmentations,
        "weighted_loss": [True],
        "use_amp": [False],
    }

    other_args = {
        "verbose": [False],
    }

    config = {
        "model": model_args,
        "training": training_args,
        "other": other_args,
    }

    # ==== END PARAMETERS ====

    all_run_configs = list(recursive_product(config))
    total_runs = len(all_run_configs)

    logger.info(f"Total configurations to run: {total_runs}")

    completed_runs = 0
    total_elapsed_time = 0.0

    for i, run_config in enumerate(all_run_configs):
        run_hash = make_hash(run_config["model"], run_config["training"])

        if run_already_exists(run_hash):
            logger.info(f"[{i+1}/{total_runs}] Run {run_hash} already exists. Skipping...")
            continue

        # === GIT CHECK ===
        git_metadata = get_git_metadata()
        if git_metadata["dirty"]:
            logger.error(f"Git repository is dirty (uncommitted changes). Run {run_hash} may not be reproducible.")
            exit(1)

        if git_metadata["branch"] != "main":
            logger.error(f"Current branch is '{git_metadata['branch']}', expected 'main'. Please switch to 'main' before launching experiments.")
            exit(1)

        run_dir = format_run_dir(run_hash)
        run_config["other"]["name"] = run_dir
        run_dir_path = os.path.join(settings.experiment_dir, run_dir)
        run_config["other"]["run_hash"] = run_hash
        run_config["other"]["run_dir"] = run_dir_path
        run_config["other"]["git_commit"] = git_metadata["commit"]

        default_cfg = OmegaConf.structured(Config)
        run_cfg = OmegaConf.create(run_config)
        cfg = OmegaConf.merge(default_cfg, run_cfg)
        OmegaConf.resolve(cfg)

        os.makedirs(run_dir_path)

        config_path = os.path.join(run_dir_path, CONFIG_FILE)
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f.name)

        start = time.time()
        launch_training(config_path)
        end = time.time()
        run_time = end - start
        total_elapsed_time += run_time
        completed_runs += 1

        avg_time = total_elapsed_time / completed_runs
        remaining_runs = total_runs - (i + 1)
        est_remaining_time = remaining_runs * avg_time

        elapsed_time_fmt = str(timedelta(seconds=int(run_time)))
        est_remaining_fmt = str(timedelta(seconds=int(est_remaining_time)))

        logger.info(f"[{i+1}/{total_runs}] Run {run_hash} completed in {elapsed_time_fmt}. Estimated time remaining: {est_remaining_fmt}")

if __name__ == "__main__":
    main()
