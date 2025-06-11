##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## train
##

import argparse
from omegaconf import OmegaConf
import numpy as np
from typing import Tuple

from mask_generator.utils import set_deterministic_behavior, DatasetLoaderFactory
from mask_generator.models.utils import create_model
from mask_generator.config import Config
from mask_generator.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train the model.')

    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file.')
    return parser.parse_args()

def prepare_pairs(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training and testing pairs based on the configuration."""

    train_pairs = DatasetLoaderFactory.get_loader(
        config.training.train_dataset_path,
    ).get_pairs()

    test_pairs = DatasetLoaderFactory.get_loader(
        config.training.eval_dataset_path,
    ).get_pairs()

    print(f"[INFO] Successfully loaded {len(train_pairs)} train pairs of images and masks.")
    print(f"[INFO] Successfully loaded {len(test_pairs)} test pairs of images and masks.")

    return train_pairs, test_pairs

def main():
    args = parse_args()

    try:
        run_cfg = OmegaConf.load(args.config)

        default_cfg = OmegaConf.structured(Config)

        cfg = OmegaConf.merge(default_cfg, run_cfg)
        OmegaConf.resolve(cfg)

        config: Config = OmegaConf.to_object(cfg)

    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        return

    assert isinstance(config, Config), "Configuration must be an instance of Config class."

    set_deterministic_behavior(config.training.seed)

    train_pairs, test_pairs = prepare_pairs(config)

    model, pad_divisor = create_model(config.model)

    trainer = Trainer(config, pad_divisor)
    trainer.fit(model, train_pairs, test_pairs)

if __name__ == "__main__":
    main()
