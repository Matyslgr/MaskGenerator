##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## train
##

import argparse
from omegaconf import OmegaConf
import numpy as np

from utils import set_deterministic_behavior, DatasetLoaderFactory
from MaskGenerator.mask_generator.models.my_unet import create_model
from typing import Tuple
from mask_generator.config import Config, ModelConfig, TrainingConfig, OtherConfig
from trainer import Trainer

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

    run_cfg = OmegaConf.load(args.config)

    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, run_cfg)
    OmegaConf.resolve(cfg)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    config = Config(
        model=ModelConfig(**cfg_dict['model']),
        training=TrainingConfig(**cfg_dict['training']),
        other=OtherConfig(**cfg_dict['other'])
    )
    if not isinstance(config, Config):
        raise TypeError("Configuration must be an instance of Config class.")

    set_deterministic_behavior(config.training.seed)

    train_pairs, test_pairs = prepare_pairs(config)

    model = create_model(config.model)

    trainer = Trainer(config)
    trainer.fit(model, train_pairs, test_pairs)

if __name__ == "__main__":
    main()
