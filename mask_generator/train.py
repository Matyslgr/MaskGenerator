##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## train
##

import os
import logging
import argparse
import numpy as np
from typing import Tuple
from omegaconf import OmegaConf

from mask_generator.config import Config
from mask_generator.trainer import Trainer
import mask_generator.settings as settings
from mask_generator.logger import setup_logging
from mask_generator.models.utils import create_model
from mask_generator.qat_utils import prepare_qat_model, convert_qat_to_quantized, export_to_onnx
from mask_generator.utils import set_deterministic_behavior, DatasetLoaderFactory

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

    logger = setup_logging(settings.logger_name, log_file=os.path.join(config.other.run_dir, settings.logging_filename))

    set_deterministic_behavior(config.training.seed)

    train_pairs, test_pairs = prepare_pairs(config)

    logger.info(f"Loaded {len(train_pairs)} training pairs and {len(test_pairs)} testing pairs.")

    model, pad_divisor = create_model(config.model)
    logger.info(f"Model created with pad_divisor: {pad_divisor}")

    if config.training.qat:
        model = prepare_qat_model(model, config.training.qat_backend)

    trainer = Trainer(config, pad_divisor)
    model = trainer.fit(model, train_pairs, test_pairs)

    if config.training.qat:
        model = convert_qat_to_quantized(model)
    input_shape = (1, 3, config.training.train_image_size[0], config.training.train_image_size[1])
    export_to_onnx(model, os.path.join(config.other.run_dir, settings.onnx_filename), input_shape=input_shape)

if __name__ == "__main__":
    main()
