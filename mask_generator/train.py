##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## train
##

import os
import yaml
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
from mask_generator.quantization_utils import prepare_qat_model, convert_qat_to_quantized, export_to_onnx
from mask_generator.utils import set_deterministic_behavior, load_datasets
from mask_generator.logger import setup_logging

logger = setup_logging(__file__, level=logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train the model.')

    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file.')
    return parser.parse_args()

def save_metadata(run_dir: str, pad_divisor: int):
    metadata_path = os.path.join(run_dir, "metadata.yaml")
    metadata = {"pad_divisor": pad_divisor}

    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

def main():
    args = parse_args()

    try:
        run_cfg = OmegaConf.load(args.config)

        default_cfg = OmegaConf.structured(Config)

        cfg = OmegaConf.merge(default_cfg, run_cfg)
        OmegaConf.resolve(cfg)

        config: Config = OmegaConf.to_object(cfg)

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    assert isinstance(config, Config), "Configuration must be an instance of Config class."

    logger = setup_logging(settings.logger_name, level=logging.level.DEBUG, log_file=os.path.join(config.other.run_dir, settings.logging_filename))

    set_deterministic_behavior(config.training.seed)

    model, pad_divisor = create_model(config.model)
    logger.info(f"Model created with pad_divisor: {pad_divisor}")

    save_metadata(config.other.run_dir, pad_divisor)
    logger.info(f"Saved pad_divisor to metadata.yaml")

    if config.training.qat.enabled:
        model = prepare_qat_model(model, config.training.qat.backend)

    train_dataset, val_dataset, test_dataset = load_datasets(
        config=config,
        pad_divisor=pad_divisor
    )

    logger.info(f"Loaded datasets: {len(train_dataset)} train samples, {len(val_dataset)} val samples, {len(test_dataset)} test samples")

    trainer = Trainer(config)
    model = trainer.fit(model, train_dataset, val_dataset, test_dataset)

    if config.training.qat.enabled:
        model = convert_qat_to_quantized(model)

    input_shape = (1, 3, config.training.train_image_size[0], config.training.train_image_size[1])
    export_to_onnx(model, os.path.join(config.other.run_dir, settings.onnx_filename), input_shape=input_shape)

if __name__ == "__main__":
    main()
