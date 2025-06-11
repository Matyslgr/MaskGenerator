##
## EPITECH PROJECT, 2025
## root [SSH: pinkcc]
## File description:
## config_schema
##

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 1
    filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    n_convs: int = 2
    dropout: float = 0.0
    quantize: bool = False

@dataclass
class TrainingConfig:
    train_dataset_path: str
    eval_dataset_path: str
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 100
    lr: float = 0.001
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 30
    delta: float = 0.0
    train_image_size: Tuple[int, int] = (256, 256)
    augmentations: List[str] = field(default_factory=lambda: [])
    weighted_loss: bool = True
    use_amp: bool = False
    qat: bool = False
    qat_backend: str = "fbgemm"

@dataclass
class OtherConfig:
    name: str
    run_hash: str
    run_dir: str
    git_commit: str
    verbose: bool = False

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    other: OtherConfig
