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
    arch: str = "my_unet"
    in_channels: int = 3
    out_channels: int = 1
    model_args: dict = field(default_factory=lambda: {})

@dataclass
class QATConfig:
    enabled: bool = False
    backend: str = "fbgemm"

@dataclass
class LossConfig:
    name: str = "bce"
    weight: List[float] = field(default_factory=lambda: [1.0])
    params: dict = field(default_factory=lambda: {})

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
    use_amp: bool = False
    qat: QATConfig = field(default_factory=QATConfig)
    loss: List[LossConfig] = field(default_factory=lambda: [LossConfig(name="bce", weight=[1.0])])

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
