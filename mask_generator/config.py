##
## EPITECH PROJECT, 2025
## root [SSH: pinkcc]
## File description:
## config_schema
##

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class ModelConfig:
    arch: str = "my_unet"
    in_channels: int = 3
    out_channels: int = 1
    model_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.in_channels > 0, "Input channels must be a positive integer."
        assert self.out_channels > 0, "Output channels must be a positive integer."
        if self.arch == "my_unet":
            required_keys = {"n_convs", "filters", "dropout"}
            missing = required_keys - self.model_args.keys()
            assert not missing, f"Missing required model arguments for 'my_unet': {missing}"
        else:
            for k in ["encoder_name", "encoder_weights"]:
                assert self.model_args.get(k) is not None, f"Missing '{k}' in model_args for architecture {self.arch}"


@dataclass
class TrainingConfig:
    train_dataset_path: str = "/root/MaskGenerator/Datasets/train/simu_v0"
    eval_dataset_path: str = "/root/MaskGenerator/Datasets/test/MoLane"
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

@dataclass
class OtherConfig:
    name: str
    run_hash: str
    run_dir: str
    verbose: bool = False

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    other: OtherConfig
