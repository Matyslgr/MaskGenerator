##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## grid_search
##

import itertools
import os
import mask_generator.settings as settings


def dict_product(d: dict):
    """Generate all combinations of dictionary values."""
    keys, values = zip(*d.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def recursive_product(d: dict):
    groups = d.keys()
    groups_product = {}
    for g in groups:
        if isinstance(d[g], list) and all(isinstance(item, dict) for item in d[g]):
            groups_product[g] = d[g]
        else:
            groups_product[g] = list(dict_product(d[g]))

    for combined in itertools.product(*groups_product.values()):
        yield dict(zip(groups, combined))

def grid_search() -> list[dict]:
    """
    Perform a grid search over the provided configuration dictionary.

    Args:
        config (dict): Configuration dictionary with parameters to search.

    Returns:
        list[dict]: A list of dictionaries, each representing a unique combination of parameters.
    """

    # ==== PARAMETERS ====
    POSSIBLES_AUGMENTATIONS = ["geometry", "dropout", "color_invariance", "color_variation", "blur", "noise", "weather"]


    model_args = [
        {
            "arch": "my_unet",
            "model_args": {
                "n_convs": 2,
                "filters": [32, 64, 128, 256],
                "dropout": 0.0
            }
        },
        {
            "arch": "my_unet",
            "model_args": {
                "n_convs": 2,
                "filters": [32, 64, 128],
                "dropout": 0.0
            }
        },
        # {
        #     "arch": "my_unet",
        #     "model_args": {
        #         "n_convs": 2,
        #         "filters": [32, 64, 128, 256, 512],
        #         "dropout": 0.0
        #     }
        # }
        # {
        #     "arch": "unet",
        #     "model_args": {
        #         "encoder_name": "mobilenet_v2",
        #         "encoder_weights": "imagenet",
        #         "activation": None
        #     }
        # },
        # {
        #     "arch": "unet",
        #     "model_args": {
        #         "encoder_name": "mobilenet_v2",
        #         "encoder_weights": "imagenet",
        #         "decoder_attention_type": "scse",
        #         "activation": None
        #     }
        # },
        # {
        #     "arch": "unet",
        #     "model_args": {
        #         "encoder_name": "efficientnet-b0",
        #         "encoder_weights": "imagenet",
        #         "decoder_attention_type": "scse",
        #         "activation": None
        #     }
        # },
        # {
        #     "arch": "fpn",
        #     "model_args": {
        #         "encoder_name": "mobilenet_v2",
        #         "encoder_weights": "imagenet",
        #         "decoder_attention_type": "scse",
        #         "activation": None
        #     }
        # },
        # {
        #     "arch": "linknet",
        #     "model_args": {
        #         "encoder_name": "resnet18",
        #         "encoder_weights": "imagenet",
        #         "decoder_attention_type": "scse",
        #         "activation": None
        #     }
        # },
        # {
        #     "arch": "deep_lab_v3",
        #     "model_args": {
        #         "encoder_name": "mobilenet_v2",
        #         "encoder_weights": "imagenet",
        #         "activation": None
        #     }
        # }
    ]

    train_dataset = [
        [
            {
            "csv": os.path.join(settings.dataset_dir, "simu_v0", "simu.csv"),
            "augmentations": POSSIBLES_AUGMENTATIONS,
            },
            {
                "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_val_target.csv"),
                "augmentations": POSSIBLES_AUGMENTATIONS,
            }
        ],
        [
            {
                "csv": os.path.join(settings.dataset_dir, "simu_v0", "simu.csv"),
                "augmentations": POSSIBLES_AUGMENTATIONS,
            },
            {
                "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_val_target.csv"),
                "augmentations": [],
            }
        ],
        [
            {
                "csv": os.path.join(settings.dataset_dir, "simu_v0", "simu.csv"),
                "augmentations": POSSIBLES_AUGMENTATIONS,
            },
            {
                "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_val_target.csv"),
                "augmentations": POSSIBLES_AUGMENTATIONS,
            },
            {
                "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_val_source.csv"),
                "augmentations": POSSIBLES_AUGMENTATIONS,
            }
        ]
    ]

    eval_dataset = [
        [
            {
                "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_test.csv"),
            }
        ]
    ]

    qat_args = [
        {"enabled": False}
    ]

    loss_args = [
        [
            {
                "name": "bce",
                "weight": 0.5,
                "params": {
                    "pos_weight": True
                }
            },
            {
                "name": "dice",
                "weight": 0.5,
                "params": {
                    "smooth": 1.0
                }
            }
        ],
        [
            {
                "name": "bce",
                "weight": 0.3,
                "params": {
                    "pos_weight": True
                }
            },
            {
                "name": "dice",
                "weight": 0.7,
                "params": {
                    "smooth": 1.0
                }
            }
        ],
        # [
        #     {
        #         "name": "bce",
        #         "weight": 0.15,
        #         "params": {
        #             "pos_weight": True
        #         }
        #     },
        #     {
        #         "name": "dice",
        #         "weight": 0.85,
        #         "params": {
        #             "smooth": 1.0
        #         }
        #     }
        # ],
    ]

    training_args = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "seed": [i + 42 for i in range(1)],
        "batch_size": [32],
        "num_epochs": [100],
        "lr": [0.001],
        "step_size": [10],
        "gamma": [0.1],
        "patience": [30],
        "delta": [0.0],
        "image_size": [[128, 384]],  # (height, width)
        "use_amp": [False],
        "qat": qat_args,
        "loss": loss_args,
    }

    other_args = {
        "verbose": [False],
    }

    config = {
        "model": model_args,
        "training": training_args,
        "other": other_args,
    }

    return list(recursive_product(config))
