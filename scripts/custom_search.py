##
## EPITECH PROJECT, 2025
## MaskGenerator [SSH: l4-scaleway]
## File description:
## custom_search
##

import os
import copy
import mask_generator.settings as settings

def merge_dicts(base: dict, overrides: dict) -> dict:
    """
    Recursively merge `overrides` into `base` without modifying `base`.
    Returns a new merged dictionary.
    """
    result = copy.deepcopy(base)

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def custom_search() -> list[dict]:
    """
    Perform a custom search over the provided configuration dictionary.

    Returns:
        list[dict]: A list of dictionaries, each representing a unique combination of parameters.
    """

    # ==== PARAMETERS ====
    POSSIBLES_AUGMENTATIONS = ["geometry", "dropout", "color_invariance", "color_variation", "blur", "noise", "weather"]


    model_args = {
        "arch": "my_unet",
        "model_args": {
            "n_convs": 2,
            "filters": [32, 64, 128, 256],
            "dropout": 0.0
        }
    }
        # {
        #     "arch": "my_unet",
        #     "model_args": {
        #         "n_convs": 2,
        #         "filters": [32, 64, 128],
        #         "dropout": 0.0
        #     }
        # },
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

    train_dataset = [
        {
            "csv": os.path.join(settings.dataset_dir, "simu_v0", "simu.csv"),
            "augmentations": POSSIBLES_AUGMENTATIONS,
        },
        {
            "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_val_target.csv"),
            "augmentations": POSSIBLES_AUGMENTATIONS,
        }
    ]

    eval_dataset = [
        {
            "csv": os.path.join(settings.dataset_dir, "CARLANE", "MoLane", "molane_test.csv"),
        }
    ]

    qat_args = {"enabled": False}

    loss_args = [
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
    ]
        # [
        #     {
        #         "name": "bce",
        #         "weight": 0.5,
        #         "params": {
        #             "pos_weight": True
        #         }
        #     },
        #     {
        #         "name": "dice",
        #         "weight": 0.5,
        #         "params": {
        #             "smooth": 1.0
        #         }
        #     }
        # ],

    training_args = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "seed": 42,
        "batch_size": 32,
        "num_epochs": 100,
        "lr": 0.001,
        "step_size": 10,
        "gamma": 0.1,
        "patience": 30,
        "delta": 0.0,
        "image_size": [128, 384],  # (height, width)
        "use_amp": False,
        "qat": qat_args,
        "loss": loss_args,
    }

    other_args = {
        "verbose": False,
    }

    base_config = {
        "model": model_args,
        "training": training_args,
        "other": other_args,
    }

    overrides_list = [
        {
            "training": {
                "loss": [
                    {
                        "name": "lovasz_hinge",
                        "weight": 1,
                        "params": {
                            "per_image": True,
                        }
                    }
                ]
            }
        },

        {
            "training": {
                "loss": [
                    {
                        "name": "lovasz_hinge",
                        "weight": 1,
                        "params": {
                            "per_image": False,
                        }
                    }
                ]
            }
        },

        {
            "training": {
                "loss": [
                    {
                        "name": "lovasz_hinge",
                        "weight": 0.7,
                        "params": {
                            "per_image": True,
                        }
                    },
                    {
                        "name": "boundary",
                        "weight": 0.3,
                        "params": {
                            "theta0": 3,
                            "theta": 5
                        }
                    }
                ]
            }
        },

        {
            "training": {
                "loss": [
                    {
                        "name": "dice",
                        "weight": 0.7,
                        "params": {
                            "smooth": 1.0
                        }
                    },
                    {
                        "name": "boundary",
                        "weight": 0.3,
                        "params": {
                            "theta0": 3,
                            "theta": 5
                        }
                    }
                ]
            }
        }

        {
            "training": {
                "loss": [
                    {
                        "name": "lovasz_hinge",
                        "weight": 0.6,
                        "params": {
                            "per_image": True,
                        }
                    },
                    {
                        "name": "dice",
                        "weight": 0.4,
                        "params": {
                            "smooth": 1.0
                        }
                    }
                ]
            }
        },

        {
            "training": {
                "loss": [
                    {
                        "name": "lovasz_hinge",
                        "weight": 0.5,
                        "params": {}
                    },
                    {
                        "name": "dice",
                        "weight": 0.3,
                        "params": {
                            "smooth": 1.0
                        }
                    },
                    {
                        "name": "boundary",
                        "weight": 0.2,
                        "params": {
                            "theta0": 3,
                            "theta": 5
                        }
                    }
                ]
            }
        },

    ]

    all_run_configs = [merge_dicts(base_config, override) for override in overrides_list]

    return all_run_configs
