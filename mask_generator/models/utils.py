##
## EPITECH PROJECT, 2025
## root [SSH: robocar-scaleway]
## File description:
## utils
##

import torch
import torch.nn as nn
# import segmentation_models_pytorch as smp
from mask_generator.config import ModelConfig
from mask_generator.models.my_unet import MyUNet

# def get_smp_pad_divisor(model: nn.Module, input_size=(256, 256)) -> int:
#     with torch.no_grad():
#         x = torch.randn(1, 3, *input_size)
#         features = model.encoder(x)
#         h_in, w_in = x.shape[-2:]
#         h_out, w_out = features[-1].shape[-2:]
#         stride_h = h_in // h_out
#         stride_w = w_in // w_out
#         assert stride_h == stride_w, "Non-uniform stride"
#         return stride_h

# def create_model(model_config: ModelConfig) -> nn.Module:
#     """Create a segmentation model based on the provided configuration.
#     Args:
#         model_config (ModelConfig): Configuration for the model.
#     Returns:
#         nn.Module: The created segmentation model.
#         int: The padding divisor for the model.
#     """
#     if model_config.arch == "my_unet":
#         model = MyUNet(
#             in_channels=model_config.in_channels,
#             out_channels=model_config.out_channels,
#             **model_config.model_args
#         )
#         pad_divisor = 2 ** len(model_config.model_args.get('filters', [32, 64, 128, 256]))
#     else:
#         model = smp.create_model(
#             arch=model_config.arch,
#             in_channels=model_config.in_channels,
#             classes=model_config.out_channels,
#             **model_config.model_args
#         )
#         pad_divisor = get_smp_pad_divisor(model)

#     if not isinstance(model, nn.Module):
#         raise TypeError(f"Model must be an instance of nn.Module, got {type(model)}")

#     return model, pad_divisor

def create_model(model_config: ModelConfig) -> nn.Module:
    """Create a segmentation model based on the provided configuration.

    Args:
        model_config (ModelConfig): Configuration for the model.

    Returns:
        nn.Module: The created segmentation model.
    """
    model = MyUNet(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        filters=model_config.filters,
        n_convs=model_config.n_convs,
        dropout=model_config.dropout,
        quantize=model_config.quantize
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"Model must be an instance of nn.Module, got {type(model)}")

    pad_divisor = 2 ** len(model_config.filters)
    return model, pad_divisor
