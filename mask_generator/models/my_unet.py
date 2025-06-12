##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## model
##

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.quantization as tq
from collections import OrderedDict

from mask_generator.qat_utils import create_activation_fake_quant

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, padding=1, dropout=0.0, inplace=False):
        super().__init__()
        layers = []
        self.modules_names = []

        if n_convs < 1:
            raise ValueError("n_convs must be at least 1")

        for i in range(n_convs):
            conv_name = f"conv_{i+1}"
            bn_name = f"bn_{i+1}"
            relu_name = f"relu_{i+1}"

            layers.append((conv_name, nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False)))
            layers.append((bn_name, nn.BatchNorm2d(out_channels)))
            layers.append((relu_name, nn.ReLU(inplace=inplace)))

            self.modules_names.append((conv_name, bn_name, relu_name))

            in_channels = out_channels

        if dropout > 0:
            layers.append(("dropout", nn.Dropout(dropout)))

        self.block = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.block(x)

    def fuse_model(self):
        """
        Fuses the Conv2d, BatchNorm2d, and ReLU layers for quantization.
        """
        for conv_name, bn_name, relu_name in self.modules_names:
            if hasattr(self.block, conv_name) and hasattr(self.block, bn_name) and hasattr(self.block, relu_name):
                torch.quantization.fuse_modules(self.block, [conv_name, bn_name, relu_name], inplace=True)
            else:
                raise ValueError(f"Cannot fuse {conv_name}, {bn_name}, {relu_name} - one of the modules is missing.")

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, padding=1, dropout=0.0, inplace=False):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, n_convs, padding, dropout, inplace)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip

    def fuse_model(self):
        self.conv.fuse_model()
        # Note: Pooling layers are not fused, as they do not have learnable parameters.

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, padding=1, dropout=0.0, inplace=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, n_convs, padding, dropout, inplace)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)

    def fuse_model(self):
        self.conv.fuse_model()
        # Note: Transpose convolution layers are not fused, as they do not have learnable parameters.

class MyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[32, 64, 128, 256], n_convs=2, dropout=0.0, quantize=False, inplace=False):
        super().__init__()

        self.quantize = quantize

        self.input_fake_quant = create_activation_fake_quant() if quantize else nn.Identity()
        self.encoder_fake_quant = create_activation_fake_quant() if quantize else nn.Identity()
        self.decoder_fake_quant = create_activation_fake_quant() if quantize else nn.Identity()
        self.output_fake_quant = create_activation_fake_quant() if quantize else nn.Identity()

        self.quant = tq.QuantStub() if quantize else nn.Identity()
        self.dequant = tq.DeQuantStub() if quantize else nn.Identity()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder blocks
        input_channels = in_channels
        for f in filters:
            self.encoders.append(EncoderBlock(input_channels, f, n_convs=n_convs, dropout=dropout, inplace=inplace))
            input_channels = f

        # --- Bottleneck
        self.bottleneck = ConvBlock(filters[-1], filters[-1]*2, n_convs=n_convs, dropout=dropout, inplace=inplace)
        # --- Decoder blocks (reverse)
        rev_num_filters = filters[::-1]
        input_channels = filters[-1]*2
        for f in rev_num_filters:
            self.decoders.append(DecoderBlock(input_channels, f, n_convs=n_convs, dropout=dropout, inplace=inplace))
            input_channels = f

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.quant(x)
        x = self.input_fake_quant(x)

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip = self.encoder_fake_quant(skip)
            skips.append(skip)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skips[idx])
            x = self.decoder_fake_quant(x)

        x = self.final_conv(x)

        x = self.output_fake_quant(x)
        x = self.dequant(x)

        return x

    def fuse_model(self):
        """
        Fuses all Conv2d, BatchNorm2d, and ReLU layers in the model for quantization.
        """
        for encoder in self.encoders:
            encoder.fuse_model()
        self.bottleneck.fuse_model()
        for decoder in self.decoders:
            decoder.fuse_model()
