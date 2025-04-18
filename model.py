##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## model
##

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, padding=1, dropout=0.0):

        super().__init__()
        layers = []
        for _ in range(n_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, padding=1, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, n_convs, padding, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2, padding=1, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, n_convs, padding, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = TF.resize(x, size=skip.shape[2:])
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)

class MyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], n_convs=2, dropout=0.0):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder blocks
        channels = in_channels
        for feature in features:
            self.encoders.append(EncoderBlock(channels, feature, n_convs=n_convs, dropout=dropout))
            channels = feature

        # --- Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, n_convs=n_convs, dropout=dropout)
        # --- Decoder blocks (reverse)
        rev_features = features[::-1]
        channels = features[-1]*2
        for feature in rev_features:
            self.decoders.append(DecoderBlock(channels, feature, n_convs=n_convs, dropout=dropout))
            channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skips[idx])

        return self.final_conv(x)
