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
            print(f"Resizing: {x.shape[2:]} to {skip.shape[2:]}")
            x = TF.resize(x, size=skip.shape[2:])
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)

class MyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=[32, 64, 128, 256], n_convs=2, dropout=0.0):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder blocks
        input_channels = in_channels
        for filters in num_filters:
            self.encoders.append(EncoderBlock(input_channels, filters, n_convs=n_convs, dropout=dropout))
            input_channels = filters

        # --- Bottleneck
        self.bottleneck = ConvBlock(num_filters[-1], num_filters[-1]*2, n_convs=n_convs, dropout=dropout)
        # --- Decoder blocks (reverse)
        rev_num_filters = num_filters[::-1]
        input_channels = num_filters[-1]*2
        for filters in rev_num_filters:
            self.decoders.append(DecoderBlock(input_channels, filters, n_convs=n_convs, dropout=dropout))
            input_channels = filters

        self.final_conv = nn.Conv2d(num_filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]
        div = 2 ** len(self.encoders)
        if h % div != 0 or w % div != 0:
            raise ValueError(f"Input size ({h}, {w}) must be divisible by {div}")

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skips[idx])

        return self.final_conv(x)

if __name__ == "__main__":
    import torchsummary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyUNet(in_channels=3, out_channels=1, num_filters=[32, 64, 128, 256], n_convs=2, dropout=0.0).to(device)
    print(model)

    # Print the model summary
    torchsummary.summary(model, (3, 256, 256))  # Input shape: (channels, height, width)

    # Test the model with a random input
    x = torch.randn(1, 3, 256, 256).to(device)  # Batch size of 1, 3 channels, 256x256 image
    output = model(x)
    print("Output shape:", output.shape)  # Should be (1, 1, 256, 256) for a segmentation task
