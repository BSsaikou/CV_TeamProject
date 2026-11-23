"""2D U-Net for single-channel FLAIR segmentation."""

import torch
from torch import nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        widths = [base_channels * 2**level for level in range(5)]
        self.encoder = nn.ModuleList(
            double_conv(source, target)
            for source, target in zip([1, *widths[:3]], widths[:4])
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(widths[3], widths[4])
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(source, target, kernel_size=2, stride=2)
            for source, target in zip(widths[:0:-1], widths[-2::-1])
        )
        self.decoder = nn.ModuleList(
            double_conv(2 * width, width) for width in reversed(widths[:4])
        )
        self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, image):
        skips = []
        x = image
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for upsample, block, skip in zip(self.upsample, self.decoder, reversed(skips)):
            x = upsample(x, output_size=skip.shape)
            x = block(torch.cat((skip, x), dim=1))
        return self.output(x)
