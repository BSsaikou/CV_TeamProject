"""DeepLabv3+ for single-channel FLAIR segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=kernel_size // 2, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


def separable_conv(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=dilation,
                  dilation=dilation, groups=in_channels, bias=False),
        nn.GroupNorm(8, in_channels),
        nn.ReLU(inplace=True),
        conv_block(in_channels, out_channels, kernel_size=1),
    )


def resize_like(features, reference):
    return F.interpolate(features, size=reference.shape[-2:],
                         mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            conv_block(in_channels, out_channels, kernel_size=1),
            *(separable_conv(in_channels, out_channels, rate)
              for rate in (6, 12, 18)),
        ])
        # GroupNorm also supports a single sample after global pooling.
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_block(in_channels, out_channels, kernel_size=1),
        )
        self.project = conv_block(5 * out_channels, out_channels, kernel_size=1)

    def forward(self, features):
        branches = [branch(features) for branch in self.branches]
        branches.append(resize_like(self.image_pool(features), features))
        return self.project(torch.cat(branches, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            conv_block(1, 32, stride=2),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
        )
        self.encoder = nn.Sequential(
            conv_block(64, 128, stride=2),
            conv_block(128, 128),
            conv_block(128, 256, stride=2),
            separable_conv(256, 256, dilation=2),
        )
        self.aspp = ASPP(256, 128)
        self.low_level_projection = conv_block(64, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            separable_conv(128 + 48, 128),
            separable_conv(128, 128),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, image):
        low_level = self.stem(image)
        context = self.aspp(self.encoder(low_level))
        low_level = self.low_level_projection(low_level)
        fused = torch.cat((resize_like(context, low_level), low_level), dim=1)
        return resize_like(self.decoder(fused), image)
