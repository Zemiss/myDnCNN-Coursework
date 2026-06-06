"""U-Net denoising model used by the DnCNN course project."""

import jittor as jt
from jittor import nn


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def execute(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Down-sampling block: max-pooling followed by double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def execute(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-sampling block with a skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def execute(self, x1, x2):
        x1 = self.up(x1)
        x = jt.concat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def execute(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """A compact U-Net for blind grayscale image denoising.

    The input contains two channels: the noisy grayscale image and a noise-level
    map, following the same conditioning idea as FFDNet. The network predicts
    the residual noise, and the clean image is recovered by subtraction.
    """

    def __init__(self, channels=1):
        super().__init__()
        self.inc = DoubleConv(channels + 1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, channels)

    def execute(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
