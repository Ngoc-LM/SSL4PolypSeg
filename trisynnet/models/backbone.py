"""ResNet34-UNet backbone used as the segmentation network for both the
student and teacher branches of the Mean Teacher framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """Standard ResNet34 encoder with a symmetric U-Net decoder."""

    def __init__(self, num_classes=1, dropout=0.1):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        self.bridge = ConvBlock(512, 512)
        self.dec5 = DecoderBlock(512, 512, 256)
        self.dec4 = DecoderBlock(256, 256, 128)
        self.dec3 = DecoderBlock(128, 128, 64)
        self.dec2 = DecoderBlock(64, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.final_head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e_pool = self.maxpool(e1)
        e2 = self.enc2(e_pool)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bridge(e5)
        d5 = self.dec5(b, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.final_head(d1)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        return out


class ResNet34UNet_SSL(ResNet34UNet):
    """Adds a feature-extraction head (global-pooled bottleneck embedding)
    used by SAFPM's memory bank and by D-BioMix's mixing decisions."""

    def extract_features(self, x):
        e1 = self.enc1(x)
        e_pool = self.maxpool(e1)
        e2 = self.enc2(e_pool)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)  # [B, 512, H/32, W/32]

        features = F.adaptive_avg_pool2d(e5, (1, 1))
        return features.flatten(1)
