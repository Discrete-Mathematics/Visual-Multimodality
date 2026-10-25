"""MineGate: reliability-gated RGB-D segmentation for underground mines.

Depth from the Kinect is missing on a large fraction of pixels. MineGate
encodes depth separately, then adds it to the RGB feature only where the
depth is valid and the gate says it is useful. Invalid pixels stay RGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MixVisionTransformerB2
from .labels import NUM_CLASSES


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthEncoder(nn.Module):
    """Light encoder for log-depth, validity and depth gradient."""

    def __init__(self, dims):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, dims[0] // 2, 7, 2, 3),
            ConvBNReLU(dims[0] // 2, dims[0], 3, 2, 1),
            ConvBNReLU(dims[0], dims[0], 3, 1, 1),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(dims[0], dims[1], 3, 2, 1),
            ConvBNReLU(dims[1], dims[1], 3, 1, 1),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(dims[1], dims[2], 3, 2, 1),
            ConvBNReLU(dims[2], dims[2], 3, 1, 1),
        )
        self.stage4 = nn.Sequential(
            ConvBNReLU(dims[2], dims[3], 3, 2, 1),
            ConvBNReLU(dims[3], dims[3], 3, 1, 1),
        )

    def forward(self, depth):
        feat1 = self.stage1(depth)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return [feat1, feat2, feat3, feat4]


class ReliabilityGate(nn.Module):
    """Residual depth injection. The hard validity mask blocks Kinect holes."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 32)
        self.score = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, depth, valid, luma):
        if depth.shape[-2:] != rgb.shape[-2:]:
            depth = F.interpolate(depth, rgb.shape[-2:], mode="bilinear", align_corners=False)
        if valid.shape[-2:] != rgb.shape[-2:]:
            valid = F.interpolate(valid, rgb.shape[-2:], mode="nearest")
            luma = F.interpolate(luma, rgb.shape[-2:], mode="bilinear", align_corners=False)
        raw = torch.sigmoid(self.score(torch.cat([rgb, depth, valid, luma], dim=1)))
        alpha = raw * valid
        fused = rgb + alpha * self.delta(depth)
        return fused, alpha


class FuseDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, embed_dim, 1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for channels in in_channels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.predict = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        target = features[0].shape[-2:]
        projected = []
        for projection, feature in zip(self.projections, features):
            current = projection(feature)
            if current.shape[-2:] != target:
                current = F.interpolate(current, target, mode="bilinear", align_corners=False)
            projected.append(current)
        return self.predict(self.fuse(torch.cat(projected, dim=1)))


class MineGate(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=None):
        super().__init__()
        dims = [64, 128, 320, 512]
        self.rgb_encoder = MixVisionTransformerB2()
        if pretrained:
            self.rgb_encoder.load_pretrained(pretrained)
        self.depth_encoder = DepthEncoder(dims)
        self.gates = nn.ModuleList([ReliabilityGate(channels) for channels in dims])
        self.decoder = FuseDecoder(dims, num_classes)

    def forward(self, rgb, depth, valid, luma):
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        fused = []
        gates = []
        for gate, rgb_feat, depth_feat in zip(self.gates, rgb_features, depth_features):
            merged, alpha = gate(rgb_feat, depth_feat, valid, luma)
            fused.append(merged)
            gates.append(alpha)
        logits = self.decoder(fused)
        logits = F.interpolate(logits, rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits, gates
