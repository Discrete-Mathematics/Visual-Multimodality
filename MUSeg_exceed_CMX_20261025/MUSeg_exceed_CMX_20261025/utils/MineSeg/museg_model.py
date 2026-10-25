"""Depth-edge bidirectional fusion network for RGB-D mine segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from museg_data import NUM_CLASSES


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthEncoder(nn.Module):
    """Lightweight encoder on log-depth plus a depth-edge channel."""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(2, 64, 3, 2, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
        )
        self.stage2 = nn.Sequential(ConvBNReLU(128, 128, 3, 2, 1), ConvBNReLU(128, 128, 3, 1, 1))
        self.stage3 = nn.Sequential(ConvBNReLU(128, 256, 3, 2, 1), ConvBNReLU(256, 256, 3, 1, 1))
        self.stage4 = nn.Sequential(ConvBNReLU(256, 256, 3, 2, 1), ConvBNReLU(256, 256, 3, 1, 1))

    def forward(self, x):
        d1 = self.stage1(x)
        d2 = self.stage2(d1)
        d3 = self.stage3(d2)
        d4 = self.stage4(d3)
        return d1, d2, d3, d4


class BiGateFusion(nn.Module):
    """RGB tells depth which regions are reliable; depth edges sharpen RGB."""

    def __init__(self, channels):
        super().__init__()
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.depth_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, depth_feat):
        paired = torch.cat([rgb_feat, depth_feat], dim=1)
        rgb_refined = rgb_feat * (1 + self.rgb_gate(paired))
        depth_refined = depth_feat * self.depth_gate(paired)
        return rgb_feat + self.proj(torch.cat([rgb_refined, depth_refined], dim=1))


class FPNDecoder(nn.Module):
    def __init__(self, num_classes, use_boundary=True):
        super().__init__()
        self.use_boundary = use_boundary
        self.lat4 = nn.Conv2d(256, 128, 1, bias=False)
        self.lat3 = nn.Conv2d(256, 128, 1, bias=False)
        self.lat2 = nn.Conv2d(128, 128, 1, bias=False)
        self.lat1 = nn.Conv2d(128, 128, 1, bias=False)
        self.smooth = ConvBNReLU(128, 128, 3, 1, 1)
        self.classifier = nn.Conv2d(128, num_classes, 1)
        if use_boundary:
            self.boundary = nn.Conv2d(128, 1, 1)

    def forward(self, features):
        f1, f2, f3, f4 = features
        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.smooth(p1)
        boundary = None
        if self.use_boundary:
            boundary = self.boundary(hidden)
            hidden = hidden * (1 + torch.sigmoid(boundary))
        return self.classifier(hidden), boundary


def depth_edge(depth):
    kernel_x = depth.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = depth.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(depth, kernel_x, padding=1)
    grad_y = F.conv2d(depth, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
    scale = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return magnitude / scale


class EdgeSpatialGate(nn.Module):
    """Use depth and its edges to spatially modulate RGB features. No second encoder."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat, depth, edge):
        depth_s = F.interpolate(depth, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        edge_s = F.interpolate(edge, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([feat, depth_s, edge_s], dim=1))
        return feat * (1 + gate)


class DEBFNet(nn.Module):
    """RGB-D segmentor with an ImageNet ResNet-50 backbone.

    mode:
      rgb   — no depth
      early — RGB and depth concatenated at the stem
      debf  — bidirectional gated fusion of RGB and depth-edge features
      edge  — spatial gates driven by depth and depth edges
    """

    def __init__(self, num_classes=NUM_CLASSES, mode="debf", use_boundary=True, pretrained=True):
        super().__init__()
        if mode not in {"rgb", "early", "debf", "edge"}:
            raise ValueError(mode)
        self.mode = mode
        self.use_boundary = use_boundary
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)
        if mode == "early":
            old = backbone.conv1
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv1.weight[:, :3] = old.weight
                conv1.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = conv1
        self.backbone = backbone
        self.depth_encoder = DepthEncoder() if mode == "debf" else None
        self.rgb_proj = nn.ModuleList(
            [
                nn.Conv2d(256, 128, 1, bias=False),
                nn.Conv2d(512, 128, 1, bias=False),
                nn.Conv2d(1024, 256, 1, bias=False),
                nn.Conv2d(2048, 256, 1, bias=False),
            ]
        )
        if mode == "debf":
            self.fuse = nn.ModuleList([BiGateFusion(c) for c in (128, 128, 256, 256)])
        if mode == "edge":
            self.edge_gates = nn.ModuleList([EdgeSpatialGate(c) for c in (128, 128, 256, 256)])
        self.decoder = FPNDecoder(num_classes, use_boundary=use_boundary)
        self.num_classes = num_classes

    def rgb_features(self, x):
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x)
        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)
        return f1, f2, f3, f4

    def forward(self, rgb, depth):
        if self.mode == "early":
            features = self.rgb_features(torch.cat([rgb, depth], dim=1))
        else:
            features = self.rgb_features(rgb)
        features = tuple(proj(feat) for proj, feat in zip(self.rgb_proj, features))
        if self.mode == "debf":
            edge = depth_edge(depth)
            depth_feats = self.depth_encoder(torch.cat([depth, edge], dim=1))
            features = tuple(
                fuse(rgb_feat, depth_feat) for fuse, rgb_feat, depth_feat in zip(self.fuse, features, depth_feats)
            )
        elif self.mode == "edge":
            edge = depth_edge(depth)
            features = tuple(gate(feat, depth, edge) for gate, feat in zip(self.edge_gates, features))
        logits, boundary = self.decoder(features)
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        if boundary is not None:
            boundary = F.interpolate(boundary, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits, boundary
