"""DEBFNet and SelectiveMine on a shared MiT RGB-D backbone.

DEBFNet fuses each scale with depth-edge bidirectional gates. Coarse scales
also exchange information with bidirectional cross-attention. SelectiveMine
keeps an RGB appearance head and the DEBF head, then mixes them with a
spatial router that can fall back to RGB when depth is absent.
"""

import torch
from torch import nn
import torch.nn.functional as F

from mit_loader import encode, load_encoder

NUM_CLASSES = 15
CHANNELS = (64, 128, 320, 512)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BiCrossAttention(nn.Module):
    """Bidirectional cross-attention with a zero-initialized residual."""

    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(channels)
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.q_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_depth = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_depth = nn.Conv2d(channels, channels, 1, bias=False)
        self.q_depth = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_depth = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.proj_rgb.weight)
        nn.init.zeros_(self.proj_depth.weight)

    def _attend(self, query, key, value):
        batch, channels, height, width = query.shape
        head_dim = channels // self.heads

        def pack(tensor):
            tensor = tensor.view(batch, self.heads, head_dim, height * width)
            return tensor.transpose(2, 3)

        query, key, value = pack(query), pack(key), pack(value)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, value.float())
        mixed = mixed.transpose(2, 3).contiguous().view(batch, channels, height, width)
        return mixed.to(dtype=query.dtype)

    def forward(self, rgb, depth):
        rgb = rgb + self.proj_rgb(self._attend(self.q_rgb(rgb), self.k_depth(depth), self.v_depth(depth)))
        depth = depth + self.proj_depth(self._attend(self.q_depth(depth), self.k_rgb(rgb), self.v_rgb(rgb)))
        return rgb, depth


class DepthEdgeBidirectionalGate(nn.Module):
    """RGB and depth reweight each other; depth edges enter the depth stream."""

    def __init__(self, channels, cross_attention=False):
        super().__init__()
        self.cross = BiCrossAttention(channels) if cross_attention else None
        self.edge_proj = nn.Conv2d(1, channels, 1, bias=False)
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
        self.out = nn.Conv2d(channels * 2, channels, 1, bias=False)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, rgb, depth, edge, valid):
        edge = F.interpolate(edge, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        valid = F.interpolate(valid, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        depth = (depth + self.edge_proj(edge)) * valid
        if self.cross is not None and rgb.shape[-1] * rgb.shape[-2] <= 4096:
            rgb, depth = self.cross(rgb, depth)
        paired = torch.cat([rgb, depth], dim=1)
        rgb_refined = rgb * (1.0 + self.rgb_gate(paired))
        depth_refined = depth * self.depth_gate(paired)
        return rgb + self.out(torch.cat([rgb_refined, depth_refined], dim=1))


class SegFormerHead(nn.Module):
    def __init__(self, num_classes, embed=256, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList(nn.Conv2d(channels, embed, 1) for channels in CHANNELS)
        self.fuse = nn.Sequential(
            nn.Conv2d(embed * 4, embed, 1, bias=False),
            nn.BatchNorm2d(embed),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(embed, num_classes, 1)
        self.boundary = nn.Conv2d(embed, 1, 1)

    def forward(self, features):
        target = features[0].shape[-2:]
        projected = []
        for layer, feature in zip(self.proj, features):
            projected.append(F.interpolate(layer(feature), size=target, mode="bilinear", align_corners=False))
        hidden = self.fuse(torch.cat(projected, dim=1))
        return self.classifier(hidden), self.boundary(hidden)


class LightDepthEncoder(nn.Module):
    """Stride-32 convolutional depth encoder. A second MiT overfits sparse Kinect depth."""

    def __init__(self):
        super().__init__()
        dims = CHANNELS

        def stage(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stages = nn.ModuleList(
            [
                stage(3, dims[0], 4),
                stage(dims[0], dims[1], 2),
                stage(dims[1], dims[2], 2),
                stage(dims[2], dims[3], 2),
            ]
        )

    def forward(self, depth):
        features = []
        hidden = depth
        for stage in self.stages:
            hidden = stage(hidden)
            features.append(hidden)
        return features


class MineSegmentor(nn.Module):
    def __init__(self, variant, weight_path, mode, num_classes=NUM_CLASSES):
        super().__init__()
        if mode not in {"debf", "selective"}:
            raise ValueError(mode)
        self.mode = mode
        self.num_classes = num_classes
        self.rgb_encoder = load_encoder(variant, weight_path)
        self.depth_encoder = LightDepthEncoder()
        # Coarse maps are small enough for cross-attention. Fine maps stay convolutional.
        self.fuse = nn.ModuleList(
            DepthEdgeBidirectionalGate(channels, cross_attention=(index == 3))
            for index, channels in enumerate(CHANNELS)
        )
        self.decoder = SegFormerHead(num_classes)
        if mode == "selective":
            self.appearance = SegFormerHead(num_classes)
            self.router = nn.Sequential(
                nn.Conv2d(num_classes * 3 + 5, 64, 1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1, groups=64),
                nn.GELU(),
                nn.Conv2d(64, 1, 1),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.constant_(self.router[-1].bias, 1.5)

    def forward(self, rgb, depth, valid, edge):
        rgb_features = encode(self.rgb_encoder, rgb)
        depth_features = self.depth_encoder(depth)
        fused = [
            gate(rgb_feat, depth_feat, edge, valid)
            for gate, rgb_feat, depth_feat in zip(self.fuse, rgb_features, depth_features)
        ]
        fused_logits, boundary = self.decoder(fused)
        size = rgb.shape[-2:]
        fused_logits = F.interpolate(fused_logits, size=size, mode="bilinear", align_corners=False)
        boundary = F.interpolate(boundary, size=size, mode="bilinear", align_corners=False)
        if self.mode == "debf":
            return {"logits": fused_logits, "boundary": boundary, "log_space": False}

        rgb_logits, _ = self.appearance(rgb_features)
        rgb_logits = F.interpolate(rgb_logits, size=size, mode="bilinear", align_corners=False)
        present = (valid.flatten(1).sum(1) > 1).view(-1, 1, 1, 1)
        log_rgb = F.log_softmax(rgb_logits.float(), dim=1)
        log_fused = F.log_softmax(fused_logits.float(), dim=1)
        rgb_p = log_rgb.detach().exp()
        fused_p = log_fused.detach().exp()
        entropy_rgb = -(rgb_p * log_rgb.detach()).sum(1, keepdim=True)
        entropy_fused = -(fused_p * log_fused.detach()).sum(1, keepdim=True)
        luma = rgb.detach().mean(1, keepdim=True).float()
        evidence = torch.cat(
            [
                rgb_p,
                fused_p,
                (rgb_p - fused_p).abs(),
                entropy_rgb,
                entropy_fused,
                luma,
                valid.detach().float(),
                edge.detach().float(),
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.router(evidence).float())
        gate = torch.where(present, gate, torch.zeros_like(gate)).clamp(1e-4, 1.0 - 1e-4)
        log_mix = torch.logaddexp(log_rgb + torch.log(1.0 - gate), log_fused + torch.log(gate))
        return {
            "logits": log_mix,
            "rgb_logits": rgb_logits,
            "fused_logits": fused_logits,
            "boundary": boundary,
            "gate": gate,
            "log_space": True,
        }
