"""MineHHA: frozen MineGate plus an HHA residual that starts at zero.

The 0.4763 checkpoint is the full MineGate forward (RGB encoder, depth
encoder, reliability gate, decoder). Feeding the decoder raw RGB features
scores about 0.42, so this model keeps that forward intact and frozen.

A second MiT-B2 reads disparity, camera-up height, surface angle and the
Kinect validity mask. Its logits are multiplied by tanh(open), which starts
at 0, and the last convolution starts at 0. The sum therefore matches
MineGate until the residual has something useful to add.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MixVisionTransformerB2
from .labels import NUM_CLASSES
from .gate import MineGate


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


class FeatureInject(nn.Module):
    """Add HHA features onto a fused RGB-D map. The last convolution is zero."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.up = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.dw.weight)
        with torch.no_grad():
            self.dw.weight[:, 0, 1, 1] = 1.0
        nn.init.zeros_(self.up.weight)

    def forward(self, feature):
        return self.up(self.dw(self.down(feature)))


class MineHHA(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        dims = [64, 128, 320, 512]
        self.base = MineGate(num_classes=num_classes)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.inject = nn.ModuleList(FeatureInject(channels) for channels in dims)
        # 4 channels -> the 3-channel stem. Starts as [I | 0], so validity is ignored.
        self.stem = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            weight = torch.zeros(3, 4, 1, 1)
            weight[:, :3] = torch.eye(3).view(3, 3, 1, 1)
            self.stem.weight.copy_(weight)
        self.geo_encoder = MixVisionTransformerB2()
        self.geo_decoder = FuseDecoder(dims, num_classes, embed_dim=256)
        nn.init.zeros_(self.geo_decoder.predict.weight)
        nn.init.zeros_(self.geo_decoder.predict.bias)
        self.open_raw = nn.Parameter(torch.zeros(()))
        # 0 keeps the sum equal to MineGate. The trainer raises it after epoch 0.
        self.register_buffer("mix", torch.zeros((), dtype=torch.float32), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def set_mix(self, epoch):
        if epoch < 1:
            value = 0.0
        else:
            value = min(1.0, float(epoch) / 4.0)
        self.mix.fill_(value)

    def residual_scale(self):
        scale = torch.tanh(self.open_raw).detach() * self.mix.detach()
        return float(scale)

    def head_magnitude(self):
        return float(self.geo_decoder.predict.weight.detach().abs().mean())

    def inject_magnitude(self):
        weight = torch.cat([block.up.weight.detach().flatten() for block in self.inject])
        return float(weight.abs().mean())

    def unfreeze_decoder(self):
        for parameter in self.base.decoder.parameters():
            parameter.requires_grad = True

    def load_minegate(self, path):
        with Path(path).open("rb") as handle:
            try:
                checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(handle, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = self.base.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "MineGate mismatch missing=%s unexpected=%s" % (list(missing)[:8], list(unexpected)[:8])
            )
        self.geo_encoder.load_state_dict(self.base.rgb_encoder.state_dict())
        nn.init.zeros_(self.geo_decoder.predict.weight)
        nn.init.zeros_(self.geo_decoder.predict.bias)
        with torch.no_grad():
            self.open_raw.zero_()
        print("loaded full MineGate from %s" % path, flush=True)

    def load_trained(self, path):
        with Path(path).open("rb") as handle:
            try:
                checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(handle, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if not key.startswith("inject.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                "MineHHA mismatch missing=%s unexpected=%s" % (bad_missing[:8], list(unexpected)[:8])
            )
        print(
            "loaded trained MineHHA from %s, open %.4f"
            % (path, float(self.open_raw.detach())),
            flush=True,
        )

    def _fused_features(self, rgb, depth, valid, luma):
        with torch.no_grad():
            rgb_features = self.base.rgb_encoder(rgb)
            depth_features = self.base.depth_encoder(depth)
            fused = []
            for gate, rgb_feature, depth_feature in zip(self.base.gates, rgb_features, depth_features):
                fused_feature, _gate_value = gate(rgb_feature, depth_feature, valid, luma)
                fused.append(fused_feature)
        return [feature.detach() for feature in fused]

    def forward(self, rgb, depth, valid, luma, hha):
        fused = self._fused_features(rgb, depth, valid, luma)
        features = self.geo_encoder(self.stem(hha))
        mixed = []
        for feature, hha_feature, inject in zip(fused, features, self.inject):
            if hha_feature.shape[-2:] != feature.shape[-2:]:
                hha_feature = F.interpolate(hha_feature, feature.shape[-2:], mode="bilinear", align_corners=False)
            mixed.append(feature + inject(hha_feature))
        base_logits = self.base.decoder(mixed)
        base_logits = F.interpolate(base_logits, rgb.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.geo_decoder(features)
        delta = F.interpolate(delta, rgb.shape[-2:], mode="bilinear", align_corners=False)
        scale = torch.tanh(self.open_raw) * self.mix
        return base_logits + scale * delta, delta
