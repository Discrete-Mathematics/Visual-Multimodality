"""Observation-supported geometry relay; a research hypothesis, not a benchmark result.

The inherited weights preserve the existing MineHHA checkpoint. New residual
adapters start at zero. The logits/inject/relay arms share the same backbone.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .legacy.hha import MineHHA


def shift(x, dy, dx):
    """Neighbour x[y+dy,x+dx], with zero padding and no circular wrap."""
    h, w = x.shape[-2:]
    pad = max(abs(dy), abs(dx))
    x = F.pad(x, (pad, pad, pad, pad))
    return x[..., pad + dy:pad + dy + h, pad + dx:pad + dx + w]


class GeometryRelay(nn.Module):
    """Separate measured-support and transported evidence before zero-init fusion.

    Base RGB-D fused affinities select neighbours; only cells with observed depth
    contribute. Support mass and predicted reconstruction quality limit transport.
    These are feature-level support indicators, not calibrated sensor uncertainty.
    """

    def __init__(self, channels, hidden=32):
        super().__init__()
        self.offsets = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)]
        self.offsets += [(y, x) for y in (-2, 0, 2) for x in (-2, 0, 2)
                         if (y, x) != (0, 0)]
        self.query = nn.Conv2d(channels, hidden, 1, bias=False)
        self.key = nn.Conv2d(channels, hidden, 1, bias=False)
        self.value = nn.Conv2d(channels, hidden, 1, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(4.0)))
        self.distance_cost = nn.Parameter(torch.tensor(-1.0))
        self.depth_cost = nn.Parameter(torch.tensor(1.0))
        self.quality = nn.Sequential(nn.Conv2d(hidden * 2 + 3, hidden, 1),
                                     nn.GELU(), nn.Conv2d(hidden, 1, 1))
        self.recover = nn.Conv2d(hidden, channels, 1, bias=False)
        self.up = nn.Conv2d(hidden * 2, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, rgb_feature, geo_feature, valid, luma, log_depth=None):
        size = rgb_feature.shape[-2:]
        # Area coverage preserves sparse source evidence better than nearest masks.
        coverage = F.adaptive_avg_pool2d(valid.float(), size)
        brightness = F.adaptive_avg_pool2d(luma.float(), size)
        mean_depth = None
        if log_depth is not None:
            mean_depth = F.adaptive_avg_pool2d(log_depth.float() * valid.float(), size) / coverage.clamp_min(1e-6)
        query = F.normalize(self.query(rgb_feature).float(), dim=1, eps=1e-6)
        key = F.normalize(self.key(rgb_feature).float(), dim=1, eps=1e-6)
        value = self.value(geo_feature)
        temperature = self.log_temperature.exp().clamp(max=20.0)
        cost = F.softplus(self.distance_cost)
        scores = []
        available = []
        for dy, dx in self.offsets:
            score = ((query * shift(key, dy, dx)).sum(1, keepdim=True)
                     * temperature - cost * math.hypot(dy, dx))
            if mean_depth is not None:
                both_seen = coverage * shift(coverage, dy, dx)
                score = score - 4.0 * F.softplus(self.depth_cost) * both_seen * (mean_depth - shift(mean_depth, dy, dx)).abs()
            scores.append(score)
            available.append(shift(coverage, dy, dx))
        affinity = torch.softmax(torch.cat(scores, 1), dim=1)
        supported = affinity * torch.cat(available, 1)
        support = supported.sum(1, keepdim=True)
        weights = supported / support.clamp_min(1e-6)
        transported = torch.zeros_like(value)
        for index, (dy, dx) in enumerate(self.offsets):
            transported = transported + weights[:, index:index + 1].to(value.dtype) * shift(value, dy, dx)
        quality_logit = self.quality(torch.cat([
            value, transported, coverage.to(value.dtype), support.to(value.dtype),
            brightness.to(value.dtype)], 1))
        confidence = support.to(value.dtype) * quality_logit.sigmoid()
        observed = coverage.to(value.dtype) * value
        inferred = (1.0 - coverage.to(value.dtype)) * confidence * transported
        correction = self.up(torch.cat([observed, inferred], 1))
        return correction, {
            "reconstruction": self.recover(transported),
            "quality_logit": quality_logit,
            "support": support,
            "coverage": coverage,
        }


class MineRelay(MineHHA):
    def __init__(self, mode="relay"):
        super().__init__()
        if mode not in ("logits", "inject", "relay"):
            raise ValueError(mode)
        self.mode = mode
        self.relay = nn.ModuleList(GeometryRelay(c) for c in (64, 128, 320, 512))
        self.joint = False
        self.configure_training(joint=False)

    def configure_training(self, joint=False):
        self.joint = joint
        for p in self.parameters():
            p.requires_grad = True
        for p in self.base.parameters():
            p.requires_grad = joint
        self.unfreeze_decoder()
        for p in self.inject.parameters():
            p.requires_grad = self.mode == "inject"
        for p in self.relay.parameters():
            p.requires_grad = self.mode == "relay"

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if not self.joint:
            self.base.eval()  # Keep inherited BN statistics stable for continuation.
        return self

    def initialize_imagenet(self, path):
        self.base.rgb_encoder.load_pretrained(path)
        self.geo_encoder.load_state_dict(self.base.rgb_encoder.state_dict())
        nn.init.kaiming_uniform_(self.geo_decoder.predict.weight, a=math.sqrt(5))
        nn.init.zeros_(self.geo_decoder.predict.bias)
        with torch.no_grad():
            self.open_raw.fill_(math.atanh(0.5))
        self.configure_training(joint=True)
        self.mix.fill_(1.0)

    def initialize_hha(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        missing, extra = self.load_state_dict(state, strict=False)
        bad = [key for key in missing if not key.startswith(("inject.", "relay."))]
        if bad or extra:
            raise RuntimeError(f"Incompatible HHA checkpoint: missing={bad[:8]}, extra={extra[:8]}")
        # v2 should have no trained feature injection. Do not silently discard it.
        if any(k.startswith("inject.") for k in state):
            raise RuntimeError("Expected the historical logits-only v2 checkpoint, not a later injection run")
        self.configure_training(joint=False)
        self.mix.fill_(1.0)
        return checkpoint

    def _fused_features(self, rgb, depth, valid, luma):
        with torch.set_grad_enabled(self.training and self.joint):
            rgb_features = self.base.rgb_encoder(rgb)
            depth_features = self.base.depth_encoder(depth)
            return [gate(r, d, valid, luma)[0]
                    for gate, r, d in zip(self.base.gates, rgb_features, depth_features)]

    def forward(self, rgb, depth, valid, luma, hha):
        fused = self._fused_features(rgb, depth, valid, luma)
        geometry = self.geo_encoder(self.stem(hha))
        mixed, relay_state = [], []
        for i, (feature, geo) in enumerate(zip(fused, geometry)):
            if geo.shape[-2:] != feature.shape[-2:]:
                geo = F.interpolate(geo, feature.shape[-2:], mode="bilinear", align_corners=False)
            if self.mode == "relay":
                delta, state = self.relay[i](feature, geo, valid, luma, depth[:, :1])
                relay_state.append(state)
            elif self.mode == "inject":
                delta = self.inject[i](geo)
            else:
                delta = 0.0
            mixed.append(feature + delta)
        size = rgb.shape[-2:]
        base = F.interpolate(self.base.decoder(mixed), size, mode="bilinear", align_corners=False)
        geo_logits = F.interpolate(self.geo_decoder(geometry), size, mode="bilinear", align_corners=False)
        # Preserve historical behaviour on nonempty depth; suppress a wholly absent stream.
        sensor_present = (valid.flatten(1).sum(1) > 0).to(base.dtype).view(-1, 1, 1, 1)
        logits = base + torch.tanh(self.open_raw) * self.mix * sensor_present * geo_logits
        return {"logits": logits, "geo_logits": geo_logits, "sensor_present": sensor_present,
                "geometry": geometry, "relay": relay_state}
