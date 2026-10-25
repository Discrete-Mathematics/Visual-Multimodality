"""Depth completion and low-light residual in front of an RGB-D segmentor.

The last convolution is zero, and completed depth is mixed in only through a
sigmoid that also starts near zero. A loaded checkpoint therefore sees almost
the same tensors on the first step. Training hides a random subset of valid
depth pixels and asks the filler to reconstruct them, which is the supervision
for real Kinect holes at test time.
"""

import torch
from torch import nn
import torch.nn.functional as F


def multiscale_fill(value, mask):
    """Normalized-convolution fill. Observed pixels are left unchanged."""
    filled = value
    support = mask.clamp(0, 1)
    # Odd kernels keep the spatial size. Even kernels with padding k//2 do not.
    for kernel in (15, 9, 5, 3):
        padding = kernel // 2
        weight = F.avg_pool2d(support, kernel, stride=1, padding=padding)
        pooled = F.avg_pool2d(filled * support, kernel, stride=1, padding=padding)
        estimate = pooled / weight.clamp_min(1e-4)
        holes = support < 0.5
        usable = holes & (weight > 1e-3)
        filled = torch.where(usable, estimate, filled)
        support = torch.where(usable, torch.ones_like(support), support)
    return filled


class ComplementFront(nn.Module):
    """Shared front for SelectiveMine's packed depth and HHA."""

    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(3 + 1 + 1 + 4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1 + 4 + 3 + 1, 1),
        )
        last = self.refine[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 0.0)
        nn.init.constant_(last.bias[8], -4.0)

    def _predict(self, rgb, luma, value, hha, valid):
        filled = multiscale_fill(value, valid)
        hha_filled = torch.cat([multiscale_fill(hha[:, c : c + 1], valid) for c in range(hha.shape[1])], dim=1)
        residual = self.refine(torch.cat([rgb, luma, filled, hha_filled], dim=1))
        depth_residual = residual[:, :1]
        hha_residual = residual[:, 1:5]
        rgb_residual = residual[:, 5:8]
        confidence = torch.sigmoid(residual[:, 8:9])
        return filled, hha_filled, depth_residual, hha_residual, rgb_residual, confidence

    def forward(self, rgb, depth, valid, luma, hha):
        value = depth[:, :1]
        aux = {}
        if self.training:
            hide = (torch.rand_like(valid) < 0.3) & (valid > 0.5)
            visible = valid * (~hide).float()
            filled_h, _, depth_h, _, _, _ = self._predict(rgb, luma, value, hha, visible)
            aux["pred"] = filled_h + depth_h
            aux["target"] = value.detach()
            aux["supervise"] = hide.float()
        filled, hha_filled, depth_residual, hha_residual, rgb_residual, confidence = self._predict(
            rgb, luma, value, hha, valid
        )
        hole = 1.0 - valid.clamp(0, 1)
        new_value = value + hole * confidence * (filled + depth_residual - value)
        new_hha = hha + hole * confidence * (hha_filled + hha_residual - hha)
        darkness = ((0.45 - luma) / 0.45).clamp(0, 1)
        new_rgb = rgb + darkness * rgb_residual
        new_valid = (valid + hole * confidence).clamp(0, 1)
        magnitude = depth[:, 2:3]
        new_depth = torch.cat([new_value, new_valid, magnitude], dim=1)
        aux["confidence"] = confidence
        return new_rgb, new_depth, new_valid, luma, new_hha, aux


class DEBFFront(nn.Module):
    """Same idea for DEBFNet's normalized (log-depth, edge, valid) tensor."""

    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(3 + 1 + 1 + 1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1 + 1 + 3 + 1, 1),
        )
        last = self.refine[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 0.0)
        nn.init.constant_(last.bias[5], -4.0)

    @staticmethod
    def _denorm(depth):
        return depth * 0.25 + 0.5

    @staticmethod
    def _renorm(depth):
        return (depth - 0.5) / 0.25

    def forward(self, rgb, depth, valid, edge):
        raw = self._denorm(depth)
        log_depth = raw[:, :1].clamp(0, 1)
        luma = rgb.mean(dim=1, keepdim=True)
        aux = {}
        if self.training:
            hide = (torch.rand_like(valid) < 0.3) & (valid > 0.5)
            visible = valid * (~hide).float()
            filled_h = multiscale_fill(log_depth, visible)
            residual_h = self.refine(torch.cat([rgb, luma, filled_h, edge], dim=1))
            aux["pred"] = filled_h + residual_h[:, :1]
            aux["target"] = log_depth.detach()
            aux["supervise"] = hide.float()
        filled = multiscale_fill(log_depth, valid)
        residual = self.refine(torch.cat([rgb, luma, filled, edge], dim=1))
        confidence = torch.sigmoid(residual[:, 5:6])
        hole = 1.0 - valid.clamp(0, 1)
        new_log = log_depth + hole * confidence * (filled + residual[:, :1] - log_depth)
        new_edge = edge + hole * confidence * residual[:, 1:2]
        darkness = ((0.15 - luma) / 0.15).clamp(0, 1)
        new_rgb = rgb + darkness * residual[:, 2:5]
        new_valid = (valid + hole * confidence).clamp(0, 1)
        packed = torch.cat([new_log, new_edge, new_valid], dim=1).clamp(0, 1)
        aux["confidence"] = confidence
        return new_rgb, self._renorm(packed), new_valid, new_edge, aux


def completion_loss(aux):
    if "pred" not in aux:
        return aux["confidence"].float().sum() * 0.0
    error = (aux["pred"].float() - aux["target"].float()).abs()
    weight = aux["supervise"].float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0)
