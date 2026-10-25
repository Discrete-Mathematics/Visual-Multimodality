"""Semantic utility labels are training-only and detached from both experts."""

import math
import torch
from torch.nn import functional as F


def weighted_mean(values, valid, label, class_weights=None):
    weights = valid.to(values.dtype)
    if class_weights is not None:
        weights = weights * class_weights.to(values.device)[label.clamp(0, 15)]
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def utility_target(rgb_logits, fused_logits, label, temperature=.5):
    """Soft preference for the expert assigning more probability to the truth.

    This is a training target for relative predictive usefulness, not calibrated
    sensor reliability. Class weights do not alter its target temperature.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    valid = label != 255
    if torch.any(valid & ((label < 0) | (label >= 16))):
        raise ValueError('Unexpected semantic labels')
    safe = label.clamp(0, 15).unsqueeze(1)
    rgb_nll = -F.log_softmax(rgb_logits.detach().float(), 1).gather(1, safe).squeeze(1)
    fused_nll = -F.log_softmax(fused_logits.detach().float(), 1).gather(1, safe).squeeze(1)
    preference = torch.sigmoid((rgb_nll - fused_nll) / temperature)
    return torch.where(valid, preference, .5).detach()


def selective_loss(output, label, class_weights=None, rgb_weight=.5, fused_weight=.5,
                   geo_weight=.2, utility_weight=.05, temperature=.5):
    coefficients = (rgb_weight, fused_weight, geo_weight, utility_weight)
    if not all(math.isfinite(v) and v >= 0 for v in coefficients):
        raise ValueError('Loss weights must be finite and nonnegative')
    if output['selection'] == 'control' and utility_weight:
        raise ValueError('Control has no learned utility router; set utility_weight=0')
    target = utility_target(output['rgb_logits'], output['fused_logits'], label, temperature)
    valid = label != 255
    available = valid & (output['sensor_present'][:, 0] > 0)
    main = F.nll_loss(output['logits'].float(), label, ignore_index=255, reduction='none')
    rgb = F.cross_entropy(output['rgb_logits'].float(), label, ignore_index=255, reduction='none')
    fused = F.cross_entropy(output['fused_logits'].float(), label, ignore_index=255, reduction='none')
    geo = F.cross_entropy(output['geo_logits'].float(), label, ignore_index=255, reduction='none')
    quality = F.binary_cross_entropy_with_logits(output['router_logit'][:, 0].float(), target, reduction='none')
    terms = {'main': weighted_mean(main, valid, label, class_weights),
             'rgb': weighted_mean(rgb, valid, label, class_weights),
             'fused': weighted_mean(fused, available, label, class_weights),
             'geo': weighted_mean(geo, available, label, class_weights),
             'utility': weighted_mean(quality, available, label, class_weights)}
    total = (terms['main'] + rgb_weight * terms['rgb'] + fused_weight * terms['fused']) / (1 + rgb_weight + fused_weight)
    return total + geo_weight * terms['geo'] + utility_weight * terms['utility'], terms
