"""Recovery supervision uses only synthetically hidden, originally observed cells."""

import torch
from torch.nn import functional as F


def recovery_loss(clean, damaged, hidden):
    total = hidden.sum() * 0.0
    quality_total = total
    active = 0
    for target, state in zip(clean["geometry"], damaged["relay"]):
        target = target.detach().float()
        prediction = state["reconstruction"].float()
        eligible = F.adaptive_avg_pool2d(hidden.float(), target.shape[-2:])
        eligible = eligible * (state["support"].detach() > 1e-4).float()
        error = (1.0 - F.cosine_similarity(prediction, target, dim=1, eps=1e-6)).unsqueeze(1)
        count = eligible.sum().clamp_min(1.0)
        total = total + (error * eligible).sum() / count
        # A relative reconstruction-quality target, not a true-depth uncertainty label.
        quality_target = torch.exp(-error.detach() / 0.5)
        bce = F.binary_cross_entropy_with_logits(state["quality_logit"].float(), quality_target, reduction="none")
        quality_total = quality_total + (bce * eligible).sum() / count
        active += 1
    return total / max(active, 1), quality_total / max(active, 1)


def segmentation_loss(output, label, weights, auxiliary=0.2):
    main = F.cross_entropy(output["logits"].float(), label, weight=weights, ignore_index=255)
    has_depth = output["sensor_present"].flatten() > 0
    if has_depth.any():
        aux = F.cross_entropy(output["geo_logits"][has_depth].float(), label[has_depth],
                              weight=weights, ignore_index=255)
    else:
        aux = output["geo_logits"].sum() * 0.0
    return main + auxiliary * aux
