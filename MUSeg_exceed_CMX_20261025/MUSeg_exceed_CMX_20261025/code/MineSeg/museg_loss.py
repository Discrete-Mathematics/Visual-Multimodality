import torch
import torch.nn.functional as F

from museg_data import IGNORE_INDEX, NUM_CLASSES


def boundary_target(target):
    ignore = target == IGNORE_INDEX
    coded = target.float().unsqueeze(1)
    coded = coded.masked_fill(ignore.unsqueeze(1), -1.0)
    dilated = F.max_pool2d(coded, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-coded, kernel_size=3, stride=1, padding=1)
    boundary = ((dilated - eroded).abs() > 0).squeeze(1) & ~ignore
    return boundary.float()


def segmentation_loss(logits, boundary_logits, target, class_weight, dice_weight=0.5, boundary_weight=0.4):
    class_weight = class_weight.to(logits.device)
    valid = target != IGNORE_INDEX
    ce_map = F.cross_entropy(logits, target, weight=class_weight, ignore_index=IGNORE_INDEX, reduction="none")
    if boundary_logits is not None and boundary_weight > 0:
        boundary = boundary_target(target)
        pixel_weight = 1.0 + 2.0 * boundary
    else:
        boundary = None
        pixel_weight = torch.ones_like(ce_map)
    ce = (ce_map * pixel_weight).sum() / valid.sum().clamp_min(1)

    safe = target.clone()
    safe[~valid] = 0
    one_hot = F.one_hot(safe, NUM_CLASSES).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()
    probs = torch.softmax(logits.float(), dim=1) * mask
    one_hot = one_hot * mask
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) / (denom + 1.0)).mean()

    boundary_loss = logits.new_zeros(())
    if boundary is not None:
        logits_b = boundary_logits.squeeze(1).float()
        # Rare positive edges: up-weight them inside a masked BCE.
        loss_map = F.binary_cross_entropy_with_logits(logits_b, boundary, reduction="none")
        pos = boundary.sum().clamp_min(1)
        neg = (valid.float().sum() - boundary.sum()).clamp_min(1)
        weights = torch.where(boundary > 0.5, neg / pos, torch.ones_like(boundary))
        weights = weights * valid.float()
        boundary_loss = (loss_map * weights).sum() / weights.sum().clamp_min(1)

    total = ce + dice_weight * dice + boundary_weight * boundary_loss
    parts = {"ce": float(ce.detach()), "dice": float(dice.detach()), "boundary": float(boundary_loss.detach())}
    return total, parts
