"""Losses and the 15-class MUSeg confusion-matrix metrics."""

import numpy as np
import torch
import torch.nn.functional as F

from data import NUM_CLASSES, IGNORE

CLASS_NAMES = [
    "person",
    "cable",
    "tube",
    "indicator",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "rail area",
    "support equipment",
    "door",
    "tools and materials",
    "rescue equipment",
    "container",
    "metal fixture",
    "anchoring equipment",
]


def confusion_update(matrix, logits, target, log_space=False):
    classes = matrix.shape[0]
    prediction = logits.argmax(dim=1)
    valid = target != IGNORE
    prediction = prediction[valid]
    gt = target[valid]
    index = gt * classes + prediction
    bins = torch.bincount(index, minlength=classes * classes)
    matrix += bins.reshape(classes, classes).to(matrix.dtype)


def scores_from_confusion(matrix):
    matrix = matrix.float()
    tp = matrix.diag()
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    union = support + predicted - tp
    iou = tp / union.clamp_min(1)
    recall = tp / support.clamp_min(1)
    present = support > 0
    miou = iou[present].mean() if present.any() else torch.zeros(())
    mpa = recall[present].mean() if present.any() else torch.zeros(())
    pa = tp.sum() / matrix.sum().clamp_min(1)
    return {
        "miou": float(miou),
        "mpa": float(mpa),
        "pa": float(pa),
        "iou": [float(v) for v in iou],
        "recall": [float(v) for v in recall],
        "support": [int(v) for v in support],
    }


def _lovasz_grad(gt_sorted):
    length = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if length > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(probs, target):
    losses = []
    valid = target != IGNORE
    if valid.sum() == 0:
        return probs.sum() * 0.0
    probs = probs.permute(0, 2, 3, 1)[valid]
    labels = target[valid]
    for class_id in range(probs.shape[1]):
        foreground = (labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probs[:, class_id]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        grad = _lovasz_grad(foreground[perm])
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def dice_loss(probs, target):
    valid = target != IGNORE
    safe = target.clone()
    safe[~valid] = 0
    classes = probs.shape[1]
    one_hot = F.one_hot(safe, classes).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1).float()
    probs = probs * mask
    one_hot = one_hot * mask
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    if not present.any():
        return probs.sum() * 0.0
    dice = (2 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice[present].mean()


def boundary_target(target):
    ignore = target == IGNORE
    coded = target.float().unsqueeze(1)
    coded = coded.masked_fill(ignore.unsqueeze(1), -1.0)
    dilated = F.max_pool2d(coded, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-coded, kernel_size=3, stride=1, padding=1)
    boundary = ((dilated - eroded).abs() > 0).squeeze(1) & ~ignore
    return boundary.float()


def segmentation_loss(output, target, class_weight, simple=False):
    class_weight = class_weight.to(output["logits"].device)
    if output["log_space"]:
        main = F.nll_loss(output["logits"], target, weight=class_weight, ignore_index=IGNORE)
        probs = output["logits"].exp()
        aux = 0.5 * F.cross_entropy(output["rgb_logits"], target, weight=class_weight, ignore_index=IGNORE)
        aux = aux + 0.5 * F.cross_entropy(output["fused_logits"], target, weight=class_weight, ignore_index=IGNORE)
        valid = target != IGNORE
        nll_rgb = F.cross_entropy(output["rgb_logits"].detach(), target, ignore_index=IGNORE, reduction="none")
        nll_fused = F.cross_entropy(output["fused_logits"].detach(), target, ignore_index=IGNORE, reduction="none")
        utility = torch.sigmoid((nll_rgb - nll_fused) / 0.5)
        router = F.binary_cross_entropy(output["gate"].squeeze(1).clamp(1e-4, 1 - 1e-4), utility, reduction="none")
        router = (router * valid.float()).sum() / valid.float().sum().clamp_min(1)
    else:
        main = F.cross_entropy(output["logits"], target, weight=class_weight, ignore_index=IGNORE)
        probs = torch.softmax(output["logits"].float(), dim=1)
        aux = output["logits"].sum() * 0.0
        router = output["logits"].sum() * 0.0
    dice = dice_loss(probs, target)
    if simple:
        total = main + aux + 0.5 * dice + 0.05 * router
        return total, {
            "main": float(main.detach()),
            "aux": float(aux.detach()),
            "dice": float(dice.detach()),
            "router": float(router.detach()),
        }
    lovasz = lovasz_softmax(probs, target)
    boundary = output["logits"].sum() * 0.0
    if output["boundary"] is not None:
        edge = boundary_target(target)
        valid = (target != IGNORE).float()
        loss_map = F.binary_cross_entropy_with_logits(output["boundary"].squeeze(1).float(), edge, reduction="none")
        pos = edge.sum().clamp_min(1)
        neg = (valid.sum() - edge.sum()).clamp_min(1)
        weights = torch.where(edge > 0.5, neg / pos, torch.ones_like(edge)) * valid
        boundary = (loss_map * weights).sum() / weights.sum().clamp_min(1)
    total = main + aux + 0.4 * dice + 0.5 * lovasz + 0.2 * boundary + 0.05 * router
    parts = {
        "main": float(main.detach()),
        "aux": float(aux.detach()),
        "dice": float(dice.detach()),
        "lovasz": float(lovasz.detach()),
        "boundary": float(boundary.detach()),
        "router": float(router.detach()),
    }
    return total, parts


class ModelEMA:
    def __init__(self, model):
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        decay = min(0.999, (1.0 + self.step) / (10.0 + self.step))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[key] = value.detach().clone()


def apply_state(model, state):
    model.load_state_dict(state)


@torch.no_grad()
def paper_update(matrix, missed, logits, target):
    """15 semantic classes. Predictions of unlabeled background count as misses."""
    prediction = logits.argmax(dim=1)
    valid = (target >= 1) & (target <= 15)
    prediction = prediction[valid]
    gt = target[valid] - 1
    mapped = prediction - 1
    recognized = (mapped >= 0) & (mapped < 15)
    if recognized.any():
        index = gt[recognized] * 15 + mapped[recognized]
        bins = torch.bincount(index, minlength=15 * 15)
        matrix += bins.reshape(15, 15).to(matrix.dtype)
    if (~recognized).any():
        missed += torch.bincount(gt[~recognized], minlength=15).to(missed.dtype)


def scores_from_paper(matrix, missed):
    matrix = matrix.float()
    missed = missed.float()
    tp = matrix.diag()
    fp = matrix.sum(0) - tp
    fn = matrix.sum(1) - tp + missed
    support = matrix.sum(1) + missed
    union = tp + fp + fn
    iou = tp / union.clamp_min(1)
    recall = tp / support.clamp_min(1)
    present = support > 0
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "mpa": float(recall[present].mean()) if present.any() else 0.0,
        "pa": float(tp.sum() / support.sum().clamp_min(1)),
        "iou": [float(v) for v in iou],
        "recall": [float(v) for v in recall],
        "support": [int(v) for v in support],
    }


def evaluate_loader(model, loader, device, log_space, num_classes=NUM_CLASSES):
    matrix = torch.zeros(num_classes, num_classes, device=device)
    paper = torch.zeros(15, 15, device=device) if num_classes == 16 else None
    missed = torch.zeros(15, device=device) if num_classes == 16 else None
    mines = {}
    was_training = model.training
    model.eval()
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        edge = batch["edge"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(rgb, depth, valid, edge)
        confusion_update(matrix, output["logits"], target, log_space=log_space)
        if paper is not None:
            paper_update(paper, missed, output["logits"], target)
        for index, mine_id in enumerate(batch["mine"]):
            if mine_id not in mines:
                mines[mine_id] = torch.zeros(num_classes, num_classes, device=device)
            confusion_update(
                mines[mine_id],
                output["logits"][index : index + 1],
                target[index : index + 1],
                log_space=log_space,
            )
    if was_training:
        model.train()
    summary = scores_from_confusion(matrix)
    summary["per_mine"] = {mine_id: scores_from_confusion(conf) for mine_id, conf in sorted(mines.items())}
    if paper is not None:
        summary["paper"] = scores_from_paper(paper, missed)
    return summary
