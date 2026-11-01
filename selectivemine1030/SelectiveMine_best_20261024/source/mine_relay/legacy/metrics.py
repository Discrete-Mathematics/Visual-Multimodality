"""Pixel accuracy, mean pixel accuracy and mean IoU."""

import numpy as np

from .labels import NUM_CLASSES


def confusion_from_batch(confusion, prediction, target, ignore_index=255):
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_index
    prediction = prediction[valid]
    target = target[valid]
    inside = (prediction >= 0) & (prediction < NUM_CLASSES)
    prediction = prediction[inside]
    target = target[inside]
    encoded = NUM_CLASSES * target.astype(np.int64) + prediction.astype(np.int64)
    bincount = np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES)
    confusion += bincount.reshape(NUM_CLASSES, NUM_CLASSES)
    return confusion


def scores_from_confusion(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    row_sum = confusion.sum(axis=1).astype(np.float64)
    column_sum = confusion.sum(axis=0).astype(np.float64)
    union = row_sum + column_sum - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    pixel_acc = np.divide(true_positive, row_sum, out=np.zeros_like(true_positive), where=row_sum > 0)
    present = row_sum > 0
    total = confusion.sum()
    return {
        "pa": float(true_positive.sum() / total) if total else 0.0,
        "mpa": float(pixel_acc[present].mean()) if present.any() else 0.0,
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "iou": iou,
    }
