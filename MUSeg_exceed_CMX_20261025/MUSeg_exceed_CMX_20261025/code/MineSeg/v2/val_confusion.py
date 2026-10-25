"""Validation confusion of the frozen SelectiveMine checkpoint. Does not read test.txt."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")
from finetune_selective import PAPER_CLASSES, evaluate, prepare_model
from mine_relay.data import MineDataset, read_ids
from mine_relay.engine import forward_batch

NAMES = [
    "background",
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


def main():
    root = Path("/root/autodl-tmp/MUSeg/flat")
    split = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/val.txt")
    device = "cuda"
    model = prepare_model(device)
    payload = torch.load(
        "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    dataset = MineDataset(root, split, (768, 896), train=False, rare_crop=0.0, corruption="none", seed=2026)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    matrix = np.zeros((16, 16), dtype=np.float64)
    for batch in loader:
        output = forward_batch(model, batch)
        prediction = output["logits"].argmax(1).cpu().numpy()
        target = batch["label"].numpy()
        for index in range(prediction.shape[0]):
            gt = target[index]
            pred = prediction[index]
            labeled = gt != 255
            gt_l, pred_l = gt[labeled], pred[labeled]
            ok = (gt_l >= 0) & (gt_l < 16) & (pred_l >= 0) & (pred_l < 16)
            if ok.any():
                matrix += np.bincount(gt_l[ok] * 16 + pred_l[ok], minlength=256).reshape(16, 16)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    tp = np.diag(matrix)
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    pairs = []
    for i in range(16):
        for j in range(16):
            if errors[i, j] > 0:
                pairs.append((errors[i, j], NAMES[i], NAMES[j]))
    pairs.sort(reverse=True)
    total = matrix.sum()
    correct = tp.sum()
    summary = {
        "images": len(read_ids(split)),
        "pixels": int(total),
        "pa16": float(correct / max(total, 1)),
        "top_confusions": [
            {"count": int(count), "share_of_errors": float(count / max(total - correct, 1)), "true": src, "pred": dst}
            for count, src, dst in pairs[:25]
        ],
        "per_class_recall": [
            {"name": NAMES[i], "support": int(support[i]), "recall": float(tp[i] / max(support[i], 1))}
            for i in range(16)
        ],
        "per_class_precision": [
            {"name": NAMES[i], "predicted": int(predicted[i]), "precision": float(tp[i] / max(predicted[i], 1))}
            for i in range(16)
        ],
    }
    out = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/val_confusion.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"pa16": summary["pa16"], "top": summary["top_confusions"][:12]}, indent=2))
    _ = PAPER_CLASSES, evaluate


if __name__ == "__main__":
    main()
