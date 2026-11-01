"""Validation sweep of resolution and a background-logit bias for selective_pa3.

The pair is frozen on FTransUNet val.txt. test.txt is read once, after the choice.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source/tools")

from finetune_selective import prepare_model
from mine_relay.data import MineDataset
from mine_relay.engine import forward_batch

CHECKPOINT = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3/best.pth")
ROOT = Path("/root/autodl-tmp/MUSeg/flat")
SPLIT = {
    "val": Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/val.txt"),
    "test": Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/test.txt"),
}
OUT = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3_prior/sweep.json")
RESOLUTIONS = ((768, 896), (800, 928), (832, 960), (704, 832))
BIASES = (-0.15, 0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90)
CMX = {"pa": 86.30, "mpa": 72.72, "miou": 61.83}
# Stay near the training operating point. These floors are validation constraints
# used only to reject settings that give up the two metrics already above CMX.
MIOU_FLOOR = 61.20
MPA_FLOOR = 71.80


def summarize(matrix):
    tp = np.diag(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    union = support + predicted - tp
    present = support > 0
    iou = tp / np.maximum(union, 1.0)
    recall = tp / np.maximum(support, 1.0)
    foreground = np.arange(1, 16)
    foreground_present = support[foreground] > 0
    return {
        "miou16": float(iou[present].mean()) * 100,
        "mpa16": float(recall[present].mean()) * 100,
        "pa16": float(tp.sum() / max(matrix.sum(), 1.0)) * 100,
        "miou15": float(iou[foreground][foreground_present].mean()) * 100,
        "mpa15": float(recall[foreground][foreground_present].mean()) * 100,
        "pa15": float(tp[foreground].sum() / max(support[foreground].sum(), 1.0)) * 100,
        "per_class_iou16": [float(v) * 100 for v in iou],
        "per_class_support16": [int(v) for v in support],
    }


def beats(row):
    return row["pa16"] > CMX["pa"] and row["mpa16"] > CMX["mpa"] and row["miou16"] > CMX["miou"]


@torch.no_grad()
def run_split(model, split, height, width):
    dataset = MineDataset(ROOT, split, (height, width), train=False, seed=2026)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    matrices = {bias: np.zeros((16, 16), dtype=np.float64) for bias in BIASES}
    for batch in loader:
        logits = forward_batch(model, batch)["logits"].float()
        other_logit, other_index = logits[:, 1:].max(dim=1)
        other_cls = other_index + 1
        background = logits[:, 0]
        target = batch["label"].to(logits.device, non_blocking=True)
        valid = target != 255
        for bias in BIASES:
            # Class 0 is the lowest index, so it wins ties, matching argmax.
            prediction = torch.where(background + bias >= other_logit, torch.zeros_like(other_cls), other_cls)
            gt = target[valid]
            pr = prediction[valid]
            ok = (gt >= 0) & (gt < 16) & (pr >= 0) & (pr < 16)
            if ok.any():
                index = (gt[ok] * 16 + pr[ok]).detach().cpu().numpy()
                matrices[bias] += np.bincount(index, minlength=256).reshape(16, 16)
    return {f"{bias:.2f}": summarize(matrices[bias]) for bias in BIASES}


def choose(table):
    ranked = []
    for key, row in table.items():
        height, width, bias = key.split("|")
        feasible = row["miou16"] >= MIOU_FLOOR and row["mpa16"] >= MPA_FLOOR
        ranked.append((feasible, row["pa16"], row["miou16"], -abs(float(bias)), key, row))
    ranked.sort(reverse=True)
    return ranked[0]


def main():
    device = "cuda"
    model = prepare_model(device)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.eval()
    val = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUT.with_name("val_partial.json")
    for height, width in RESOLUTIONS:
        print(f"val {height}x{width}", flush=True)
        scores = run_split(model, SPLIT["val"], height, width)
        for bias, row in scores.items():
            val[f"{height}|{width}|{bias}"] = row
            print(
                f"  bias {bias} PA {row['pa16']:.2f} MPA {row['mpa16']:.2f} mIoU {row['miou16']:.2f}",
                flush=True,
            )
        partial.write_text(json.dumps({
            k: {kk: vv for kk, vv in row.items() if not kk.startswith("per_class")}
            for k, row in val.items()
        }, indent=2))
    _, _, _, _, chosen_key, chosen_row = choose(val)
    height, width, bias = chosen_key.split("|")
    print(f"chosen {chosen_key} PA {chosen_row['pa16']:.2f}", flush=True)
    print(f"test {height}x{width} bias {bias}", flush=True)
    test_scores = run_split(model, SPLIT["test"], int(height), int(width))
    test_row = test_scores[f"{float(bias):.2f}"]
    zero_key = f"{height}|{width}|0.00"
    result = {
        "checkpoint": str(CHECKPOINT),
        "epoch": payload.get("epoch"),
        "selection": {
            "split": str(SPLIT["val"]),
            "rule": "max validation 16-class PA among settings with mIoU>=61.20 and MPA>=71.80",
            "miou_floor": MIOU_FLOOR,
            "mpa_floor": MPA_FLOOR,
            "chosen_key": chosen_key,
            "chosen_bias": float(bias),
            "height": int(height),
            "width": int(width),
        },
        "val_chosen": {k: v for k, v in chosen_row.items() if not k.startswith("per_class")},
        "val": {k: {kk: vv for kk, vv in row.items() if not kk.startswith("per_class")} for k, row in val.items()},
        "test_at_chosen": test_row,
        "test_at_zero_same_resolution": test_scores["0.00"],
        "beats_cmx_16": beats(test_row),
        "cmx": CMX,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "chosen": chosen_key,
        "test": {k: test_row[k] for k in ("pa16", "mpa16", "miou16")},
        "beats_cmx_16": result["beats_cmx_16"],
        "zero_same_res": {k: test_scores["0.00"][k] for k in ("pa16", "mpa16", "miou16")},
    }, indent=2), flush=True)
    del zero_key


if __name__ == "__main__":
    main()
