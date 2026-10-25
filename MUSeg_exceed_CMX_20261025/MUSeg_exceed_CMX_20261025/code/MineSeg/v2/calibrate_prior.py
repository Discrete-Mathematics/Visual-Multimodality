"""Val-only sweep of a background-logit prior, then one test pass.

The prior is chosen on FTransUNet val.txt. test.txt is read only after the
prior is frozen.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source/tools")

from finetune_selective import prepare_model
from mine_relay.data import MineDataset
from mine_relay.engine import forward_batch

BIASES = (-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 0.9)
CMX = {"pa": 86.30, "mpa": 72.72, "miou": 61.83}
SPLIT = {
    "val": Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/val.txt"),
    "test": Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/test.txt"),
}


def empty():
    return {bias: np.zeros((16, 16), dtype=np.float64) for bias in BIASES}


def update(matrices, logits, target):
    target = target.numpy()
    for bias in BIASES:
        scores = logits.clone()
        scores[:, 0] += bias
        pred = scores.argmax(1).cpu().numpy()
        for index in range(pred.shape[0]):
            valid = target[index] != 255
            gt = target[index][valid]
            pr = pred[index][valid]
            ok = (gt >= 0) & (gt < 16) & (pr >= 0) & (pr < 16)
            if ok.any():
                matrices[bias] += np.bincount(gt[ok] * 16 + pr[ok], minlength=256).reshape(16, 16)


def summarize(matrix):
    tp = np.diag(matrix)
    support = matrix.sum(1)
    pred = matrix.sum(0)
    union = support + pred - tp
    present = support > 0
    iou = tp / np.maximum(union, 1.0)
    recall = tp / np.maximum(support, 1.0)
    fg = np.arange(1, 16)
    fg_present = support[fg] > 0
    return {
        "miou16": float(iou[present].mean()) * 100,
        "mpa16": float(recall[present].mean()) * 100,
        "pa16": float(tp.sum() / max(matrix.sum(), 1.0)) * 100,
        "miou15": float(iou[fg][fg_present].mean()) * 100,
        "mpa15": float(recall[fg][fg_present].mean()) * 100,
        "pa15": float(tp[fg].sum() / max(support[fg].sum(), 1.0)) * 100,
    }


def beats(row):
    return row["miou16"] > CMX["miou"] and row["mpa16"] > CMX["mpa"] and row["pa16"] > CMX["pa"]


def run_split(model, split, height, width):
    dataset = MineDataset("/root/autodl-tmp/MUSeg/flat", split, (height, width), train=False, seed=2026)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    matrices = empty()
    for batch in loader:
        output = forward_batch(model, batch)
        update(matrices, output["logits"].float(), batch["label"])
    return {str(bias): summarize(matrices[bias]) for bias in BIASES}


def main():
    device = "cuda"
    model = prepare_model(device)
    payload = torch.load(
        "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.eval()
    val = run_split(model, SPLIT["val"], 768, 896)
    ranked = sorted(val.items(), key=lambda item: (beats(item[1]), item[1]["miou16"]), reverse=True)
    chosen = ranked[0][0]
    test = run_split(model, SPLIT["test"], 768, 896)
    result = {
        "checkpoint": "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
        "epoch": payload.get("epoch"),
        "chosen_on": "val",
        "chosen_bias": float(chosen),
        "val": val,
        "test_at_chosen_bias": test[chosen],
        "test_at_zero": test["0.0"],
        "beats_cmx_16": beats(test[chosen]),
    }
    out = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/prior_sweep.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"chosen_bias": result["chosen_bias"], "val_best": val[chosen], "test": test[chosen], "beats": result["beats_cmx_16"]}, indent=2))


if __name__ == "__main__":
    main()
