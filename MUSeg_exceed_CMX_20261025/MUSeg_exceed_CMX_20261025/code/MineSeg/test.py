"""Evaluate a trained checkpoint. Model selection must already be finished on val."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from museg_data import CLASS_NAMES, NUM_CLASSES, build_index, read_ids, MUSegDataset
from museg_model import DEBFNet
from train import confusion_update, scores_from_confusion

ROOT = Path("/root/autodl-tmp/MineSeg")
PALETTE = np.array(
    [
        [220, 20, 60],
        [255, 140, 0],
        [255, 215, 0],
        [50, 205, 50],
        [0, 206, 209],
        [30, 144, 255],
        [65, 105, 225],
        [138, 43, 226],
        [255, 105, 180],
        [160, 82, 45],
        [112, 128, 144],
        [0, 128, 0],
        [255, 99, 71],
        [70, 130, 180],
        [218, 165, 32],
    ],
    dtype=np.uint8,
)


def colorize(label):
    color = np.zeros(label.shape + (3,), dtype=np.uint8)
    valid = (label >= 0) & (label < NUM_CLASSES)
    color[valid] = PALETTE[label[valid]]
    return color


def main():
    os.environ["OMP_NUM_THREADS"] = "8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--vis", type=int, default=12)
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()

    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    model = DEBFNet(NUM_CLASSES, mode=train_args["mode"], use_boundary=not train_args.get("no_boundary", False))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    index = build_index()
    ids = read_ids(args.split)
    dataset = MUSegDataset(ids, index, size=train_args["size"], train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    mine_matrices = {mine: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for mine in range(1, 7)}
    saved = 0
    outdir = Path(args.outdir) if args.outdir else ROOT / "runs" / train_args["mode"] / f"eval_{args.split}"
    vis_dir = outdir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits, _ = model(rgb, depth)
                if args.tta:
                    logits_f, _ = model(torch.flip(rgb, dims=[3]), torch.flip(depth, dims=[3]))
                    logits = 0.5 * (logits + torch.flip(logits_f, dims=[3]))
            confusion_update(matrix, logits, label)
            pred = logits.argmax(dim=1)
            for mine in range(1, 7):
                mask = torch.tensor([m == mine for m in batch["mine"]], device=device)
                if mask.any():
                    confusion_update(mine_matrices[mine], logits[mask], label[mask])
            if saved < args.vis:
                pred_cpu = pred.cpu().numpy()
                label_cpu = label.cpu().numpy()
                rgb_cpu = batch["rgb"].numpy()
                depth_cpu = batch["depth"].numpy()
                for i in range(pred_cpu.shape[0]):
                    if saved >= args.vis:
                        break
                    image = rgb_cpu[i].transpose(1, 2, 0)
                    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                    depth_vis = (depth_cpu[i, 0] * 255).clip(0, 255).astype(np.uint8)
                    depth_vis = np.stack([depth_vis] * 3, axis=-1)
                    panel = np.concatenate(
                        [image, depth_vis, colorize(label_cpu[i]), colorize(pred_cpu[i])],
                        axis=1,
                    )
                    Image.fromarray(panel).save(vis_dir / f"{batch['id'][i]}.jpg", quality=90)
                    saved += 1

    metrics = scores_from_confusion(matrix)
    per_class = {CLASS_NAMES[i]: {"iou": metrics["iou"][i], "support": metrics["support"][i]} for i in range(NUM_CLASSES)}
    per_mine = {}
    for mine, mine_matrix in mine_matrices.items():
        mine_scores = scores_from_confusion(mine_matrix)
        per_mine[f"{mine:02d}"] = {"miou": mine_scores["miou"], "pixel_acc": mine_scores["pixel_acc"], "iou": mine_scores["iou"]}
    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "tta": args.tta,
        "miou": metrics["miou"],
        "pixel_acc": metrics["pixel_acc"],
        "per_class": per_class,
        "per_mine": per_mine,
        "selected_on": "val",
        "val_miou_at_selection": ckpt.get("val_miou"),
        "epoch": ckpt.get("epoch"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"split": args.split, "miou": metrics["miou"], "pixel_acc": metrics["pixel_acc"], "tta": args.tta}, indent=2))
    print("wrote", outdir / "metrics.json")


if __name__ == "__main__":
    main()
