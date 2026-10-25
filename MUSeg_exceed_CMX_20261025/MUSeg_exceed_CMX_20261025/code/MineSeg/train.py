"""Train an RGB-D segmentor on the MUSeg split lists."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from museg_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    MUSegDataset,
    build_index,
    build_paste_bank,
    class_weights,
    read_ids,
)
from museg_loss import segmentation_loss
from museg_model import DEBFNet

ROOT = Path("/root/autodl-tmp/MineSeg")
PROGRESS = ROOT / "PROGRESS.txt"


def log_progress(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}\n"
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(message, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def confusion_update(matrix, logits, target):
    pred = logits.argmax(dim=1)
    valid = target != 255
    pred = pred[valid].detach().to(matrix.device)
    gt = target[valid].detach().to(matrix.device)
    index = gt * NUM_CLASSES + pred
    bins = torch.bincount(index, minlength=NUM_CLASSES * NUM_CLASSES)
    matrix += bins.reshape(NUM_CLASSES, NUM_CLASSES)


def scores_from_confusion(matrix):
    matrix = matrix.float()
    tp = matrix.diag()
    support = matrix.sum(dim=1)
    union = support + matrix.sum(dim=0) - tp
    iou = tp / union.clamp_min(1)
    present = support > 0
    miou = iou[present].mean() if present.any() else torch.tensor(0.0)
    pixel_acc = tp.sum() / matrix.sum().clamp_min(1)
    return {
        "miou": float(miou),
        "pixel_acc": float(pixel_acc),
        "iou": [float(v) for v in iou],
        "support": [int(v) for v in support],
    }


def lr_factor(epoch, epochs, warmup):
    if epoch < warmup:
        return float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    os.environ["OMP_NUM_THREADS"] = "8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rgb", "early", "debf", "edge"], default="debf")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--paste", action="store_true", help="paste long-tail instances into training images")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.4)
    parser.add_argument("--outdir", default="")
    parser.add_argument("--max-steps", type=int, default=0, help="debug: stop an epoch after N steps")
    args = parser.parse_args()
    if args.size % 32 != 0:
        raise SystemExit("--size must be a multiple of 32")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir) if args.outdir else ROOT / "runs" / args.mode
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2))

    index = build_index()
    train_ids = read_ids("train")
    val_ids = read_ids("val")
    use_depth_aug = args.mode != "rgb"
    paste_bank = None
    if args.paste:
        paste_bank, paste_counts = build_paste_bank(train_ids, index)
        log_progress(f"长尾粘贴库 {paste_counts} 共 {len(paste_bank)} 块")
    train_set = MUSegDataset(
        train_ids, index, size=args.size, train=True, use_depth_aug=use_depth_aug, paste_bank=paste_bank
    )
    val_set = MUSegDataset(val_ids, index, size=args.size, train=False, use_depth_aug=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = DEBFNet(NUM_CLASSES, mode=args.mode, use_boundary=not args.no_boundary, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1, "base_lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr, "base_lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    weights = class_weights()
    history = []
    best = -1.0
    log_progress(
        f"开始训练 mode={args.mode} epochs={args.epochs} batch={args.batch_size} "
        f"size={args.size} params={n_params:.2f}M train={len(train_set)} val={len(val_set)}"
    )

    for epoch in range(args.epochs):
        factor = lr_factor(epoch, args.epochs, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        model.train()
        t0 = time.time()
        loss_sum, seen = 0.0, 0
        part_sum = {"ce": 0.0, "dice": 0.0, "boundary": 0.0}
        for step, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits, boundary = model(rgb, depth)
            loss, parts = segmentation_loss(
                logits.float(),
                None if boundary is None else boundary.float(),
                label,
                weights,
                dice_weight=args.dice_weight,
                boundary_weight=0.0 if args.no_boundary else args.boundary_weight,
            )
            if not torch.isfinite(loss):
                log_progress(f"非有限 loss，停止。epoch={epoch} step={step}")
                raise SystemExit(2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            bs = rgb.shape[0]
            loss_sum += float(loss.detach()) * bs
            seen += bs
            for key in part_sum:
                part_sum[key] += parts[key] * bs
            if args.max_steps and step + 1 >= args.max_steps:
                break

        model.eval()
        matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                rgb = batch["rgb"].to(device, non_blocking=True)
                depth = batch["depth"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits, _ = model(rgb, depth)
                confusion_update(matrix, logits, label)
                if args.max_steps and step + 1 >= args.max_steps:
                    break
        metrics = scores_from_confusion(matrix)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / max(1, seen),
            "parts": {k: v / max(1, seen) for k, v in part_sum.items()},
            "val_miou": metrics["miou"],
            "val_pixel_acc": metrics["pixel_acc"],
            "val_iou": metrics["iou"],
            "seconds": elapsed,
            "lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))
        worst = sorted(range(NUM_CLASSES), key=lambda i: metrics["iou"][i])[:3]
        worst_txt = ", ".join(f"{CLASS_NAMES[i]}={metrics['iou'][i]:.3f}" for i in worst)
        improved = metrics["miou"] > best
        if improved:
            best = metrics["miou"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "val_miou": best, "args": vars(args)},
                outdir / "best.pth",
            )
        message = (
            f"mode={args.mode} epoch {epoch+1}/{args.epochs} loss={row['loss']:.4f} "
            f"val_mIoU={metrics['miou']:.4f} pixAcc={metrics['pixel_acc']:.4f} "
            f"best={best:.4f} worst=[{worst_txt}] {elapsed:.0f}s"
        )
        log_progress(message)
        free_gb = shutil_disk_free_gb("/root/autodl-tmp")
        if free_gb < 5:
            log_progress(f"磁盘剩余 {free_gb:.1f}GB，只保留 best.pth")
            last = outdir / "last.pth"
            if last.exists():
                last.unlink()

    log_progress(f"训练结束 mode={args.mode} best_val_mIoU={best:.4f} 权重 {outdir / 'best.pth'}")


def shutil_disk_free_gb(path):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / (1024 ** 3)


if __name__ == "__main__":
    main()
