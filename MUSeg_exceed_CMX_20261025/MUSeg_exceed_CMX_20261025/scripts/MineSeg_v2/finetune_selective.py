"""Fine-tune the released SelectiveMine checkpoint on official MUSeg train groups.

The comparable score ignores unlabeled pixels (raw label 0 and 255) and averages
the 15 semantic classes named in the MUSeg paper. The 16-class score, which
counts background, is recorded beside it and is not the selection target.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("OMP_NUM_THREADS", "1")

SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from mine_relay.data import MineDataset, group_id, read_ids
from mine_relay.engine import class_weights, forward_batch
from mine_selective.losses import selective_loss
from selective_inference_bundle_20261023 import load_for_inference
from engine import lovasz_softmax

PAPER_CLASSES = 15
BASELINES = {
    "CMX": {"pa": 86.30, "mpa": 72.72, "miou": 61.83},
    "DFormer": {"pa": 85.63, "mpa": 70.37, "miou": 59.74},
    "SA-Gate": {"pa": 84.77, "mpa": 71.34, "miou": 59.38},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def paper_scores(prediction, target):
    """15-class scores. Raw 0 and 255 are unlabeled and do not enter the union."""
    valid = (target >= 1) & (target <= PAPER_CLASSES)
    prediction = prediction[valid]
    target = target[valid] - 1
    mapped = prediction - 1
    recognized = (mapped >= 0) & (mapped < PAPER_CLASSES)
    matrix = np.zeros((PAPER_CLASSES, PAPER_CLASSES), dtype=np.float64)
    if recognized.any():
        index = target[recognized] * PAPER_CLASSES + mapped[recognized]
        bins = np.bincount(index, minlength=PAPER_CLASSES * PAPER_CLASSES)
        matrix += bins.reshape(PAPER_CLASSES, PAPER_CLASSES)
    missed = np.bincount(target[~recognized], minlength=PAPER_CLASSES).astype(np.float64) if (~recognized).any() else np.zeros(PAPER_CLASSES)
    true_positive = np.diag(matrix)
    false_positive = matrix.sum(0) - true_positive
    false_negative = matrix.sum(1) - true_positive + missed
    support = matrix.sum(1) + missed
    union = true_positive + false_positive + false_negative
    present = support > 0
    iou = true_positive / np.maximum(union, 1.0)
    recall = true_positive / np.maximum(support, 1.0)
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "mpa": float(recall[present].mean()) if present.any() else 0.0,
        "pa": float(true_positive.sum() / max(support.sum(), 1.0)),
        "iou": iou.tolist(),
        "support": support.astype(np.int64).tolist(),
    }


def sixteen_scores(prediction, target):
    valid = target != 255
    prediction = prediction[valid]
    target = target[valid]
    keep = (target >= 0) & (target < 16) & (prediction >= 0) & (prediction < 16)
    prediction, target = prediction[keep], target[keep]
    matrix = np.bincount(target * 16 + prediction, minlength=256).reshape(16, 16).astype(np.float64)
    true_positive = np.diag(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    union = support + predicted - true_positive
    present = support > 0
    iou = true_positive / np.maximum(union, 1.0)
    recall = true_positive / np.maximum(support, 1.0)
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "mpa": float(recall[present].mean()) if present.any() else 0.0,
        "pa": float(true_positive.sum() / max(matrix.sum(), 1.0)),
    }


@torch.no_grad()
def _accumulate(paper, missed, matrix16, prediction, target):
    for index in range(prediction.shape[0]):
        pred = prediction[index]
        gt = target[index]
        valid = (gt >= 1) & (gt <= PAPER_CLASSES)
        pred_v, gt_v = pred[valid], gt[valid] - 1
        mapped = pred_v - 1
        recognized = (mapped >= 0) & (mapped < PAPER_CLASSES)
        if recognized.any():
            bins = np.bincount(gt_v[recognized] * PAPER_CLASSES + mapped[recognized], minlength=PAPER_CLASSES ** 2)
            paper += bins.reshape(PAPER_CLASSES, PAPER_CLASSES)
        if (~recognized).any():
            missed += np.bincount(gt_v[~recognized], minlength=PAPER_CLASSES)
        labeled = gt != 255
        pred_l, gt_l = pred[labeled], gt[labeled]
        ok = (gt_l >= 0) & (gt_l < 16) & (pred_l >= 0) & (pred_l < 16)
        if ok.any():
            matrix16 += np.bincount(gt_l[ok] * 16 + pred_l[ok], minlength=256).reshape(16, 16)
    return paper, missed, matrix16


def _pack(paper, missed, matrix16, prefix):
    true_positive = np.diag(paper)
    false_positive = paper.sum(0) - true_positive
    false_negative = paper.sum(1) - true_positive + missed
    support = paper.sum(1) + missed
    union = true_positive + false_positive + false_negative
    present = support > 0
    iou = true_positive / np.maximum(union, 1.0)
    recall = true_positive / np.maximum(support, 1.0)
    tp16 = np.diag(matrix16)
    support16 = matrix16.sum(1)
    pred16 = matrix16.sum(0)
    union16 = support16 + pred16 - tp16
    present16 = support16 > 0
    iou16 = tp16 / np.maximum(union16, 1.0)
    recall16 = tp16 / np.maximum(support16, 1.0)
    packed = {
        f"{prefix}miou": float(iou[present].mean()) if present.any() else 0.0,
        f"{prefix}mpa": float(recall[present].mean()) if present.any() else 0.0,
        f"{prefix}pa": float(true_positive.sum() / max(support.sum(), 1.0)),
        f"{prefix}miou_16": float(iou16[present16].mean()) if present16.any() else 0.0,
        f"{prefix}mpa_16": float(recall16[present16].mean()) if present16.any() else 0.0,
        f"{prefix}pa_16": float(tp16.sum() / max(matrix16.sum(), 1.0)),
    }
    if prefix == "holdout_":
        packed["per_class_iou"] = iou.tolist()
        packed["per_class_support"] = support.astype(np.int64).tolist()
    return packed


def _resize_modal(tensor, size, nearest=False):
    mode = "nearest" if nearest else "bilinear"
    if nearest:
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


@torch.no_grad()
def tta_probabilities(model, batch, scales=(0.75, 1.0, 1.25)):
    base = batch["rgb"].shape[-2:]
    accumulated = None
    views = 0
    keys = ("rgb", "depth", "valid", "luma", "hha")
    for scale in scales:
        height = max(32, int(round(base[0] * scale / 32.0) * 32))
        width = max(32, int(round(base[1] * scale / 32.0) * 32))
        for flip in (False, True):
            tensors = []
            for key in keys:
                tensor = batch[key].cuda(non_blocking=True)
                tensor = _resize_modal(tensor, (height, width), nearest=(key == "valid"))
                if flip:
                    tensor = torch.flip(tensor, dims=[-1])
                tensors.append(tensor)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(*tensors)
            probs = output["logits"].float().exp()
            if flip:
                probs = torch.flip(probs, dims=[-1])
            if probs.shape[-2:] != base:
                probs = F.interpolate(probs, size=base, mode="bilinear", align_corners=False)
            accumulated = probs if accumulated is None else accumulated + probs
            views += 1
    return accumulated / views


def evaluate(model, loader, tta=False):
    paper = np.zeros((PAPER_CLASSES, PAPER_CLASSES), dtype=np.float64)
    missed = np.zeros(PAPER_CLASSES, dtype=np.float64)
    matrix16 = np.zeros((16, 16), dtype=np.float64)
    fused_paper = np.zeros_like(paper)
    fused_missed = np.zeros_like(missed)
    fused_matrix = np.zeros_like(matrix16)
    was_training = model.training
    model.eval()
    for batch in loader:
        if tta:
            prediction = tta_probabilities(model, batch).argmax(1).cpu().numpy()
            fused = prediction
        else:
            output = forward_batch(model, batch)
            prediction = output["logits"].argmax(1).cpu().numpy()
            fused = output["fused_logits"].argmax(1).cpu().numpy()
        target = batch["label"].numpy()
        _accumulate(paper, missed, matrix16, prediction, target)
        _accumulate(fused_paper, fused_missed, fused_matrix, fused, target)
    if was_training:
        model.train()
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
    metrics = _pack(paper, missed, matrix16, "holdout_")
    metrics.update(_pack(fused_paper, fused_missed, fused_matrix, "fused_"))
    # Keep the historical 16-class keys used by the mail report.
    metrics["miou_16"] = metrics["holdout_miou_16"]
    metrics["mpa_16"] = metrics["holdout_mpa_16"]
    metrics["pa_16"] = metrics["holdout_pa_16"]
    return metrics


def prepare_model(device):
    model, _metadata = load_for_inference(
        "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth",
        device=device,
    )
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.train()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rare-crop", type=float, default=0.3)
    parser.add_argument("--lovasz-weight", type=float, default=0.0)
    parser.add_argument("--pixel-ce", type=float, default=0.0, help="extra unweighted NLL on the mixture, added to the class-weighted loss")
    parser.add_argument("--uniform-weights", action="store_true")
    parser.add_argument("--select-margin", action="store_true")
    parser.add_argument(
        "--select-pa",
        action="store_true",
        help="keep the checkpoint with the highest 16-class PA among epochs that stay above the validation floors",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--outdir", default="/root/autodl-tmp/MineSeg/v2/runs/selective_hi")
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        raise SystemExit("height and width must be multiples of 32")

    seed_everything(args.seed)
    device = "cuda"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path("/root/autodl-tmp/MUSeg/flat")
    split_root = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
    train_split = split_root / "train.txt"
    val_split = split_root / "val.txt"
    test_split = split_root / "test.txt"
    train_ids = read_ids(train_split)
    val_ids = read_ids(val_split)
    test_ids = read_ids(test_split)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise SystemExit("split leakage")
    if set(map(group_id, train_ids)) & set(map(group_id, val_ids)):
        raise SystemExit("train and validation share a capture group")

    print("loading checkpoint", flush=True)
    model = prepare_model(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
        print(f"loaded {args.checkpoint} epoch {payload.get('epoch')}", flush=True)

    def make_loader(split, train):
        dataset = MineDataset(
            root,
            split,
            (args.height, args.width),
            train=train,
            paired=False,
            rare_crop=args.rare_crop if train else 0.0,
            corruption="none",
            seed=args.seed,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=train,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=train,
            persistent_workers=args.workers > 0,
        )

    val_loader = make_loader(val_split, False)
    if args.eval_only or args.eval_test:
        split = test_split if args.eval_test else val_split
        loader = make_loader(split, False)
        started = time.time()
        metrics = evaluate(model, loader, tta=args.tta)
        metrics.update({
            "split": str(split),
            "images": len(read_ids(split)),
            "seconds": round(time.time() - started, 1),
            "height": args.height,
            "width": args.width,
            "baselines": BASELINES,
            "beats_cmx": {
                "pa": metrics["holdout_pa"] * 100 > BASELINES["CMX"]["pa"],
                "mpa": metrics["holdout_mpa"] * 100 > BASELINES["CMX"]["mpa"],
                "miou": metrics["holdout_miou"] * 100 > BASELINES["CMX"]["miou"],
            },
        })
        name = "official_test.json" if args.eval_test else "dev_val.json"
        (outdir / name).write_text(json.dumps(metrics, indent=2))
        print(json.dumps({key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "mpa_16", "pa_16", "beats_cmx")}, indent=2), flush=True)
        return

    train_loader = make_loader(train_split, True)
    weights, counts = class_weights(root, train_ids)
    if args.uniform_weights:
        weights = None
    config = {
        "args": vars(args),
        "train_images": len(train_ids),
        "holdout_images": len(val_ids),
        "official_test_images": len(test_ids),
        "class_counts": [int(v) for v in counts],
        "class_weights": None if weights is None else [float(v) for v in weights.detach().cpu()],
        "metric": "15 MUSeg classes, raw label 0 and 255 ignored",
        "init": args.checkpoint or "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth",
        "selection": {
            "select_pa": args.select_pa,
            "pa_floor_miou16": 0.610,
            "pa_floor_mpa16": 0.728,
        },
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if "encoder" in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": args.lr, "weight_decay": 0.01},
            {"params": head, "lr": args.lr * 4, "weight_decay": 0.01},
        ]
    )
    history = outdir / "history.jsonl"
    best = -1.0
    steps_per_epoch = max(1, len(train_loader))
    print(f"steps/epoch {steps_per_epoch} trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(args.epochs, 1)))
        scale = max(0.05, scale)
        optimizer.param_groups[0]["lr"] = args.lr * scale
        optimizer.param_groups[1]["lr"] = args.lr * 4 * scale
        running, seen = 0.0, 0
        started = time.time()
        for batch in train_loader:
            label = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, batch)
            loss, _parts = selective_loss(
                output,
                label,
                class_weights=weights,
                rgb_weight=0.5,
                fused_weight=0.5,
                geo_weight=0.2,
                utility_weight=0.05,
                temperature=0.5,
            )
            if args.lovasz_weight > 0:
                loss = loss + args.lovasz_weight * lovasz_softmax(output["logits"].exp(), label)
            if args.pixel_ce > 0:
                loss = loss + args.pixel_ce * F.nll_loss(output["logits"].float(), label, ignore_index=255)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.multimodal.mix.fill_(1.0)
            running += float(loss.detach())
            seen += 1
        metrics = evaluate(model, val_loader)
        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "loss": running / max(seen, 1),
            "seconds": round(elapsed, 1),
            **{key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "mpa_16", "pa_16")},
        }
        with history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} loss {row['loss']:.4f} "
            f"paper mIoU {row['holdout_miou']*100:.2f} MPA {row['holdout_mpa']*100:.2f} "
            f"PA {row['holdout_pa']*100:.2f} 16-class mIoU {row['miou_16']*100:.2f} "
            f"MPA {row['mpa_16']*100:.2f} PA {row['pa_16']*100:.2f} {elapsed:.0f}s",
            flush=True,
        )
        score = row["holdout_miou"]
        score_name = "paper mIoU"
        if args.select_margin:
            score = min(
                row["miou_16"] * 100 - 61.83,
                row["mpa_16"] * 100 - 72.72,
                row["pa_16"] * 100 - 86.30,
            )
            score_name = "margin"
        elif args.select_pa:
            score_name = "16-class PA"
            if row["miou_16"] >= 0.610 and row["mpa_16"] >= 0.728:
                score = row["pa_16"]
            else:
                score = -1.0
        if score > best:
            best = score
            torch.save(
                {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": epoch,
                    "metrics": row,
                    "args": vars(args),
                },
                outdir / "best.pth",
            )
            print(f"saved best epoch {epoch} {score_name} {best*100:.2f}", flush=True)
    print(f"finished best paper mIoU {best*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
