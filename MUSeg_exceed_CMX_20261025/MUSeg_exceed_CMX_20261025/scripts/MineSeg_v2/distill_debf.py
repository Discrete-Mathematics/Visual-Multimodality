"""Distill DEBFNet from the frozen SelectiveMine checkpoint.

Validation is FTransUNet val.txt. The test split is loaded only when --eval-test
is passed, and training never reads it. No depth-completion front is used.
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
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("OMP_NUM_THREADS", "1")

V2 = Path("/root/autodl-tmp/MineSeg/v2")
SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(V2))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from engine import CLASS_NAMES, lovasz_softmax
from mine_relay.data import MineDataset, group_id, read_ids
from models import MineSegmentor
from selective_inference_bundle_20261023 import load_for_inference

BASELINES = {
    "CMX": {"pa": 86.30, "mpa": 72.72, "miou": 61.83},
    "DFormer": {"pa": 85.63, "mpa": 70.37, "miou": 59.74},
    "SA-Gate": {"pa": 84.77, "mpa": 71.34, "miou": 59.38},
}
SPLIT_ROOT = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
WEIGHTS = {"b2": "/root/autodl-tmp/weights/mit-b2.bin", "b5": "/root/autodl-tmp/weights/mit-b5.bin"}
IGNORE = 255


class PairedDataset(Dataset):
    """One crop encoded for SelectiveMine and for DEBFNet."""

    def __init__(self, split, height, width, train, rare_crop):
        self.inner = MineDataset(
            Path("/root/autodl-tmp/MUSeg/flat"),
            split,
            (height, width),
            train=train,
            rare_crop=rare_crop if train else 0.0,
            corruption="none",
            seed=2026,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        sample = self.inner[index]
        log_depth = sample["depth"][0]
        valid = sample["valid"]
        gy, gx = torch.gradient(log_depth)
        edge = torch.sqrt(gx * gx + gy * gy)
        edge = edge / edge.max().clamp_min(1e-6)
        edge = edge * valid[0]
        packed = torch.stack([log_depth, edge, valid[0]], dim=0)
        packed = (packed - 0.5) / 0.25
        return {
            "rgb": sample["rgb"].contiguous(),
            "depth": sample["depth"].contiguous(),
            "valid": valid.contiguous(),
            "luma": sample["luma"].contiguous(),
            "hha": sample["hha"].contiguous(),
            "debf_depth": packed.contiguous(),
            "edge": edge.unsqueeze(0).contiguous(),
            "label": sample["label"].contiguous(),
        }


def _scores(matrix):
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
        "iou": iou,
    }


@torch.no_grad()
def evaluate(model, loader):
    matrix16 = np.zeros((16, 16), dtype=np.float64)
    matrix15 = np.zeros((15, 15), dtype=np.float64)
    missed = np.zeros(15, dtype=np.float64)
    was = model.training
    model.eval()
    for batch in loader:
        rgb = batch["rgb"].cuda(non_blocking=True)
        depth = batch["debf_depth"].cuda(non_blocking=True)
        valid = batch["valid"].cuda(non_blocking=True)
        edge = batch["edge"].cuda(non_blocking=True)
        target = batch["label"].numpy()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(rgb, depth, valid, edge)["logits"].float().argmax(1).cpu().numpy()
        for index in range(prediction.shape[0]):
            gt = target[index]
            pred = prediction[index]
            labeled = gt != IGNORE
            gt_l, pred_l = gt[labeled], pred[labeled]
            ok = (gt_l >= 0) & (gt_l < 16) & (pred_l >= 0) & (pred_l < 16)
            if ok.any():
                matrix16 += np.bincount(gt_l[ok] * 16 + pred_l[ok], minlength=256).reshape(16, 16)
            fg = (gt >= 1) & (gt <= 15)
            gt_v, pred_v = gt[fg] - 1, pred[fg] - 1
            recognized = (pred_v >= 0) & (pred_v < 15)
            if recognized.any():
                matrix15 += np.bincount(gt_v[recognized] * 15 + pred_v[recognized], minlength=225).reshape(15, 15)
            if (~recognized).any():
                missed += np.bincount(gt_v[~recognized], minlength=15)
    if was:
        model.train()
    full = _scores(matrix16)
    true_positive = np.diag(matrix15)
    false_positive = matrix15.sum(0) - true_positive
    false_negative = matrix15.sum(1) - true_positive + missed
    support = matrix15.sum(1) + missed
    union = true_positive + false_positive + false_negative
    present = support > 0
    iou15 = true_positive / np.maximum(union, 1.0)
    recall15 = true_positive / np.maximum(support, 1.0)
    return {
        "miou_16": full["miou"],
        "mpa_16": full["mpa"],
        "pa_16": full["pa"],
        "miou_15": float(iou15[present].mean()) if present.any() else 0.0,
        "mpa_15": float(recall15[present].mean()) if present.any() else 0.0,
        "pa_15": float(true_positive.sum() / max(support.sum(), 1.0)),
        "per_class_iou16": full["iou"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-mult", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--variant", choices=["b2", "b5"], default="b2")
    parser.add_argument("--kl-weight", type=float, default=3.0)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--lovasz-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--rare-crop", type=float, default=0.25)
    parser.add_argument("--teacher", default="/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth")
    parser.add_argument("--outdir", default="/root/autodl-tmp/MineSeg/v2/runs/debf_distill_b2")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_ids = read_ids(SPLIT_ROOT / "train.txt")
    val_ids = read_ids(SPLIT_ROOT / "val.txt")
    test_ids = read_ids(SPLIT_ROOT / "test.txt")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise SystemExit("split leakage")
    if set(map(group_id, train_ids)) & set(map(group_id, val_ids)):
        raise SystemExit("train and validation share a capture group")
    if set(map(group_id, train_ids)) & set(map(group_id, test_ids)):
        raise SystemExit("train and test share a capture group")

    student = MineSegmentor(args.variant, WEIGHTS[args.variant], "debf", num_classes=16).cuda()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"]
        current = student.state_dict()
        filtered = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
        student.load_state_dict(filtered, strict=False)
        expanded = False
        weight = state.get("decoder.classifier.weight")
        bias = state.get("decoder.classifier.bias")
        if weight is not None and weight.shape[0] == 15 and student.decoder.classifier.weight.shape[0] == 16:
            with torch.no_grad():
                student.decoder.classifier.weight[1:].copy_(weight)
                student.decoder.classifier.bias[1:].copy_(bias)
                student.decoder.classifier.weight[0].zero_()
                student.decoder.classifier.bias[0].fill_(-2.0)
            expanded = True
        print(f"loaded student {args.checkpoint} copied={len(filtered)} classifier_expanded={expanded}", flush=True)
    teacher, _ = load_for_inference(
        "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth",
        device="cuda",
    )
    teacher_payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_payload["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    print(f"teacher {args.teacher} epoch {teacher_payload.get('epoch')}", flush=True)

    def make_loader(split, train):
        dataset = PairedDataset(split, args.height, args.width, train, args.rare_crop)
        return DataLoader(
            dataset,
            batch_size=args.batch_size if train else max(1, args.batch_size // 2),
            shuffle=train,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=train,
            persistent_workers=args.workers > 0,
        )

    val_loader = make_loader(SPLIT_ROOT / "val.txt", False)
    if args.eval_test:
        metrics = evaluate(student, make_loader(SPLIT_ROOT / "test.txt", False))
        metrics.update({"split": str(SPLIT_ROOT / "test.txt"), "images": len(test_ids), "baselines": BASELINES})
        (outdir / "official_test.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: metrics[k] for k in ("miou_16", "mpa_16", "pa_16", "miou_15", "mpa_15", "pa_15")}, indent=2), flush=True)
        return

    train_loader = make_loader(SPLIT_ROOT / "train.txt", True)
    encoder, head = [], []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if "encoder" in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": args.lr, "weight_decay": 0.01},
            {"params": head, "lr": args.lr * args.head_mult, "weight_decay": 0.01},
        ]
    )
    config = {
        "model": "DEBFNet",
        "teacher": args.teacher,
        "teacher_epoch": teacher_payload.get("epoch"),
        "args": vars(args),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_images": len(test_ids),
        "train_split": str(SPLIT_ROOT / "train.txt"),
        "val_split": str(SPLIT_ROOT / "val.txt"),
        "test_split": str(SPLIT_ROOT / "test.txt"),
        "class_names": ["background", *CLASS_NAMES],
        "baselines": BASELINES,
        "selection": "highest validation 16-class mIoU; test is not read during training",
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    history = outdir / "history.jsonl"
    best = -1.0
    print(f"steps/epoch {len(train_loader)} params {sum(p.numel() for p in student.parameters())/1e6:.2f}M", flush=True)
    temperature = args.temperature
    for epoch in range(1, args.epochs + 1):
        student.train()
        scale = max(0.1, 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        optimizer.param_groups[0]["lr"] = args.lr * scale
        optimizer.param_groups[1]["lr"] = args.lr * args.head_mult * scale
        running = seen = 0
        started = time.time()
        for batch in train_loader:
            rgb = batch["rgb"].cuda(non_blocking=True)
            label = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = student(
                    rgb,
                    batch["debf_depth"].cuda(non_blocking=True),
                    batch["valid"].cuda(non_blocking=True),
                    batch["edge"].cuda(non_blocking=True),
                )
            student_log = F.log_softmax(output["logits"].float() / temperature, dim=1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_out = teacher(
                    rgb,
                    batch["depth"].cuda(non_blocking=True),
                    batch["valid"].cuda(non_blocking=True),
                    batch["luma"].cuda(non_blocking=True),
                    batch["hha"].cuda(non_blocking=True),
                )
            teacher_log = teacher_out["logits"].float()
            teacher_prob = torch.softmax(teacher_log / temperature, dim=1)
            valid = label != IGNORE
            if valid.any():
                distill = F.kl_div(
                    student_log.permute(0, 2, 3, 1)[valid],
                    teacher_prob.permute(0, 2, 3, 1)[valid],
                    reduction="batchmean",
                ) * (temperature ** 2)
                ce = F.cross_entropy(output["logits"].float(), label, ignore_index=IGNORE)
            else:
                distill = student_log.sum() * 0.0
                ce = distill
            loss = args.ce_weight * ce + args.kl_weight * distill
            if args.lovasz_weight > 0:
                loss = loss + args.lovasz_weight * lovasz_softmax(torch.softmax(output["logits"].float(), dim=1), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach())
            seen += 1
        metrics = evaluate(student, val_loader)
        row = {
            "epoch": epoch,
            "loss": running / max(seen, 1),
            "seconds": round(time.time() - started, 1),
            "miou_16": metrics["miou_16"],
            "mpa_16": metrics["mpa_16"],
            "pa_16": metrics["pa_16"],
            "miou_15": metrics["miou_15"],
            "mpa_15": metrics["mpa_15"],
            "pa_15": metrics["pa_15"],
        }
        with history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} loss {row['loss']:.4f} "
            f"16-class mIoU {row['miou_16']*100:.2f} MPA {row['mpa_16']*100:.2f} PA {row['pa_16']*100:.2f} "
            f"15-class mIoU {row['miou_15']*100:.2f} {row['seconds']:.0f}s",
            flush=True,
        )
        if row["miou_16"] > best:
            best = row["miou_16"]
            torch.save(
                {
                    "model": {key: value.detach().cpu() for key, value in student.state_dict().items()},
                    "epoch": epoch,
                    "metrics": row,
                    "args": vars(args),
                    "per_class_iou16": metrics["per_class_iou16"],
                },
                outdir / "best.pth",
            )
            print(f"saved best epoch {epoch} 16-class mIoU {best*100:.2f}", flush=True)
    print(f"finished best 16-class mIoU {best*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
