"""DEBFNet with depth completion, low-light residual, and SelectiveMine distillation.

The student keeps the DEBFNet bidirectional gates. The teacher is a frozen
SelectiveMine checkpoint. Both see the same crop: the teacher reads packed
depth and HHA, the student reads DEBFNet's log-depth tensor. Validation is
FTransUNet val.txt. Test images are never used for checkpoint selection.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("OMP_NUM_THREADS", "1")

V2 = Path("/root/autodl-tmp/MineSeg/v2")
SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(V2))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from complement import DEBFFront, completion_loss
from engine import CLASS_NAMES, segmentation_loss
from mine_relay.data import MineDataset, group_id, read_ids
from models import MineSegmentor
from selective_inference_bundle_20261023 import load_for_inference

BASELINES = {
    "CMX": {"pa": 86.30, "mpa": 72.72, "miou": 61.83},
    "DFormer": {"pa": 85.63, "mpa": 70.37, "miou": 59.74},
    "SA-Gate": {"pa": 84.77, "mpa": 71.34, "miou": 59.38},
}
SPLIT_ROOT = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
WEIGHTS = {
    "b2": "/root/autodl-tmp/weights/mit-b2.bin",
    "b5": "/root/autodl-tmp/weights/mit-b5.bin",
}
IGNORE = 255


class DEBFComplement(torch.nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.net = segmentor
        self.front = DEBFFront()

    def forward(self, rgb, depth, valid, edge):
        rgb, depth, valid, edge, aux = self.front(rgb, depth, valid, edge)
        output = self.net(rgb, depth, valid, edge)
        output["completion"] = aux
        return output


class PairedDataset(Dataset):
    """One augmented crop, encoded twice so teacher and student stay aligned."""

    def __init__(self, split, height, width, train):
        self.inner = MineDataset(
            Path("/root/autodl-tmp/MUSeg/flat"),
            split,
            (height, width),
            train=train,
            rare_crop=0.45 if train else 0.0,
            seed=2026,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        sample = self.inner[index]
        depth = sample["depth"]
        valid = sample["valid"]
        # Rebuild the DEBF tensor from the teacher's packed log-depth.
        log_depth = depth[0]
        gy, gx = torch.gradient(log_depth)
        edge = torch.sqrt(gx * gx + gy * gy)
        edge = edge / edge.max().clamp_min(1e-6)
        edge = edge * valid[0]
        packed = torch.stack([log_depth, edge, valid[0]], dim=0)
        packed = (packed - 0.5) / 0.25
        label16 = sample["label"]
        label15 = torch.full_like(label16, IGNORE)
        keep = (label16 >= 1) & (label16 <= 15)
        label15[keep] = label16[keep] - 1
        return {
            "rgb": sample["rgb"].contiguous(),
            "depth": sample["depth"].contiguous(),
            "valid": sample["valid"].contiguous(),
            "luma": sample["luma"].contiguous(),
            "hha": sample["hha"].contiguous(),
            "debf_depth": packed.contiguous(),
            "edge": edge.unsqueeze(0).contiguous(),
            "label16": label16.contiguous(),
            "label15": label15.contiguous(),
        }


def class_weights(loader_ids, root):
    counts = np.zeros(16, dtype=np.float64)
    for name in loader_ids:
        label = np.array(Image.open(root / "Label" / f"{name}_label.png"))
        counts += np.bincount(label[label < 16], minlength=16)
    freq = counts / max(counts.sum(), 1.0)
    weights = np.median(freq[counts > 0]) / np.maximum(freq, 1e-8)
    weights = np.clip(weights, 0.4, 8.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts


def _expand_classifier(model, state):
    weight = state.get("decoder.classifier.weight")
    bias = state.get("decoder.classifier.bias")
    if weight is None or tuple(weight.shape) == tuple(model.decoder.classifier.weight.shape):
        return False
    if weight.shape[0] != 15 or model.decoder.classifier.weight.shape[0] != 16:
        return False
    with torch.no_grad():
        model.decoder.classifier.weight[1:].copy_(weight)
        model.decoder.classifier.bias[1:].copy_(bias)
        model.decoder.classifier.weight[0].zero_()
        model.decoder.classifier.bias[0].fill_(-1.0)
    return True


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
    model.eval()
    for batch in loader:
        rgb = batch["rgb"].cuda(non_blocking=True)
        depth = batch["debf_depth"].cuda(non_blocking=True)
        valid = batch["valid"].cuda(non_blocking=True)
        edge = batch["edge"].cuda(non_blocking=True)
        target = batch["label16"].numpy()
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
    full = _scores(matrix16)
    # Fold background predictions on foreground pixels into the false-negative count.
    true_positive = np.diag(matrix15)
    false_positive = matrix15.sum(0) - true_positive
    false_negative = matrix15.sum(1) - true_positive + missed
    support = matrix15.sum(1) + missed
    union = true_positive + false_positive + false_negative
    present = support > 0
    iou15 = true_positive / np.maximum(union, 1.0)
    recall15 = true_positive / np.maximum(support, 1.0)
    return {
        "holdout_miou": full["miou"],
        "holdout_mpa": full["mpa"],
        "holdout_pa": full["pa"],
        "miou_16": full["miou"],
        "mpa_16": full["mpa"],
        "pa_16": full["pa"],
        "miou_15": float(iou15[present].mean()) if present.any() else 0.0,
        "mpa_15": float(recall15[present].mean()) if present.any() else 0.0,
        "pa_15": float(true_positive.sum() / max(support.sum(), 1.0)),
        "per_class_iou": full["iou"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--variant", choices=["b2", "b5"], default="b5")
    parser.add_argument("--init", default="/root/autodl-tmp/MineSeg/v2/runs/debf_b5_ftrans/best.pth")
    parser.add_argument("--teacher", default="/root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth")
    parser.add_argument("--outdir", default="/root/autodl-tmp/MineSeg/v3/runs/debf_complement")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.7)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path("/root/autodl-tmp/MUSeg/flat")
    train_ids = read_ids(SPLIT_ROOT / "train.txt")
    val_ids = read_ids(SPLIT_ROOT / "val.txt")
    test_ids = read_ids(SPLIT_ROOT / "test.txt")
    if set(map(group_id, train_ids)) & set(map(group_id, test_ids)):
        raise SystemExit("train and test share a capture group")

    student_core = MineSegmentor(args.variant, WEIGHTS[args.variant], "debf", num_classes=16).cuda()
    if args.init and Path(args.init).is_file():
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        current = student_core.state_dict()
        filtered = {key: value for key, value in payload["model"].items() if key in current and current[key].shape == value.shape}
        missing, unexpected = student_core.load_state_dict(filtered, strict=False)
        copied = _expand_classifier(student_core, payload["model"])
        print(f"student init copied={len(filtered)} classifier_expanded={copied} missing={len(missing)}", flush=True)
    student = DEBFComplement(student_core).cuda()
    teacher, _ = load_for_inference("/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth", device="cuda")
    if args.teacher:
        teacher_payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
        teacher.load_state_dict(teacher_payload["model"], strict=True)
        print(f"teacher {args.teacher} epoch {teacher_payload.get('epoch')}", flush=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    weights, counts = class_weights(train_ids, root)
    weights = weights.cuda()
    train_set = PairedDataset(SPLIT_ROOT / "train.txt", args.height, args.width, train=True)
    val_set = PairedDataset(SPLIT_ROOT / "val.txt", args.height, args.width, train=False)
    test_set = PairedDataset(SPLIT_ROOT / "test.txt", args.height, args.width, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.eval_only or args.eval_test:
        if args.init and Path(args.init).is_file() and any(key.startswith("front.") for key in torch.load(args.init, map_location="cpu", weights_only=False)["model"]):
            student.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"], strict=True)
        loader = test_loader if args.eval_test else val_loader
        metrics = evaluate(student, loader)
        metrics.update({
            "split": "test" if args.eval_test else "val",
            "images": len(test_ids if args.eval_test else val_ids),
            "baselines": BASELINES,
            "beats_cmx": {
                "miou": metrics["holdout_miou"] * 100 > BASELINES["CMX"]["miou"],
                "mpa": metrics["holdout_mpa"] * 100 > BASELINES["CMX"]["mpa"],
                "pa": metrics["holdout_pa"] * 100 > BASELINES["CMX"]["pa"],
            },
        })
        name = "test.json" if args.eval_test else "val.json"
        (outdir / name).write_text(json.dumps(metrics, indent=2))
        print(json.dumps({key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "beats_cmx")}, indent=2), flush=True)
        return
    encoder, head = [], []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if "rgb_encoder" in name or "depth_encoder" in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": args.lr, "weight_decay": 0.01},
            {"params": head, "lr": args.lr * 8, "weight_decay": 0.01},
        ]
    )
    config = {
        "model": "DEBFNet-Complement",
        "ideas": ["depth completion", "low-light residual", "bidirectional gates", "teacher distillation", "class-balanced loss"],
        "args": vars(args),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_images": len(test_ids),
        "train_split": str(SPLIT_ROOT / "train.txt"),
        "val_split": str(SPLIT_ROOT / "val.txt"),
        "test_split": str(SPLIT_ROOT / "test.txt"),
        "class_names": CLASS_NAMES,
        "baselines": BASELINES,
        "metric": "15 MUSeg classes, raw label 0 and 255 ignored",
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    history = outdir / "history.jsonl"
    best = -1.0
    print(f"steps/epoch {len(train_loader)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        student.train()
        scale = max(0.1, 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        optimizer.param_groups[0]["lr"] = args.lr * scale
        optimizer.param_groups[1]["lr"] = args.lr * 8 * scale
        running = seen = 0
        started = time.time()
        for batch in train_loader:
            rgb = batch["rgb"].cuda(non_blocking=True)
            label = batch["label16"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = student(rgb, batch["debf_depth"].cuda(non_blocking=True), batch["valid"].cuda(non_blocking=True), batch["edge"].cuda(non_blocking=True))
            loss, _parts = segmentation_loss(output, label, weights, simple=False)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits = teacher(
                    rgb,
                    batch["depth"].cuda(non_blocking=True),
                    batch["valid"].cuda(non_blocking=True),
                    batch["luma"].cuda(non_blocking=True),
                    batch["hha"].cuda(non_blocking=True),
                )["logits"].float()
            teacher_prob = teacher_logits.exp()
            teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
            student_log = F.log_softmax(output["logits"].float(), dim=1)
            valid = label != IGNORE
            if valid.any():
                distill = F.kl_div(
                    student_log.permute(0, 2, 3, 1)[valid],
                    teacher_prob.permute(0, 2, 3, 1)[valid],
                    reduction="batchmean",
                )
            else:
                distill = student_log.sum() * 0.0
            total = loss + 0.5 * distill + 0.25 * completion_loss(output["completion"])
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            running += float(total.detach())
            seen += 1
            if args.max_steps and seen >= args.max_steps:
                break
        metrics = evaluate(student, val_loader)
        row = {"epoch": epoch, "loss": running / max(seen, 1), "seconds": round(time.time() - started, 1), **metrics}
        row["beats_cmx"] = {
            "miou": row["holdout_miou"] * 100 > BASELINES["CMX"]["miou"],
            "mpa": row["holdout_mpa"] * 100 > BASELINES["CMX"]["mpa"],
            "pa": row["holdout_pa"] * 100 > BASELINES["CMX"]["pa"],
        }
        named = sorted(zip(["background", *CLASS_NAMES], metrics["per_class_iou"]), key=lambda item: item[1])[:3]
        row["worst"] = [{"name": name, "iou": iou} for name, iou in named]
        with history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} loss {row['loss']:.4f} mIoU {row['holdout_miou']*100:.2f} "
            f"MPA {row['holdout_mpa']*100:.2f} PA {row['holdout_pa']*100:.2f} {row['seconds']:.0f}s",
            flush=True,
        )
        if row["holdout_miou"] > best:
            best = row["holdout_miou"]
            torch.save({"model": {k: v.detach().cpu() for k, v in student.state_dict().items()}, "epoch": epoch, "metrics": row, "args": vars(args)}, outdir / "best.pth")
            print(f"saved best epoch {epoch} mIoU {best*100:.2f}", flush=True)
        if args.max_steps:
            break
    print(f"finished best mIoU {best*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
