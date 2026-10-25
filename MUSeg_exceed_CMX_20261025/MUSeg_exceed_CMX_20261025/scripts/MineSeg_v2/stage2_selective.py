"""Continue SelectiveMine from the high-resolution checkpoint.

Images that contain the classes with the lowest IoU are sampled more often:
electronic equipment, door, tools and materials, rescue equipment, container,
and metal fixture. Selection still uses the 15 semantic classes, ignoring
raw labels 0 and 255. The official 1,576-image test list is not used for
checkpoint selection.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

os.environ.setdefault("OMP_NUM_THREADS", "1")

SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from mine_relay.data import MineDataset, group_id, read_ids
from mine_relay.engine import class_weights, forward_batch
from mine_selective.losses import selective_loss
from selective_inference_bundle_20261023 import load_for_inference

PAPER_CLASSES = 15
HARD_IDS = (6, 10, 11, 12, 13, 14)
TRAIN_SPLIT = SOURCE / "splits" / "dev_train.txt"
VAL_SPLIT = SOURCE / "splits" / "dev_val.txt"
TEST_SPLIT = Path("/root/autodl-tmp/Experiment/DatasetSplit/val.txt")
INIT_WEIGHTS = "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth"
BASELINES = {"CMX": {"pa": 86.30, "mpa": 72.72, "miou": 61.83}}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def sample_weights(root, ids):
    weights = np.ones(len(ids), dtype=np.float64)
    hard = 0
    for index, name in enumerate(ids):
        label = np.array(Image.open(root / "Label" / f"{name}_label.png"))
        if np.isin(label, HARD_IDS).any():
            weights[index] = 4.0
            hard += 1
    return torch.as_tensor(weights, dtype=torch.double), hard


def accumulate(paper, missed, matrix16, prediction, target):
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


def pack(paper, missed, matrix16):
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
    predicted16 = matrix16.sum(0)
    union16 = support16 + predicted16 - tp16
    present16 = support16 > 0
    iou16 = tp16 / np.maximum(union16, 1.0)
    recall16 = tp16 / np.maximum(support16, 1.0)
    return {
        "holdout_miou": float(iou[present].mean()) if present.any() else 0.0,
        "holdout_mpa": float(recall[present].mean()) if present.any() else 0.0,
        "holdout_pa": float(true_positive.sum() / max(support.sum(), 1.0)),
        "miou_16": float(iou16[present16].mean()) if present16.any() else 0.0,
        "mpa_16": float(recall16[present16].mean()) if present16.any() else 0.0,
        "pa_16": float(tp16.sum() / max(matrix16.sum(), 1.0)),
        "per_class_iou": iou.tolist(),
        "per_class_recall": recall.tolist(),
    }


def resize_input(tensor, size):
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    nearest = tensor.shape[1] == 1
    if nearest:
        return F.interpolate(tensor.float(), size=size, mode="nearest")
    return F.interpolate(tensor.float(), size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def predict(model, batch, tta):
    keys = ("rgb", "depth", "valid", "luma", "hha")
    base = [batch[key].cuda(non_blocking=True) for key in keys]
    height, width = base[0].shape[-2:]
    scales = (0.75, 1.0, 1.25) if tta else (1.0,)
    flips = (False, True) if tta else (False,)
    accumulated = None
    views = 0
    for scale in scales:
        size = (
            max(32, int(round(height * scale / 32.0) * 32)),
            max(32, int(round(width * scale / 32.0) * 32)),
        )
        resized = [resize_input(tensor, size) for tensor in base]
        for flip in flips:
            view = [torch.flip(tensor, dims=[-1]) if flip else tensor for tensor in resized]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*view)["logits"].float()
            if flip:
                logits = torch.flip(logits, dims=[-1])
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            probability = logits.exp()
            accumulated = probability if accumulated is None else accumulated + probability
            views += 1
    return (accumulated / views).argmax(1).cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, tta=False):
    paper = np.zeros((PAPER_CLASSES, PAPER_CLASSES), dtype=np.float64)
    missed = np.zeros(PAPER_CLASSES, dtype=np.float64)
    matrix16 = np.zeros((16, 16), dtype=np.float64)
    was_training = model.training
    model.eval()
    for batch in loader:
        accumulate(paper, missed, matrix16, predict(model, batch, tta), batch["label"].numpy())
    if was_training:
        model.train()
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
    return pack(paper, missed, matrix16)


def load_model(checkpoint):
    model, _metadata = load_for_inference(INIT_WEIGHTS, device="cuda")
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        print(f"loaded {checkpoint} epoch {payload.get('epoch')}", flush=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.train()
    return model


def beats(metrics):
    return {
        "pa": metrics["holdout_pa"] * 100 > BASELINES["CMX"]["pa"],
        "mpa": metrics["holdout_mpa"] * 100 > BASELINES["CMX"]["mpa"],
        "miou": metrics["holdout_miou"] * 100 > BASELINES["CMX"]["miou"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth")
    parser.add_argument("--outdir", default="/root/autodl-tmp/MineSeg/v2/runs/selective_hard")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--plain", action="store_true", help="keep the original class weights and uniform sampling")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path("/root/autodl-tmp/MUSeg/flat")
    train_ids, val_ids, test_ids = read_ids(TRAIN_SPLIT), read_ids(VAL_SPLIT), read_ids(TEST_SPLIT)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(map(group_id, train_ids)) & set(map(group_id, val_ids)):
        raise SystemExit("split leakage")
    model = load_model(args.checkpoint)

    def loader(split, train=False, sampler=None):
        dataset = MineDataset(root, split, (args.height, args.width), train=train, rare_crop=0.35 if train else 0.0, seed=args.seed)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=train,
            persistent_workers=args.workers > 0,
        )

    if args.eval_only or args.eval_test:
        split = TEST_SPLIT if args.eval_test else VAL_SPLIT
        started = time.time()
        metrics = evaluate(model, loader(split), tta=args.tta)
        metrics.update(split=str(split), images=len(read_ids(split)), seconds=round(time.time() - started, 1), tta=args.tta, beats_cmx=beats(metrics), baselines=BASELINES)
        name = ("official_test" if args.eval_test else "dev_val") + ("_tta" if args.tta else "") + ".json"
        (outdir / name).write_text(json.dumps(metrics, indent=2))
        print(json.dumps({key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "mpa_16", "pa_16", "beats_cmx", "tta")}, indent=2), flush=True)
        return

    sampling, hard_count = sample_weights(root, train_ids)
    print(f"hard images {hard_count}/{len(train_ids)} plain={args.plain}", flush=True)
    train_loader = loader(TRAIN_SPLIT, True, None if args.plain else WeightedRandomSampler(sampling, len(train_ids), replacement=True))
    val_loader = loader(VAL_SPLIT)
    weights, counts = class_weights(root, train_ids)
    if not args.plain:
        for class_id in HARD_IDS:
            weights[class_id] *= 1.5
        weights /= weights.mean()
    (outdir / "config.json").write_text(json.dumps({"args": vars(args), "train_images": len(train_ids), "holdout_images": len(val_ids), "official_test_images": len(test_ids), "hard_images": hard_count, "class_counts": [int(v) for v in counts]}, indent=2))
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if "encoder" in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": args.lr, "weight_decay": 0.01}, {"params": head, "lr": args.lr * 4, "weight_decay": 0.01}]
    )
    history = outdir / "history.jsonl"
    best = -1.0
    print(f"steps/epoch {len(train_loader)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
        scale = max(0.1, 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        optimizer.param_groups[0]["lr"] = args.lr * scale
        optimizer.param_groups[1]["lr"] = args.lr * 4 * scale
        running = seen = 0
        started = time.time()
        for batch in train_loader:
            label = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = selective_loss(forward_batch(model, batch), label, class_weights=weights, rgb_weight=0.5, fused_weight=0.5, geo_weight=0.2, utility_weight=0.05, temperature=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.multimodal.mix.fill_(1.0)
            running += float(loss.detach())
            seen += 1
        metrics = evaluate(model, val_loader)
        row = {"epoch": epoch, "loss": running / max(seen, 1), "seconds": round(time.time() - started, 1), **{key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "mpa_16", "pa_16")}}
        with history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} loss {row['loss']:.4f} paper mIoU {row['holdout_miou']*100:.2f} "
            f"MPA {row['holdout_mpa']*100:.2f} PA {row['holdout_pa']*100:.2f} 16-class mIoU {row['miou_16']*100:.2f} {row['seconds']:.0f}s",
            flush=True,
        )
        if row["holdout_miou"] > best:
            best = row["holdout_miou"]
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch, "metrics": row, "args": vars(args)}, outdir / "best.pth")
            print(f"saved best epoch {epoch} paper mIoU {best*100:.2f}", flush=True)
    print(f"finished best paper mIoU {best*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
