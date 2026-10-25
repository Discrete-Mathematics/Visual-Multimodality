"""Train DEBFNet or SelectiveMine on the official MUSeg training images."""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data import (
    TEST_LIST,
    TRAIN_LIST,
    VAL_LIST,
    MUSegRGBD,
    build_index,
    build_paste_bank,
    class_weights_from_counts,
    read_ids,
    scan_labels,
)
from engine import CLASS_NAMES, ModelEMA, apply_state, evaluate_loader, scores_from_confusion, segmentation_loss
from models import MineSegmentor

ROOT = Path("/root/autodl-tmp/MineSeg/v2")
WEIGHTS = {"b2": "/root/autodl-tmp/weights/mit-b2.bin", "b5": "/root/autodl-tmp/weights/mit-b5.bin"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RareSampler(torch.utils.data.Sampler):
    def __init__(self, ids, rare_ids, seed):
        self.ids = list(ids)
        rare = set(rare_ids)
        self.rare = [index for index, sample_id in enumerate(self.ids) if sample_id in rare]
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(len(self.ids)):
            if self.rare and rng.random() < 0.5:
                yield rng.choice(self.rare)
            else:
                yield rng.randrange(len(self.ids))


def param_groups(model):
    encoder, head_decay, head_nodecay = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim == 1 or name.endswith("bias") or "norm" in name.lower() or "layernorm" in name.lower()
        if name.startswith("rgb_encoder") or name.startswith("depth_encoder"):
            encoder.append(parameter)
        elif no_decay:
            head_nodecay.append(parameter)
        else:
            head_decay.append(parameter)
    return encoder, head_decay, head_nodecay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["debf", "selective"], required=True)
    parser.add_argument("--variant", choices=["b2", "b5"], default="b2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=6e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", default="")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--init", default="", help="load model weights before fine-tuning; optimizer starts fresh")
    parser.add_argument("--warmup", type=int, default=-1)
    parser.add_argument("--paste", action="store_true")
    parser.add_argument("--rare-sampler", action="store_true")
    parser.add_argument("--simple-loss", action="store_true")
    parser.add_argument("--with-background", action="store_true")
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        raise SystemExit("height and width must be multiples of 32")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    outdir = Path(args.outdir) if args.outdir else ROOT / "runs" / f"{args.model}_{args.variant}"
    outdir.mkdir(parents=True, exist_ok=True)

    index = build_index()
    train_ids = read_ids(TRAIN_LIST)
    val_ids = read_ids(VAL_LIST)
    test_ids = read_ids(TEST_LIST)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise SystemExit("train/val/test image overlap")
    fit_ids, hold_ids = list(train_ids), list(val_ids)
    if args.limit_train:
        fit_ids = fit_ids[: args.limit_train]
        hold_ids = hold_ids[: max(4, args.limit_train // 4)]
    (outdir / "fit_ids.txt").write_text("\n".join(fit_ids) + "\n")
    (outdir / "holdout_ids.txt").write_text("\n".join(hold_ids) + "\n")
    (outdir / "official_test_ids.txt").write_text("\n".join(test_ids) + "\n")

    print(f"scanning {len(fit_ids)} fit labels", flush=True)
    counts, rare_ids = scan_labels(fit_ids, index, with_background=args.with_background)
    num_classes = 16 if args.with_background else 15
    weights = class_weights_from_counts(counts, clip=(0.4, 8.0))
    paste_bank, paste_counts = ([], {})
    if args.paste:
        paste_bank, paste_counts = build_paste_bank(fit_ids, index, with_background=args.with_background)
    print(f"paste bank {paste_counts} rare images {len(rare_ids)}", flush=True)

    depth_dropout = 0.15 if args.model == "selective" else 0.0
    fit_set = MUSegRGBD(fit_ids, index, args.height, args.width, train=True, paste_bank=paste_bank, depth_dropout=depth_dropout, with_background=args.with_background)
    hold_set = MUSegRGBD(hold_ids, index, args.height, args.width, train=False, with_background=args.with_background)
    sampler = RareSampler(fit_ids, rare_ids, args.seed) if args.rare_sampler else None
    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
        loader_kwargs["persistent_workers"] = True
    fit_loader = DataLoader(
        fit_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
        **loader_kwargs,
    )
    hold_loader = DataLoader(
        hold_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        **loader_kwargs,
    )

    model = MineSegmentor(args.variant, WEIGHTS[args.variant], args.model, num_classes=num_classes).to(device)
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(
            f"loaded init weights from {args.init} epoch {payload.get('epoch')} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    encoder_params, head_decay, head_nodecay = param_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": head_decay, "lr": args.head_lr, "weight_decay": 0.01},
            {"params": head_nodecay, "lr": args.head_lr, "weight_decay": 0.0},
        ]
    )
    ema = ModelEMA(model)
    steps_per_epoch = max(1, len(fit_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup if args.warmup >= 0 else min(500, total_steps // 20)
    record = {
        "args": vars(args),
        "train_images": len(train_ids),
        "fit_images": len(fit_ids),
        "holdout_images": len(hold_ids),
        "official_test_images": len(test_ids),
        "official_test_list": str(TEST_LIST),
        "val_list": str(VAL_LIST),
        "train_list": str(TRAIN_LIST),
        "class_names": CLASS_NAMES,
        "class_weights": [float(v) for v in weights],
        "label_counts": [int(v) for v in counts],
        "metric": "paper metric ignores raw label 0 and 255; with-background trains a 16th background logit",
        "num_classes": num_classes,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (outdir / "config.json").write_text(json.dumps(record, indent=2))
    print(f"params {record['parameters']/1e6:.2f}M steps/epoch {steps_per_epoch}", flush=True)

    best_miou = -1.0
    history_path = outdir / "history.jsonl"
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if sampler is not None:
            sampler.epoch = epoch
        running = 0.0
        seen = 0
        started = time.time()
        for batch in fit_loader:
            global_step += 1
            progress = global_step / max(1, total_steps)
            if global_step <= warmup:
                scale = global_step / max(1, warmup)
            else:
                cosine = 0.5 * (1.0 + math.cos(math.pi * (global_step - warmup) / max(1, total_steps - warmup)))
                scale = max(0.01, cosine)
            optimizer.param_groups[0]["lr"] = args.lr * scale
            optimizer.param_groups[1]["lr"] = args.head_lr * scale
            optimizer.param_groups[2]["lr"] = args.head_lr * scale
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            edge = batch["edge"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(rgb, depth, valid, edge)
            loss, parts = segmentation_loss(output, target, weights, simple=args.simple_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            running += float(loss.detach())
            seen += 1
            if args.max_steps and seen >= args.max_steps:
                break
        backup = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        apply_state(model, {key: value.to(device) for key, value in ema.shadow.items()})
        holdout = evaluate_loader(model, hold_loader, device, log_space=(args.model == "selective"), num_classes=num_classes)
        paper = holdout.get("paper")
        select_miou = paper["miou"] if paper else holdout["miou"]
        apply_state(model, {key: value.to(device) for key, value in backup.items()})
        elapsed = time.time() - started
        named_iou = paper["iou"] if paper else holdout["iou"]
        worst = sorted(zip(CLASS_NAMES, named_iou), key=lambda item: item[1])[:3]
        row = {
            "epoch": epoch,
            "loss": running / max(1, seen),
            "holdout_miou": select_miou,
            "holdout_mpa": paper["mpa"] if paper else holdout["mpa"],
            "holdout_pa": paper["pa"] if paper else holdout["pa"],
            "miou_16": holdout["miou"] if paper else None,
            "seconds": round(elapsed, 1),
            "worst": [{"name": name, "iou": iou} for name, iou in worst],
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} loss {row['loss']:.4f} "
            f"holdout mIoU {select_miou*100:.2f} MPA {row['holdout_mpa']*100:.2f} "
            f"PA {row['holdout_pa']*100:.2f} worst {worst[0][0]}={worst[0][1]:.3f} {elapsed:.0f}s",
            flush=True,
        )
        if select_miou > best_miou:
            best_miou = select_miou
            torch.save(
                {
                    "model": {key: value.cpu() for key, value in ema.shadow.items()},
                    "epoch": epoch,
                    "holdout": {key: value for key, value in holdout.items() if key != "per_mine"},
                    "args": vars(args),
                },
                outdir / "best.pth",
            )
            print(f"saved best epoch {epoch} mIoU {best_miou*100:.2f}", flush=True)
        if args.max_steps:
            break
    torch.save({"model": ema.shadow, "epoch": args.epochs, "args": vars(args)}, outdir / "last_ema.pth")
    print(f"finished best holdout mIoU {best_miou*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
