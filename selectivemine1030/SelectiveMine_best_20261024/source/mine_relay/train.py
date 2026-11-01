"""AutoDL trainer: grouped development, historical continuation or fixed final run."""

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import MineRelay
from .data import MineDataset, read_ids
from .losses import recovery_loss, segmentation_loss
from .engine import (require_cuda, seed_everything, seed_worker, fingerprint,
                     validate_protocol, class_weights, forward_batch,
                     evaluate, save_checkpoint, write_json, source_fingerprints)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-split", type=Path, required=True)
    p.add_argument("--val-split", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--protocol", choices=["development", "continuation", "final"], default="development")
    p.add_argument("--mode", choices=["logits", "inject", "relay"], default="relay")
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument("--imagenet", type=Path)
    init.add_argument("--hha", type=Path)
    init.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--aux", type=float, default=0.2)
    p.add_argument("--paired-weight", type=float, default=0.0)
    p.add_argument("--recovery-weight", type=float, default=0.0)
    p.add_argument("--quality-weight", type=float, default=0.0)
    p.add_argument("--rare-crop", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-hours", type=float, default=2.0)
    p.add_argument("--hourly-rate", type=float, help="Actual instance CNY/hour; omit when unknown")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="Smoke only; invalid for research comparisons")
    return p.parse_args()


def main():
    args = arguments()
    require_cuda()
    seed_everything(args.seed)
    if args.epochs < 1 or args.max_hours <= 0 or args.max_steps < 0 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid epoch count, wall-time cap, smoke steps, batch size or worker count")
    if args.hourly_rate is not None and (not math.isfinite(args.hourly_rate) or args.hourly_rate <= 0):
        raise ValueError("Hourly rate must be finite and positive")
    if args.mode != "relay" and (args.recovery_weight or args.quality_weight):
        raise ValueError("Recovery and quality supervision require relay mode")
    if not args.paired_weight and (args.recovery_weight or args.quality_weight):
        raise ValueError("Set paired-weight > 0 for masked recovery")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint:
        old = checkpoint["config"]
        for key in ("mode", "protocol", "seed", "height", "width", "epochs", "batch_size", "workers",
                    "lr", "aux", "paired_weight", "recovery_weight", "quality_weight", "rare_crop", "max_steps", "patience"):
            if old[key] != getattr(args, key):
                raise ValueError(f"Resume mismatch for {key}: {old[key]} != {getattr(args, key)}")
        from_hha = checkpoint["initialization"] == "historical_hha"
    else:
        from_hha = args.hha is not None
    validate_protocol(args.data_root, args.train_split, args.val_split, args.protocol, from_hha)
    if args.protocol == "continuation" and not from_hha:
        raise ValueError("Continuation requires historical HHA initialization")
    if args.protocol == "final" and args.max_steps:
        raise ValueError("Smoke-step limits cannot produce a final experiment")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.run_dir / "config.json").exists():
        raise FileExistsError("Choose a fresh run directory")
    split_hashes = {"train": fingerprint(args.train_split),
                    "val": fingerprint(args.val_split) if args.val_split else None}
    if checkpoint and checkpoint["split_hashes"] != split_hashes:
        raise ValueError("Resume split hashes changed")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    source_hashes = source_fingerprints()
    if checkpoint and checkpoint["source_hashes"] != source_hashes:
        raise ValueError("Code changed since checkpoint; start a named new experiment instead")
    write_json(args.run_dir / "config.json", config)
    write_json(args.run_dir / "environment.json", {"python": platform.python_version(),
               "torch": str(torch.__version__), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "source_hashes": source_hashes,
               "split_hashes": split_hashes})
    train = MineDataset(args.data_root, args.train_split, (args.height, args.width), train=True,
                        paired=args.paired_weight > 0, rare_crop=args.rare_crop)
    weights, counts = class_weights(args.data_root, train.ids)
    write_json(args.run_dir / "class_weights.json", {"counts": counts.tolist(), "weights": weights.cpu().tolist()})
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=generator)
    if not len(loader):
        raise ValueError("Batch size is larger than the training split")
    val_loader = None
    if args.val_split:
        val_loader = DataLoader(MineDataset(args.data_root, args.val_split, (args.height, args.width)),
                                batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    model = MineRelay(args.mode)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.configure_training(joint=not from_hha)
        model.mix.fill_(1.0)
    elif args.hha:
        model.initialize_hha(args.hha)
    else:
        model.initialize_imagenet(args.imagenet)
    model.cuda()
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "encoder" in name:
            multiplier = 0.25 if from_hha else 1.0
        elif name.startswith("base.decoder") and from_hha:
            multiplier = 0.05
        elif name.startswith(("relay.", "inject.")):
            multiplier = 4.0
        else:
            multiplier = 0.5 if from_hha else 4.0
        decay = 0.0 if parameter.ndim <= 1 else 0.01
        groups.setdefault((multiplier, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": params, "lr": args.lr * scale,
                                   "base_lr": args.lr * scale, "weight_decay": decay}
                                  for (scale, decay), params in groups.items()])
    start, best, best_epoch = 0, -1.0, -1
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, best_epoch = checkpoint["next_epoch"], checkpoint["best"], checkpoint["best_epoch"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        generator.set_state(rng["loader"])
    unused_prefix = {"logits": ("relay.", "inject."), "inject": ("relay.",), "relay": ("inject.",)}[args.mode]
    params = {"total_registered": sum(p.numel() for p in model.parameters()),
              "active_inference": sum(p.numel() for n, p in model.named_parameters() if not n.startswith(unused_prefix)),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    print(json.dumps({"config": config, "parameters": params}), flush=True)
    if from_hha and not checkpoint:
        baseline = evaluate(model, val_loader)
        write_json(args.run_dir / "initial_evaluation.json", baseline)
        initial = baseline["all"]["miou"]
        print(f"historical baseline mIoU={initial:.6f}", flush=True)
        if abs(initial - 0.55) > 0.005:
            raise RuntimeError("Historical baseline did not reproduce 55.00% within 0.5 percentage point")
        best = initial
    initial_hash = checkpoint["initial_hash"] if checkpoint else fingerprint(args.hha or args.imagenet)
    started = time.monotonic()
    cumulative = checkpoint.get("elapsed_seconds", 0.0) if checkpoint else 0.0
    prior_cost = checkpoint.get("gpu_cost_estimate_yuan") if checkpoint else 0.0
    reason = "epochs_complete"
    for epoch in range(start, args.epochs):
        model.train()
        sums, steps, partial = {}, 0, False
        epoch_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader):
            progress = epoch + step / len(loader)
            warmup = min(2, max(1, args.epochs // 10))
            factor = min(1.0, (progress + 1) / warmup)
            if progress >= warmup:
                factor = 0.5 * (1 + math.cos(math.pi * (progress - warmup) / max(args.epochs - warmup, 1)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            label = batch["label"].cuda(non_blocking=True)
            clean = forward_batch(model, batch)
            main = segmentation_loss(clean, label, weights, args.aux)
            loss = main
            values = {"seg": float(main.detach())}
            if args.paired_weight > 0:
                damaged = forward_batch(model, batch, damaged=True)
                paired = segmentation_loss(damaged, label, weights, args.aux)
                recover, quality = recovery_loss(clean, damaged, batch["hidden"].cuda(non_blocking=True))
                loss = ((loss + args.paired_weight * paired) / (1.0 + args.paired_weight)
                        + args.recovery_weight * recover + args.quality_weight * quality)
                values.update(paired=float(paired.detach()), recovery=float(recover.detach()), quality=float(quality.detach()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                                error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + value
            if steps % 10 == 0:
                print(json.dumps({"epoch": epoch, "step": steps, "loss": float(loss.detach()),
                                  "grad_norm": float(norm), "elapsed_min": (time.monotonic() - started) / 60}), flush=True)
            if time.monotonic() - started >= args.max_hours * 3600:
                reason, partial = "wall_time_cap", steps < len(loader)
                break
            if args.max_steps and steps >= args.max_steps:
                reason, partial = "smoke_only", steps < len(loader)
                break
        metrics = evaluate(model, val_loader, max_batches=2 if args.max_steps else None) if val_loader and not partial else None
        improved = metrics is not None and metrics["all"]["miou"] > best
        if improved:
            best, best_epoch = metrics["all"]["miou"], epoch
        elapsed = cumulative + time.monotonic() - started
        record = {"epoch": epoch, "steps": steps, "partial": partial,
                  "loss": {key: value / max(steps, 1) for key, value in sums.items()},
                  "metrics": metrics, "best": best, "best_epoch": best_epoch,
                  "epoch_seconds": time.monotonic() - epoch_start, "elapsed_seconds": elapsed,
                  "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                  "hourly_rate_yuan": args.hourly_rate,
                  "gpu_cost_estimate_yuan": (prior_cost + (time.monotonic() - started) / 3600 * args.hourly_rate)
                      if prior_cost is not None and args.hourly_rate is not None else None,
                  "smoke_only": bool(args.max_steps)}
        with (args.run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                   "next_epoch": epoch if partial else epoch + 1, "partial_epoch_replayed_on_resume": partial,
                   "best": best, "best_epoch": best_epoch, "split_hashes": split_hashes,
                   "source_hashes": source_hashes, "initial_hash": initial_hash,
                   "train_ids": train.ids, "val_ids": read_ids(args.val_split) if args.val_split else [],
                   "initialization": "historical_hha" if from_hha else "imagenet",
                   "elapsed_seconds": elapsed,
                   "gpu_cost_estimate_yuan": record["gpu_cost_estimate_yuan"],
                   "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                           "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                           "loader": generator.get_state()}}
        save_checkpoint(args.run_dir / "last.pth", payload)
        if improved:
            save_checkpoint(args.run_dir / "best.pth", payload)
        print(json.dumps(record), flush=True)
        if reason != "epochs_complete":
            break
        if time.monotonic() - started >= args.max_hours * 3600:
            reason = "wall_time_cap"
            break
        if args.protocol != "final" and best_epoch >= 0 and epoch - best_epoch >= args.patience:
            reason = "development_patience"
            break
    write_json(args.run_dir / "status.json", {"reason": reason, "best": best, "best_epoch": best_epoch,
               "elapsed_seconds": cumulative + time.monotonic() - started,
               "hourly_rate_yuan": args.hourly_rate,
               "gpu_cost_estimate_yuan": (prior_cost + (time.monotonic() - started) / 3600 * args.hourly_rate)
                   if prior_cost is not None and args.hourly_rate is not None else None,
               "note": "wall cap stops training, not AutoDL billing; power off manually when finished"})


if __name__ == "__main__":
    main()
