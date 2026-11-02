"""Retrain SelectiveMine with one Geometry Relay factor changed.

The data split, epoch budget, input size, loss, and validation selection rule
match selective_pa3. The test split is checked for leakage and is not scored.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BATCH_SIZE,
    BG_BOOST,
    DATA_ROOT,
    EPOCHS,
    HEIGHT,
    INIT,
    LOVASZ,
    LR,
    PA_FLOOR_MIOU,
    PA_FLOOR_MPA,
    PIXEL_CE,
    RARE_CROP,
    SOURCE,
    SPLIT_DIR,
    WIDTH,
    WORKERS,
    apply_variant,
    param_counts,
    run_dir,
    variant_ids,
)

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))
sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")

from engine import lovasz_softmax  # noqa: E402
from finetune_selective import evaluate, prepare_model, seed_everything  # noqa: E402
from mine_relay.data import MineDataset, group_id, read_ids  # noqa: E402
from mine_relay.engine import class_weights, forward_batch  # noqa: E402
from mine_selective.losses import selective_loss  # noqa: E402
from patch_model import checkpoint_rgb_encoder, install  # noqa: E402

install()


def free_gb(path="/root/autodl-tmp"):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / (1024 ** 3)


def read_history(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def already_done(outdir):
    rows = read_history(outdir / "history.jsonl")
    selection = outdir / "selection.json"
    if len(rows) < EPOCHS or not (outdir / "best.pth").is_file() or not selection.is_file():
        return False
    payload = json.loads(selection.read_text(encoding="utf-8"))
    return bool(payload.get("complete")) and int(payload.get("epochs_ran", 0)) == EPOCHS


def archive_partial(outdir):
    if not outdir.exists():
        return
    stamp = time.strftime("%Y%m%d%H%M%S")
    for name in ("history.jsonl", "selection.json", "config.json"):
        path = outdir / name
        if path.exists():
            path.rename(outdir / f"{name}.{stamp}.partial")
    best = outdir / "best.pth"
    if best.exists():
        best.unlink()


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def loss_of(output, label, weights):
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
    if LOVASZ > 0:
        loss = loss + LOVASZ * lovasz_softmax(output["logits"].exp(), label)
    if PIXEL_CE > 0:
        loss = loss + PIXEL_CE * F.nll_loss(output["logits"].float(), label, ignore_index=255)
    return loss


def build_loader(split, seed, train, workers=WORKERS):
    dataset = MineDataset(
        DATA_ROOT,
        split,
        (HEIGHT, WIDTH),
        train=train,
        paired=False,
        rare_crop=RARE_CROP if train else 0.0,
        corruption="none",
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
    )


def load_initialized(device):
    model = prepare_model(device)
    checkpoint_rgb_encoder(model)
    payload = torch.load(INIT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    print(f"loaded {INIT} epoch {payload.get('epoch')}", flush=True)
    return model, payload.get("epoch")


def smoke():
    seed_everything(2026)
    model, _epoch = load_initialized("cuda")
    train_split = SPLIT_DIR / "train.txt"
    loader = build_loader(train_split, 2026, True, workers=0)
    batch = next(iter(loader))
    weights, _counts = class_weights(DATA_ROOT, read_ids(train_split))
    weights = weights.clone()
    weights[0] = weights[0] * BG_BOOST
    for name in variant_ids():
        apply_variant(model, name)
        model.train()
        output = forward_batch(model, batch)
        label = batch["label"].cuda(non_blocking=True)
        loss = loss_of(output, label, weights)
        loss.backward()
        value = float(loss.detach())
        model.zero_grad(set_to_none=True)
        counts = param_counts(model)
        print(
            f"smoke {name} loss {value:.4f} finite {math.isfinite(value)} "
            f"trainable {counts['trainable'] / 1e6:.2f}M active {counts['active'] / 1e6:.2f}M "
            f"relay {counts['relay'] / 1e6:.2f}M",
            flush=True,
        )
        if not math.isfinite(value):
            raise SystemExit(f"smoke failed {name}")
        torch.cuda.empty_cache()
    print("SMOKE_OK", flush=True)


def train_one(variant, seed):
    if free_gb() < 6:
        raise SystemExit(f"disk free {free_gb():.2f} GB, refuse to start")
    outdir = run_dir(variant, seed)
    outdir.mkdir(parents=True, exist_ok=True)
    if already_done(outdir):
        print(f"skip complete {variant} seed {seed}", flush=True)
        return
    archive_partial(outdir)
    seed_everything(seed)
    device = "cuda"
    model, init_epoch = load_initialized(device)
    apply_variant(model, variant)
    train_split = SPLIT_DIR / "train.txt"
    val_split = SPLIT_DIR / "val.txt"
    test_split = SPLIT_DIR / "test.txt"
    train_ids = read_ids(train_split)
    val_ids = read_ids(val_split)
    test_ids = read_ids(test_split)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise SystemExit("split leakage")
    if set(map(group_id, train_ids)) & set(map(group_id, val_ids)):
        raise SystemExit("train and validation share a capture group")
    weights, counts = class_weights(DATA_ROOT, train_ids)
    weights = weights.clone()
    weights[0] = weights[0] * BG_BOOST
    counts_now = param_counts(model)
    config = {
        "variant": variant,
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "height": HEIGHT,
        "width": WIDTH,
        "lr": LR,
        "workers": WORKERS,
        "rare_crop": RARE_CROP,
        "lovasz_weight": LOVASZ,
        "pixel_ce": PIXEL_CE,
        "bg_boost": BG_BOOST,
        "select_pa": True,
        "pa_floor_miou": PA_FLOOR_MIOU,
        "pa_floor_mpa": PA_FLOOR_MPA,
        "init": str(INIT),
        "init_epoch": init_epoch,
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_images_not_scored": len(test_ids),
        "params": counts_now,
        "class_counts": [int(v) for v in counts],
        "class_weights": [float(v) for v in weights.detach().cpu()],
        "splits": {"train": str(train_split), "val": str(val_split), "test": str(test_split)},
        "test_used_for_selection": False,
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    (outdir / "params.json").write_text(json.dumps(counts_now, indent=2))
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if "encoder" in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": LR, "weight_decay": 0.01},
            {"params": head, "lr": LR * 4, "weight_decay": 0.01},
        ]
    )
    train_loader = build_loader(train_split, seed, True)
    val_loader = build_loader(val_split, seed, False)
    history_path = outdir / "history.jsonl"
    best_floor = None
    best_fallback = None
    print(
        f"steps/epoch {len(train_loader)} variant {variant} seed {seed} "
        f"trainable {counts_now['trainable'] / 1e6:.2f}M",
        flush=True,
    )
    for epoch in range(1, EPOCHS + 1):
        if free_gb() < 6:
            raise SystemExit(f"disk free {free_gb():.2f} GB at epoch {epoch}")
        apply_variant(model, variant)
        model.train()
        scale = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(EPOCHS, 1)))
        scale = max(0.05, scale)
        optimizer.param_groups[0]["lr"] = LR * scale
        optimizer.param_groups[1]["lr"] = LR * 4 * scale
        running, seen = 0.0, 0
        started = time.time()
        for batch in train_loader:
            label = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, batch)
            loss = loss_of(output, label, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.multimodal.mix.fill_(1.0)
            running += float(loss.detach())
            seen += 1
        # The last training graph stays referenced by the loop variables.
        # Validation only needs predictions, so drop that graph before scoring.
        del output, loss, label, batch
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with torch.no_grad():
            metrics = evaluate(model, val_loader, tta=False)
        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "variant": variant,
            "seed": seed,
            "loss": running / max(seen, 1),
            "seconds": round(elapsed, 1),
            "holdout_miou": metrics["holdout_miou"],
            "holdout_mpa": metrics["holdout_mpa"],
            "holdout_pa": metrics["holdout_pa"],
            "miou_16": metrics["miou_16"],
            "mpa_16": metrics["mpa_16"],
            "pa_16": metrics["pa_16"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{EPOCHS} {variant} seed {seed} loss {row['loss']:.4f} "
            f"16-class mIoU {row['miou_16'] * 100:.2f} MPA {row['mpa_16'] * 100:.2f} "
            f"PA {row['pa_16'] * 100:.2f} {elapsed:.0f}s",
            flush=True,
        )
        state = clone_state(model)
        candidate = {"epoch": epoch, "metrics": row, "state": state}
        if best_fallback is None or row["miou_16"] > best_fallback["metrics"]["miou_16"]:
            best_fallback = candidate
        floor_ok = row["miou_16"] >= PA_FLOOR_MIOU and row["mpa_16"] >= PA_FLOOR_MPA
        if floor_ok and (best_floor is None or row["pa_16"] > best_floor["metrics"]["pa_16"]):
            best_floor = candidate
            _write_checkpoint(outdir, variant, seed, best_floor, floor_met=True, complete=False)
            print(f"saved floor epoch {epoch} PA {row['pa_16'] * 100:.2f}", flush=True)
    chosen = best_floor if best_floor is not None else best_fallback
    if chosen is None:
        raise SystemExit("no epoch was evaluated")
    _write_checkpoint(outdir, variant, seed, chosen, floor_met=best_floor is not None, complete=True)
    print(
        f"finished {variant} seed {seed} epoch {chosen['epoch']} "
        f"floor_met {best_floor is not None} PA {chosen['metrics']['pa_16'] * 100:.2f}",
        flush=True,
    )


def _write_checkpoint(outdir, variant, seed, chosen, floor_met, complete):
    payload = {
        "model": chosen["state"],
        "epoch": chosen["epoch"],
        "metrics": {key: value for key, value in chosen["metrics"].items()},
        "args": {"variant": variant, "seed": seed, "epochs": EPOCHS},
    }
    torch.save(payload, outdir / "best.pth")
    selection = {
        "complete": complete,
        "epochs_ran": chosen["epoch"] if not complete else EPOCHS,
        "selected_epoch": chosen["epoch"],
        "floor_met": floor_met,
        "fallback": None if floor_met else "highest validation 16-class mIoU",
        "rule": "select-pa: highest validation 16-class PA among epochs with mIoU and MPA floors",
        "pa_floor_miou": PA_FLOOR_MIOU,
        "pa_floor_mpa": PA_FLOOR_MPA,
        "metrics": chosen["metrics"],
        "test_used": False,
    }
    (outdir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")


def main():
    if "--smoke" in sys.argv:
        smoke()
        return
    if len(sys.argv) != 3:
        raise SystemExit("usage: train_relay_ablation.py VARIANT SEED")
    variant, seed = sys.argv[1], int(sys.argv[2])
    if variant not in variant_ids():
        raise SystemExit(f"unknown variant {variant}")
    train_one(variant, seed)


if __name__ == "__main__":
    main()
