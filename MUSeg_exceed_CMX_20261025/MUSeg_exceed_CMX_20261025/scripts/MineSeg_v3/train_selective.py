"""Fine-tune SelectiveMine with depth completion and low-light residual.

Splits are FTransUNet train.txt / val.txt / test.txt. Checkpoint selection uses
val.txt only. The 15-class score ignores raw labels 0 and 255, matching the
comparison used against the MUSeg paper baselines.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

os.environ.setdefault("OMP_NUM_THREADS", "1")

V2 = Path("/root/autodl-tmp/MineSeg/v2")
SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(V2))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from complement import ComplementFront, completion_loss
from engine import lovasz_softmax
from finetune_selective import evaluate, seed_everything
from mine_relay.data import MineDataset, group_id, read_ids
from mine_relay.engine import class_weights, forward_batch
from mine_selective.losses import selective_loss
from selective_inference_bundle_20261023 import load_for_inference

BASELINES = {
    "CMX": {"pa": 86.30, "mpa": 72.72, "miou": 61.83},
    "DFormer": {"pa": 85.63, "mpa": 70.37, "miou": 59.74},
    "SA-Gate": {"pa": 84.77, "mpa": 71.34, "miou": 59.38},
    "CMNeXt": {"pa": 84.05, "mpa": 66.77, "miou": 56.06},
}
HARD_IDS = (6, 10, 11, 12, 13, 14)
SPLIT_ROOT = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
INIT_RELEASED = "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth"


class ComplementSelective(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.front = ComplementFront()

    @property
    def multimodal(self):
        return self.backbone.multimodal

    def forward(self, rgb, depth, valid, luma, hha):
        rgb, depth, valid, luma, hha, aux = self.front(rgb, depth, valid, luma, hha)
        output = self.backbone(rgb, depth, valid, luma, hha)
        output["completion"] = aux
        return output


def load_model(checkpoint, device):
    backbone, _metadata = load_for_inference(INIT_RELEASED, device=device)
    model = ComplementSelective(backbone).to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        wrapped = any(key.startswith("front.") or key.startswith("backbone.") for key in state)
        if wrapped:
            model.load_state_dict(state, strict=True)
        else:
            model.backbone.load_state_dict(state, strict=True)
        epoch = payload.get("epoch") if isinstance(payload, dict) else "?"
        print(f"loaded {'wrapper' if wrapped else 'backbone'} {checkpoint} epoch {epoch}", flush=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.train()
    return model


def sample_weights(root, ids):
    import numpy as np
    from PIL import Image

    weights = np.ones(len(ids), dtype=np.float64)
    hard = 0
    for index, name in enumerate(ids):
        label = np.array(Image.open(root / "Label" / f"{name}_label.png"))
        if np.isin(label, HARD_IDS).any():
            weights[index] = 4.0
            hard += 1
    return torch.as_tensor(weights, dtype=torch.double), hard


def beats(metrics):
    """Compare the 16-class scores. Those are the ones on the same scale as Table 5:
    pixel accuracy near 86% only appears when unlabeled/background pixels are counted.
    """
    return {
        name: {
            "pa": metrics["pa_16"] * 100 > row["pa"],
            "mpa": metrics["mpa_16"] * 100 > row["mpa"],
            "miou": metrics["miou_16"] * 100 > row["miou"],
        }
        for name, row in BASELINES.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth")
    parser.add_argument("--outdir", default="/root/autodl-tmp/MineSeg/v3/runs/selective_complement")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        raise SystemExit("height and width must be multiples of 32")

    seed_everything(args.seed)
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.32)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path("/root/autodl-tmp/MUSeg/flat")
    train_split = SPLIT_ROOT / "train.txt"
    val_split = SPLIT_ROOT / "val.txt"
    test_split = SPLIT_ROOT / "test.txt"
    train_ids, val_ids, test_ids = read_ids(train_split), read_ids(val_split), read_ids(test_split)
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise SystemExit("split leakage")
    if set(map(group_id, train_ids)) & set(map(group_id, test_ids)):
        raise SystemExit("train and test share a capture group")

    model = load_model(args.checkpoint, device)

    def make_loader(split, train, sampler=None):
        dataset = MineDataset(root, split, (args.height, args.width), train=train, rare_crop=0.45 if train else 0.0, seed=args.seed)
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
        split = test_split if args.eval_test else val_split
        metrics = evaluate(model, make_loader(split, False), tta=args.tta)
        metrics.update(split=str(split), images=len(read_ids(split)), tta=args.tta, beats=beats(metrics), baselines=BASELINES)
        name = ("test" if args.eval_test else "val") + ("_tta" if args.tta else "") + ".json"
        (outdir / name).write_text(json.dumps(metrics, indent=2))
        print(json.dumps({key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "beats")}, indent=2), flush=True)
        return

    sampling, hard_count = sample_weights(root, train_ids)
    sampler = WeightedRandomSampler(sampling, len(train_ids), replacement=True)
    train_loader = make_loader(train_split, True, sampler)
    val_loader = make_loader(val_split, False)
    weights, counts = class_weights(root, train_ids)
    for class_id in HARD_IDS:
        weights[class_id] *= 1.6
    weights /= weights.mean()
    config = {
        "model": "SelectiveMine-Complement",
        "ideas": ["modality imbalance reweighting", "depth completion", "low-light residual", "hard-class sampling"],
        "args": vars(args),
        "train_split": str(train_split),
        "val_split": str(val_split),
        "test_split": str(test_split),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "test_images": len(test_ids),
        "hard_images": hard_count,
        "class_counts": [int(v) for v in counts],
        "baselines": BASELINES,
        "metric": "checkpoint selected by 16-class mIoU (background included), which matches Table 5's PA scale; 15-class scores are also logged",
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    front_params = list(model.front.parameters())
    encoder, head = [], []
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad:
            (encoder if "encoder" in name else head).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder, "lr": args.lr, "weight_decay": 0.01},
            {"params": head, "lr": args.lr * 4, "weight_decay": 0.01},
            {"params": front_params, "lr": args.lr * 8, "weight_decay": 0.0},
        ]
    )
    history = outdir / "history.jsonl"
    best = -1.0
    print(f"steps/epoch {len(train_loader)} hard {hard_count}/{len(train_ids)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
        scale = max(0.1, 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / args.epochs)))
        optimizer.param_groups[0]["lr"] = args.lr * scale
        optimizer.param_groups[1]["lr"] = args.lr * 4 * scale
        optimizer.param_groups[2]["lr"] = args.lr * 8 * scale
        running = seen = 0
        started = time.time()
        for batch in train_loader:
            label = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, batch)
            loss, _parts = selective_loss(
                output, label, class_weights=weights, rgb_weight=0.5, fused_weight=0.5, geo_weight=0.2, utility_weight=0.05, temperature=0.5
            )
            loss = loss + 0.35 * lovasz_softmax(output["logits"].exp(), label)
            loss = loss + 0.25 * completion_loss(output["completion"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.multimodal.mix.fill_(1.0)
            running += float(loss.detach())
            seen += 1
            if args.max_steps and seen >= args.max_steps:
                break
        metrics = evaluate(model, val_loader)
        row = {
            "epoch": epoch,
            "loss": running / max(seen, 1),
            "seconds": round(time.time() - started, 1),
            **{key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "mpa_16", "pa_16")},
            "beats_cmx": beats(metrics)["CMX"],
        }
        with history.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}/{args.epochs} loss {row['loss']:.4f} paper mIoU {row['holdout_miou']*100:.2f} "
            f"MPA {row['holdout_mpa']*100:.2f} PA {row['holdout_pa']*100:.2f} 16-class mIoU {row['miou_16']*100:.2f} "
            f"beats {row['beats_cmx']} {row['seconds']:.0f}s",
            flush=True,
        )
        if row["miou_16"] > best:
            best = row["miou_16"]
            torch.save(
                {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "epoch": epoch, "metrics": row, "args": vars(args)},
                outdir / "best.pth",
            )
            print(f"saved best epoch {epoch} 16-class mIoU {best*100:.2f}", flush=True)
        if args.max_steps:
            break
    print(f"finished best 16-class mIoU {best*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
