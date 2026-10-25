"""Evaluate the exported SelectiveMine weights on FTransUNet/test.txt."""

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from selective_inference_bundle_20261023 import load_for_inference
from mine_relay.data import MineDataset, read_ids
from mine_relay.engine import evaluate, require_cuda, seed_everything

MUSEG = Path("/root/autodl-tmp/MUSeg")
FLAT = Path("/root/autodl-tmp/SelectiveMine_best_20261024/museg_flat_test")
SPLIT_SRC = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/test.txt")
SPLIT = Path("/root/autodl-tmp/SelectiveMine_best_20261024/ftransunet_test.txt")
WEIGHTS = Path("/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth")
OUT_DIR = Path("/root/autodl-tmp/SelectiveMine_best_20261024/results")
FT_DIR = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")


def link_split(ids):
    mines = [p for p in sorted(MUSEG.iterdir()) if p.is_dir() and (p / "Image").is_dir()]
    index = {}
    for mine in mines:
        for image in (mine / "Image").glob("*.jpg"):
            index[image.stem] = mine
    for name in ("RGB", "Depth", "Label"):
        (FLAT / name).mkdir(parents=True, exist_ok=True)
    missing = []
    for name in ids:
        mine = index.get(name)
        if mine is None:
            missing.append(name)
            continue
        mapping = {
            FLAT / "RGB" / f"{name}.jpg": mine / "Image" / f"{name}.jpg",
            FLAT / "Depth" / f"{name}.png": mine / "Depth" / f"{name}.png",
            FLAT / "Label" / f"{name}_label.png": mine / "Label" / f"{name}_label.png",
        }
        for link, target in mapping.items():
            if not target.is_file():
                missing.append(str(target))
                continue
            if link.is_symlink() or link.exists():
                if link.resolve() != target.resolve():
                    link.unlink()
                    link.symlink_to(target)
            else:
                link.symlink_to(target)
    if missing:
        raise FileNotFoundError(f"{len(missing)} test files missing, first: {missing[:5]}")


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_text(path, payload):
    overall = payload["metrics"]["all"]
    lines = [
        "SelectiveMine best inference on FTransUNet/test.txt",
        f"time_utc: {payload['time_utc']}",
        f"weights: {payload['weights']}",
        f"selected_epoch: {payload['selected_epoch']}",
        f"completed_epochs: {payload['completed_epochs']}",
        f"split: {payload['split']}",
        f"images: {payload['images']}",
        f"input_size: {payload['input_size']}",
        f"batch_size: {payload['batch_size']}",
        f"corruption: none",
        f"PA: {overall['pa']:.6f}",
        f"MPA: {overall['mpa']:.6f}",
        f"mIoU: {overall['miou']:.6f}",
        f"mIoU_percent: {overall['miou'] * 100:.6f}",
        f"rare_mIoU: {overall['rare_miou']:.6f}",
        "class_iou:",
    ]
    for name, score in overall["class_iou"].items():
        lines.append(f"  {name}: {score:.6f}")
    lines.append("ground_truth_pixels:")
    for name, count in overall["ground_truth_pixels"].items():
        lines.append(f"  {name}: {int(count)}")
    lines.append("regions:")
    for region, metrics in payload["metrics"].items():
        if region == "all":
            continue
        miou = "n/a" if metrics["miou"] is None else f"{metrics['miou']:.6f}"
        pa = "n/a" if metrics["pa"] is None else f"{metrics['pa']:.6f}"
        mpa = "n/a" if metrics["mpa"] is None else f"{metrics['mpa']:.6f}"
        lines.append(f"  {region}: images={metrics['images']} mIoU={miou} PA={pa} MPA={mpa}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    require_cuda()
    torch.set_num_threads(4)
    seed_everything(2026)
    ids = read_ids(SPLIT_SRC)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids in FTransUNet/test.txt")
    SPLIT.write_text("\n".join(ids) + "\n", encoding="utf-8")
    link_split(ids)
    print(f"linked {len(ids)} test images", flush=True)
    model, metadata = load_for_inference(WEIGHTS, device="cuda")
    print(
        f"loaded epoch {metadata['selected_epoch']} / {metadata['completed_epochs']}",
        flush=True,
    )
    dataset = MineDataset(FLAT, SPLIT, (480, 640), corruption="none", seed=2026)
    loader = DataLoader(dataset, batch_size=8, num_workers=8, pin_memory=True)
    metrics = to_builtin(evaluate(model, loader))
    payload = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(WEIGHTS),
        "weights_sha256": json.loads(WEIGHTS.with_name("export.json").read_text())["sha256"],
        "model_tensor_sha256": metadata["model_tensor_sha256"],
        "selected_epoch": metadata["selected_epoch"],
        "completed_epochs": metadata["completed_epochs"],
        "selection": metadata["config"]["selection"],
        "fusion": metadata["config"]["fusion"],
        "split": str(SPLIT_SRC),
        "split_copy": str(SPLIT),
        "images": len(ids),
        "data_root": str(MUSEG),
        "flat_view": str(FLAT),
        "input_size": [480, 640],
        "batch_size": 8,
        "workers": 8,
        "corruption": "none",
        "seed": 2026,
        "inference": "single resize, no TTA, CUDA bf16 autocast",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "metrics": metrics,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "ftransunet_test_txt_eval.json"
    text_path = OUT_DIR / "ftransunet_test_txt_eval.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text(text_path, payload)
    (FT_DIR / "selectivemine_best_test_results.json").write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (FT_DIR / "selectivemine_best_test_results.txt").write_text(
        text_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    overall = metrics["all"]
    print(
        json.dumps(
            {
                "images": len(ids),
                "pa": overall["pa"],
                "mpa": overall["mpa"],
                "miou": overall["miou"],
                "json": str(json_path),
                "text": str(text_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
