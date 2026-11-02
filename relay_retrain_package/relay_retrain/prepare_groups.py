"""Freeze validation-only grouping thresholds before any test score exists."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    APPEARANCE_IDS,
    BOUNDARY_RADIUS,
    DATA_ROOT,
    GEOMETRY_IDS,
    HEIGHT,
    OCCLUSION_RADIUS,
    RUN,
    SMALL_AREA,
    SOURCE,
    SPLIT_DIR,
    WIDTH,
    WORKERS,
    rule_text,
)

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))
from mine_relay.data import MineDataset, read_ids  # noqa: E402
from mine_relay.legacy.dataset import LOG_DEPTH_HIGH, LOG_DEPTH_LOW  # noqa: E402

OUT = RUN / "protocol.json"


def _hist_update(hist, depth, valid):
    span = float(LOG_DEPTH_HIGH - LOG_DEPTH_LOW)
    norm = depth[:, 0].numpy()
    mask = valid[:, 0].numpy() > 0
    mm = np.exp(norm * span + float(LOG_DEPTH_LOW))
    clipped = np.clip(np.rint(mm).astype(np.int32), 0, hist.size - 1)
    chosen = clipped[mask]
    if chosen.size:
        hist += np.bincount(chosen, minlength=hist.size)


def _percentile_from_hist(hist, quantile):
    total = int(hist.sum())
    if total <= 0:
        raise RuntimeError("validation depth histogram is empty")
    cum = np.cumsum(hist)
    index = int(np.searchsorted(cum, quantile * total, side="left"))
    return int(min(index, hist.size - 1)), total


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    val_ids = read_ids(SPLIT_DIR / "val.txt")
    dataset = MineDataset(
        DATA_ROOT, SPLIT_DIR / "val.txt", (HEIGHT, WIDTH),
        train=False, paired=False, rare_crop=0.0, corruption="none", seed=2026,
    )
    if dataset.ids != val_ids:
        raise SystemExit("validation ids drifted")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=WORKERS, pin_memory=False)
    hist = np.zeros(8001, dtype=np.int64)
    rows = []
    for batch in loader:
        _hist_update(hist, batch["depth"], batch["valid"])
        coverages = batch["coverage"].tolist() if torch.is_tensor(batch["coverage"]) else list(batch["coverage"])
        brightness = batch["brightness"].tolist() if torch.is_tensor(batch["brightness"]) else list(batch["brightness"])
        for name, coverage, bright in zip(batch["id"], coverages, brightness):
            rows.append({"id": name, "coverage": float(coverage), "brightness": float(bright)})
    if len(rows) != len(val_ids) or len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("validation metadata does not match the split")
    far_mm, valid_pixels = _percentile_from_hist(hist, 0.75)
    coverage = np.array([row["coverage"] for row in rows], dtype=np.float64)
    bright = np.array([row["brightness"] for row in rows], dtype=np.float64)
    coverage_median = float(np.median(coverage))
    bright_p20 = float(np.percentile(bright, 20))
    bright_p50 = float(np.percentile(bright, 50))
    ordered = sorted(rows, key=lambda row: (row["coverage"], row["id"]))
    positive = [row for row in ordered if row["coverage"] > 0]
    viz = (positive or ordered)[:4]
    high = sum(row["coverage"] >= coverage_median for row in rows)
    low_light = sum(row["brightness"] <= bright_p20 for row in rows)
    normal_light = sum(row["brightness"] >= bright_p50 for row in rows)
    protocol = {
        "written_before_training": True,
        "uses_test_scores": False,
        "val_images": len(rows),
        "coverage_median": coverage_median,
        "brightness_p20": bright_p20,
        "brightness_p50": bright_p50,
        "far_mm_p75": far_mm,
        "far_valid_pixels": valid_pixels,
        "far_bin_mm": 1,
        "small_area": SMALL_AREA,
        "occlusion_radius": OCCLUSION_RADIUS,
        "boundary_radius": BOUNDARY_RADIUS,
        "rule_text": rule_text(),
        "group_rules": {
            "depth_high": "原图深度有效像素比例 >= 验证集中位数",
            "depth_low": "原图深度有效像素比例 < 验证集中位数",
            "light_low": "原图平均亮度 <= 验证集 20 分位",
            "light_normal": "原图平均亮度 >= 验证集 50 分位；中间图像不进入这两个照度组",
            "far": "评测分辨率上，有效深度毫米值 >= 验证集有效深度的 75 分位（1 毫米直方图）",
            "small": f"前景类八连通域面积 < {SMALL_AREA} 像素",
            "occlusion": f"有效深度像素，且落在无效深度膨胀 {OCCLUSION_RADIUS} 像素以内",
            "boundary": f"语义标签边界膨胀 {BOUNDARY_RADIUS} 像素，忽略标签 255",
        },
        "geometry_class_ids": list(GEOMETRY_IDS),
        "appearance_class_ids": list(APPEARANCE_IDS),
        "val_group_image_counts": {
            "depth_high": int(high),
            "depth_low": int(len(rows) - high),
            "light_low": int(low_light),
            "light_normal": int(normal_light),
        },
        "viz_ids": [row["id"] for row in viz],
        "viz_rows": viz,
        "images": {row["id"]: {"coverage": row["coverage"], "brightness": row["brightness"]} for row in rows},
        "log_depth_low": float(LOG_DEPTH_LOW),
        "log_depth_high": float(LOG_DEPTH_HIGH),
        "note": (
            "阈值只由验证集的深度和亮度算出，不用预测，也不用测试集。"
            "可视化图像是验证集上有效深度比例大于 0 的最低 4 张，并列时按图像 id 排序。"
        ),
    }
    if not math.isfinite(coverage_median) or not math.isfinite(bright_p20):
        raise SystemExit("non-finite threshold")
    OUT.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(
        f"wrote {OUT} median_coverage {coverage_median:.4f} "
        f"bright_p20 {bright_p20:.4f} bright_p50 {bright_p50:.4f} far_mm {far_mm} "
        f"viz {protocol['viz_ids']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
