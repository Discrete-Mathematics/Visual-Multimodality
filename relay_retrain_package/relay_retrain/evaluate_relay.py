"""Score the selected checkpoints on the test split and on pre-registered groups.

Validation is re-scored only to check that the metric code matches the number
stored at selection time. Thresholds come from protocol.json.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_ZH,
    DATA_ROOT,
    GROUP_ORDER,
    HEIGHT,
    PAPER,
    RUN,
    SEEDS,
    SOURCE,
    SPLIT_DIR,
    WIDTH,
    WORKERS,
    apply_variant,
    mean_std,
    param_counts,
    run_dir,
    variant_ids,
)

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))
sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")

from finetune_selective import PAPER_CLASSES, _accumulate, _pack, prepare_model, seed_everything  # noqa: E402
from mine_relay.data import MineDataset, read_ids  # noqa: E402
from mine_relay.engine import forward_batch  # noqa: E402
from mine_relay.legacy.labels import PALETTE  # noqa: E402
from patch_model import checkpoint_rgb_encoder, install  # noqa: E402

install()

EVAL_DIR = RUN / "eval"
FIGURES = PAPER / "figures"


def load_protocol():
    path = RUN / "protocol.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run prepare_groups.py before training")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("uses_test_scores"):
        raise SystemExit("protocol was not frozen before test scores")
    return protocol


def empty_bucket():
    return {
        "paper": np.zeros((PAPER_CLASSES, PAPER_CLASSES), dtype=np.float64),
        "missed": np.zeros(PAPER_CLASSES, dtype=np.float64),
        "matrix": np.zeros((16, 16), dtype=np.float64),
        "pixels": 0,
        "images": 0,
    }


def add_masked(bucket, pred, gt, mask):
    labeled = (gt != 255) & mask
    count = int(labeled.sum())
    if count == 0:
        return
    gated = np.array(gt, copy=True)
    gated[~mask] = 255
    _accumulate(bucket["paper"], bucket["missed"], bucket["matrix"], pred[None], gated[None])
    bucket["pixels"] += count
    bucket["images"] += 1


def pack_bucket(bucket):
    if bucket["pixels"] <= 0:
        return {"pa_16": None, "mpa_16": None, "miou_16": None, "pa_15": None, "mpa_15": None,
                "miou_15": None, "pixels": 0, "images": 0, "per_class_iou_16": None}
    packed = _pack(bucket["paper"], bucket["missed"], bucket["matrix"], "holdout_")
    matrix = bucket["matrix"]
    true_positive = np.diag(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    union = support + predicted - true_positive
    iou = true_positive / np.maximum(union, 1.0)
    return {
        "pa_16": packed["holdout_pa_16"],
        "mpa_16": packed["holdout_mpa_16"],
        "miou_16": packed["holdout_miou_16"],
        "pa_15": packed["holdout_pa"],
        "mpa_15": packed["holdout_mpa"],
        "miou_15": packed["holdout_miou"],
        "per_class_iou_16": iou.tolist(),
        "per_class_support_16": support.astype(np.int64).tolist(),
        "per_class_iou_15": packed["per_class_iou"],
        "pixels": int(bucket["pixels"]),
        "images": int(bucket["images"]),
    }


def depth_mm(norm, low, high):
    return np.exp(norm * (high - low) + low)


def boundary_mask(label, radius):
    valid = label != 255
    edge = np.zeros(label.shape, dtype=bool)
    edge[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (label[:, 1:] != label[:, :-1])
    edge[1:, :] |= valid[1:, :] & valid[:-1, :] & (label[1:, :] != label[:-1, :])
    if radius > 0:
        edge = ndimage.binary_dilation(edge, iterations=radius)
    return edge & valid


def small_mask(label, area):
    # The frozen protocol names 8-connected components. ndimage.label defaults to 4.
    structure = np.ones((3, 3), dtype=int)
    out = np.zeros(label.shape, dtype=bool)
    for cls in range(1, 16):
        region = label == cls
        if not region.any():
            continue
        labeled, count = ndimage.label(region, structure=structure)
        if count <= 0:
            continue
        sizes = np.bincount(labeled.ravel())
        tiny = np.flatnonzero((np.arange(sizes.size) != 0) & (sizes < area) & (sizes > 0))
        if tiny.size:
            out |= np.isin(labeled, tiny)
    return out


def occlusion_mask(valid, label, radius):
    invalid = ~(valid > 0)
    near = ndimage.binary_dilation(invalid, iterations=radius) if radius > 0 else invalid
    return near & (valid > 0) & (label != 255)


def make_loader(split, seed):
    dataset = MineDataset(
        DATA_ROOT, split, (HEIGHT, WIDTH),
        train=False, paired=False, rare_crop=0.0, corruption="none", seed=seed,
    )
    return DataLoader(dataset, batch_size=4, shuffle=False, num_workers=WORKERS, pin_memory=True)


def score_loader(model, loader, protocol, with_groups):
    overall = empty_bucket()
    groups = {name: empty_bucket() for name in GROUP_ORDER} if with_groups else {}
    low = float(protocol["log_depth_low"])
    high = float(protocol["log_depth_high"])
    n_scored = 0
    for batch in loader:
        with torch.no_grad():
            output = forward_batch(model, batch)
        prediction = output["logits"].argmax(1).cpu().numpy()
        target = batch["label"].numpy()
        depth = batch["depth"].numpy()
        valid = batch["valid"].numpy()
        coverages = batch["coverage"].tolist() if torch.is_tensor(batch["coverage"]) else list(batch["coverage"])
        brightness = batch["brightness"].tolist() if torch.is_tensor(batch["brightness"]) else list(batch["brightness"])
        for index, name in enumerate(batch["id"]):
            pred = prediction[index]
            gt = target[index]
            add_masked(overall, pred, gt, np.ones(gt.shape, dtype=bool))
            if not with_groups:
                continue
            coverage = float(coverages[index])
            bright = float(brightness[index])
            image_mask = np.ones(gt.shape, dtype=bool)
            if coverage >= protocol["coverage_median"]:
                add_masked(groups["depth_high"], pred, gt, image_mask)
            else:
                add_masked(groups["depth_low"], pred, gt, image_mask)
            if bright <= protocol["brightness_p20"]:
                add_masked(groups["light_low"], pred, gt, image_mask)
            elif bright >= protocol["brightness_p50"]:
                add_masked(groups["light_normal"], pred, gt, image_mask)
            mm = np.clip(np.rint(depth_mm(depth[index, 0], low, high)).astype(np.int32), 0, 8000)
            seen = valid[index, 0] > 0
            add_masked(groups["far"], pred, gt, (mm >= int(protocol["far_mm_p75"])) & seen)
            add_masked(groups["small"], pred, gt, small_mask(gt, int(protocol["small_area"])))
            add_masked(groups["occlusion"], pred, gt, occlusion_mask(seen, gt, int(protocol["occlusion_radius"])))
            add_masked(groups["boundary"], pred, gt, boundary_mask(gt, int(protocol["boundary_radius"])))
            del name
        n_scored += int(prediction.shape[0])
        if n_scored % 100 < int(prediction.shape[0]):
            print(f"scored {n_scored} images groups={with_groups}", flush=True)
    packed = {"overall": pack_bucket(overall)}
    if with_groups:
        packed["groups"] = {name: pack_bucket(bucket) for name, bucket in groups.items()}
    return packed


def load_model(variant, seed):
    outdir = run_dir(variant, seed)
    selection = json.loads((outdir / "selection.json").read_text(encoding="utf-8"))
    if not selection.get("complete"):
        raise SystemExit(f"{variant} seed {seed} is not a finished run")
    payload = torch.load(outdir / "best.pth", map_location="cpu", weights_only=False)
    if payload.get("args", {}).get("variant") != variant or int(payload["args"]["seed"]) != int(seed):
        raise SystemExit(f"checkpoint identity mismatch for {variant} seed {seed}")
    model = prepare_model("cuda")
    checkpoint_rgb_encoder(model)
    model.load_state_dict(payload["model"], strict=True)
    apply_variant(model, variant)
    model.eval()
    return model, selection, payload


def measure_efficiency(model, seed):
    dataset = MineDataset(
        DATA_ROOT, SPLIT_DIR / "val.txt", (HEIGHT, WIDTH),
        train=False, paired=False, rare_crop=0.0, corruption="none", seed=seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    counted = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    started = False
    for batch in loader:
        with torch.no_grad():
            if counted < 5:
                forward_batch(model, batch)
                counted += 1
                if counted == 5:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    starter.record()
                    started = True
                continue
            forward_batch(model, batch)
        counted += 1
        if counted >= 35:
            break
    if not started or counted < 35:
        raise SystemExit("efficiency loop did not see 35 images")
    ender.record()
    torch.cuda.synchronize()
    return {
        "milliseconds_per_image": starter.elapsed_time(ender) / 30.0,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "timed_images": 30,
        "warmup_images": 5,
        "batch_size": 1,
        **param_counts(model),
    }


def check_val_gap(selection, scored):
    stored = selection["metrics"]
    got = scored["overall"]
    gaps = {
        "pa_16": abs(got["pa_16"] - stored["pa_16"]),
        "mpa_16": abs(got["mpa_16"] - stored["mpa_16"]),
        "miou_16": abs(got["miou_16"] - stored["miou_16"]),
    }
    # Same weights and the same metric code. A gap above half a percentage point
    # means the checkpoint and the saved validation row are not the same run.
    if max(gaps.values()) > 5e-3:
        raise SystemExit(f"validation rescore disagrees with the saved selection: {gaps}")
    return {"stored": {key: stored[key] for key in ("pa_16", "mpa_16", "miou_16", "epoch")}, "abs_gap": gaps}


def colorize(label):
    canvas = np.zeros((*label.shape, 3), dtype=np.uint8)
    for index, color in enumerate(PALETTE):
        canvas[label == index] = color
    return canvas


def denormalize_hha(hha):
    # HHA is stored as four channels. The first three are the normalized color image.
    image = hha[:3]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = np.clip(image * std + mean, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def upsample_map(tensor, size):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    scaled = F.interpolate(tensor.float(), size=size, mode="bilinear", align_corners=False)
    array = scaled[0, 0].detach().cpu().numpy()
    peak = float(array.max())
    if peak > 0:
        array = array / peak
    return array


def predict_one(model, row):
    batch = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    with torch.no_grad():
        output = forward_batch(model, batch)
    pred = output["logits"].argmax(1)[0].detach().cpu().numpy()
    relay = output.get("relay") or []
    maps = {}
    if relay:
        maps["support"] = upsample_map(relay[0]["support"], pred.shape)
        if "neighbor_mass" in relay[0]:
            maps["neighbor_mass"] = upsample_map(relay[0]["neighbor_mass"], pred.shape)
    return pred, maps, batch


def save_contact_sheet(protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = MineDataset(
        DATA_ROOT, SPLIT_DIR / "val.txt", (HEIGHT, WIDTH),
        train=False, paired=False, rare_crop=0.0, corruption="none", seed=2026,
    )
    index = {name: position for position, name in enumerate(dataset.ids)}
    chosen = protocol["viz_ids"]
    full_model, _, _ = load_model("full", 2026)
    apply_variant(full_model, "full", export_weights=True)
    full_model.eval()
    full_rows = []
    for name in chosen:
        pred, maps, batch = predict_one(full_model, dataset[index[name]])
        full_rows.append((name, pred, maps, batch))
    del full_model
    torch.cuda.empty_cache()
    other_model, _, _ = load_model("no_relay", 2026)
    other_model.eval()
    other = {}
    for name in chosen:
        pred, _maps, _batch = predict_one(other_model, dataset[index[name]])
        other[name] = pred
    del other_model
    torch.cuda.empty_cache()

    FIGURES.mkdir(parents=True, exist_ok=True)
    columns = ["RGB", "invalid depth", "HHA", "neighbor mass", "full", "removed relay", "difference"]
    figure, axes = plt.subplots(len(chosen), len(columns), figsize=(16, 2.4 * len(chosen)), squeeze=False)
    for row_index, (name, pred, maps, batch) in enumerate(full_rows):
        rgb = batch["rgb"][0].numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        rgb = np.clip(np.transpose(rgb * std + mean, (1, 2, 0)), 0, 1)
        valid = batch["valid"][0, 0].numpy() > 0
        invalid = np.zeros((*valid.shape, 3), dtype=np.float32)
        invalid[~valid] = (1.0, 0.25, 0.25)
        invalid[valid] = (0.15, 0.15, 0.15)
        hha = denormalize_hha(batch["hha"][0].numpy())
        mass = maps.get("neighbor_mass", maps.get("support"))
        diff = pred != other[name]
        panels = [
            rgb,
            invalid,
            hha,
            mass,
            colorize(pred),
            colorize(other[name]),
            np.stack([diff, np.zeros_like(diff), np.zeros_like(diff)], axis=-1),
        ]
        meta = protocol["images"][name]
        for column, panel in enumerate(panels):
            ax = axes[row_index][column]
            if panel.ndim == 2:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(columns[column], fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{name}\nvalid {meta['coverage'] * 100:.1f}%", fontsize=7)
        disagree = float(diff[batch["label"][0].numpy() != 255].mean()) if np.any(batch["label"][0].numpy() != 255) else 0.0
        axes[row_index][-1].set_xlabel(f"diff {disagree * 100:.2f}%", fontsize=8)
    figure.tight_layout()
    path = FIGURES / "contact.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print(f"wrote {path}", flush=True)
    return str(path)


def summarize(records):
    summary = {}
    by_variant = {name: [] for name in variant_ids()}
    for record in records:
        by_variant[record["variant"]].append(record)
    full = {int(record["seed"]): record for record in by_variant["full"]}
    for name, rows in by_variant.items():
        rows = sorted(rows, key=lambda record: int(record["seed"]))
        metrics = {}
        for key in ("pa_16", "mpa_16", "miou_16", "pa_15", "mpa_15", "miou_15"):
            mu, sd = mean_std([record["test"][key] for record in rows])
            metrics[key] = {"mean": mu, "std": sd}
        deltas = {}
        if name != "full":
            for key in ("pa_16", "mpa_16", "miou_16"):
                paired = [record["test"][key] - full[int(record["seed"])]["test"][key] for record in rows]
                mu, sd = mean_std(paired)
                deltas[key] = {"mean": mu, "std": sd, "per_seed": paired}
        summary[name] = {"metrics": metrics, "delta_vs_full": deltas, "seeds": [int(record["seed"]) for record in rows]}
    return summary


def main():
    protocol = load_protocol()
    seed_everything(2026)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = []
    for variant in variant_ids():
        for seed in SEEDS:
            path = EVAL_DIR / f"{variant}_seed_{seed}.json"
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
                print(f"reuse {path.name}", flush=True)
                continue
            print(f"eval {variant} seed {seed}", flush=True)
            model, selection, _payload = load_model(variant, seed)
            efficiency = measure_efficiency(model, seed) if seed == SEEDS[0] else None
            val_loader = make_loader(SPLIT_DIR / "val.txt", seed)
            test_loader = make_loader(SPLIT_DIR / "test.txt", seed)
            if len(read_ids(SPLIT_DIR / "test.txt")) != 788 or len(read_ids(SPLIT_DIR / "val.txt")) != 788:
                raise SystemExit("unexpected split length")
            val_scored = score_loader(model, val_loader, protocol, with_groups=False)
            gap = check_val_gap(selection, val_scored)
            test_scored = score_loader(model, test_loader, protocol, with_groups=True)
            record = {
                "variant": variant,
                "seed": int(seed),
                "selection": {
                    "epoch": selection["selected_epoch"],
                    "floor_met": selection["floor_met"],
                    "fallback": selection["fallback"],
                    "val_pa_16": selection["metrics"]["pa_16"],
                    "val_mpa_16": selection["metrics"]["mpa_16"],
                    "val_miou_16": selection["metrics"]["miou_16"],
                },
                "val_check": gap,
                "test": test_scored["overall"],
                "groups": test_scored["groups"],
                "efficiency": efficiency,
            }
            path.write_text(json.dumps(record) + "\n")
            records.append(record)
            del model
            torch.cuda.empty_cache()
            print(
                f"test {variant} seed {seed} PA {record['test']['pa_16'] * 100:.2f} "
                f"MPA {record['test']['mpa_16'] * 100:.2f} mIoU {record['test']['miou_16'] * 100:.2f}",
                flush=True,
            )
    if len(records) != len(variant_ids()) * len(SEEDS):
        raise SystemExit("missing records")
    contact = save_contact_sheet(protocol)
    result = {
        "class_names": CLASS_ZH,
        "seeds": list(SEEDS),
        "variants": variant_ids(),
        "protocol_file": str(RUN / "protocol.json"),
        "records": records,
        "summary": summarize(records),
        "contact_sheet": contact,
        "test_used_for_selection": False,
    }
    out = RUN / "results.json"
    out.write_text(json.dumps(result) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
