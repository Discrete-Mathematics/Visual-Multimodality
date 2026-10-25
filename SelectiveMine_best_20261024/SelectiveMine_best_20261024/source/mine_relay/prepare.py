"""Run on AutoDL: audit training data and create grouped development splits."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from .data import read_ids, group_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.05 <= args.val_fraction <= 0.4:
        raise ValueError("Use a development fraction between 0.05 and 0.4")
    train = read_ids(args.data_root / "train.txt")
    test = read_ids(args.data_root / "val.txt")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("Repeated ids in official split")
    overlap = set(map(group_id, train)) & set(map(group_id, test))
    if set(train) & set(test) or overlap:
        raise ValueError(f"Official train/test overlap: groups={sorted(overlap)[:10]}")
    groups, mine_groups, counts, summary = defaultdict(list), defaultdict(list), {}, []
    for i, name in enumerate(train):
        gid = group_id(name)
        groups[gid].append(name)
        label = np.array(Image.open(args.data_root / "Label" / (name + "_label.png")))
        if not np.isin(label, list(range(16)) + [255]).all():
            raise ValueError(f"Unknown labels in {name}")
        rgb = Image.open(args.data_root / "RGB" / (name + ".jpg"))
        depth = np.array(Image.open(args.data_root / "Depth" / (name + ".png")))
        if label.shape != depth.shape or rgb.size != (label.shape[1], label.shape[0]):
            raise ValueError(f"Unaligned {name}")
        if depth.dtype != np.uint16:
            raise ValueError(f"Depth must be uint16: {name}, {depth.dtype}")
        counts[name] = np.bincount(label[label < 16], minlength=16)
        summary.append({"id": name, "group": gid, "mine": name.split("-")[0],
                        "depth_valid_fraction": float((depth > 0).mean()),
                        "rgb_mean": float(np.array(rgb, dtype=np.float32).mean()),
                        "class_pixels": counts[name].tolist()})
        if (i + 1) % 200 == 0:
            print(f"audited training samples {i + 1}/{len(train)}", flush=True)
    for gid in groups:
        mine_groups[gid.split("-")[0]].append(gid)
    selected = None
    # Selection uses class coverage only, never model performance or test labels.
    for attempt in range(100):
        rng = random.Random(args.seed + attempt)
        val_groups = set()
        for mine in sorted(mine_groups):
            candidates = sorted(mine_groups[mine])
            rng.shuffle(candidates)
            n = max(1, min(len(candidates) - 1, round(len(candidates) * args.val_fraction)))
            val_groups.update(candidates[:n])
        dev_val = [n for n in train if group_id(n) in val_groups]
        dev_train = [n for n in train if group_id(n) not in val_groups]
        a = np.sum([counts[n] for n in dev_train], axis=0)
        b = np.sum([counts[n] for n in dev_val], axis=0)
        if (a > 0).all() and (b > 0).all():
            selected = dev_train, dev_val, attempt
            break
    if selected is None:
        raise RuntimeError("Could not put every class in both development splits; inspect group rarity")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory; do not silently change a frozen split")
    dev_train, dev_val, attempt = selected
    splits = {"dev_train": dev_train, "dev_val": dev_val, "official_train": train, "official_test": test}
    manifest = {"seed": args.seed, "coverage_retry": attempt,
                "group_rule": "first four hyphen-separated fields", "splits": {}}
    for name, ids in splits.items():
        content = "\n".join(ids) + "\n"
        (args.output / (name + ".txt")).write_text(content, encoding="utf-8")
        manifest["splits"][name] = {"images": len(ids), "groups": len(set(map(group_id, ids))),
                                   "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (args.output / "training_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
