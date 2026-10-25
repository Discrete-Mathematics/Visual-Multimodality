"""Explicit split files, paired raw-depth corruption and crop geometry.

Camera-up geometry is retained for checkpoint compatibility. It is not the
gravity-aligned height-above-ground representation of canonical HHA.
"""

import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from .legacy.geometry import (FX, FY, CX, CY, NATIVE_W, NATIVE_H, encode_hha,
                              _normalize_geometry, _resize_triplet, _pad_triplet,
                              _color_jitter)
from .legacy.dataset import encode_depth, RGB_MEAN, RGB_STD
from .evaluation_stats import array_fingerprint

RARE = (1, 11, 12, 13)


def read_ids(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def group_id(name):
    pieces = name.split("-")
    if len(pieces) < 7:
        raise ValueError(f"Unexpected MUSeg id: {name}")
    return "-".join(pieces[:4])


def corrupt_depth(depth, rng, kind="holes"):
    result = depth.copy()
    h, w = depth.shape
    if kind == "dropout":
        result.fill(0)
    elif kind == "holes":
        for _ in range(int(rng.integers(1, 5))):
            ch = max(1, int(h * rng.uniform(0.08, 0.3)))
            cw = max(1, int(w * rng.uniform(0.08, 0.3)))
            y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
            result[y:y + ch, x:x + cw] = 0
        result[rng.random((h, w)) < 0.05] = 0
    elif kind == "noise":
        valid = result > 0
        noisy = result.astype(np.float32) + rng.normal(0, 50, result.shape)
        result[valid] = np.clip(noisy[valid], 1, 65535).astype(np.uint16)
    elif kind != "none":
        raise ValueError(kind)
    return result


def pack_geometry(depth, camera):
    hha, _ = encode_hha(depth, *camera)
    packed, valid = encode_depth(depth)
    return {"depth": packed, "valid": valid, "hha": _normalize_geometry(hha, valid)}


class MineDataset(torch.utils.data.Dataset):
    def __init__(self, root, split_file, size=(480, 640), train=False,
                 paired=False, rare_crop=0.0, corruption="none", seed=2026):
        self.root = Path(root)
        self.ids = read_ids(split_file)
        self.height, self.width = size
        self.train, self.paired, self.rare_crop = train, paired, rare_crop
        self.corruption, self.seed = corruption, seed
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate image ids in split")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        rgb = Image.open(self.root / "RGB" / (name + ".jpg")).convert("RGB")
        depth = np.array(Image.open(self.root / "Depth" / (name + ".png")), dtype=np.uint16)
        label = np.array(Image.open(self.root / "Label" / (name + "_label.png")), dtype=np.uint8)
        if rgb.size != (depth.shape[1], depth.shape[0]) or depth.shape != label.shape:
            raise ValueError(f"Unaligned sample {name}")
        native_w, native_h = rgb.size
        fx, fy = FX * native_w / NATIVE_W, FY * native_h / NATIVE_H
        cx, cy = CX * native_w / NATIVE_W, CY * native_h / NATIVE_H
        raw_coverage = float((depth > 0).mean())
        raw_brightness = float(np.array(rgb, dtype=np.float32).mean() / 255.0)
        mirror = False
        if self.train:
            scale = random.uniform(0.7, 1.4)
            resized_w, resized_h = max(1, int(native_w * scale)), max(1, int(native_h * scale))
            rgb, depth, label = _resize_triplet(rgb, depth, label, (resized_w, resized_h))
            fx, cx = fx * resized_w / native_w, cx * resized_w / native_w
            fy, cy = fy * resized_h / native_h, cy * resized_h / native_h
            rgb, depth, label = _pad_triplet(rgb, depth, label, self.height, self.width)
            w, h = rgb.size
            top, left = random.randint(0, h - self.height), random.randint(0, w - self.width)
            if random.random() < self.rare_crop:
                ys, xs = np.where(np.isin(label, RARE))
                if len(ys):
                    k = random.randrange(len(ys))
                    top = int(np.clip(ys[k] - random.randrange(self.height), 0, h - self.height))
                    left = int(np.clip(xs[k] - random.randrange(self.width), 0, w - self.width))
            rgb = rgb.crop((left, top, left + self.width, top + self.height))
            depth = np.ascontiguousarray(depth[top:top + self.height, left:left + self.width])
            label = np.ascontiguousarray(label[top:top + self.height, left:left + self.width])
            cx, cy = cx - left, cy - top
            if random.random() < 0.8:
                rgb = _color_jitter(rgb)
            mirror = random.random() < 0.5
            rng = np.random.default_rng(random.getrandbits(63))
        else:
            rgb, depth, label = _resize_triplet(rgb, depth, label, (self.width, self.height))
            fx, cx = fx * self.width / native_w, cx * self.width / native_w
            fy, cy = fy * self.height / native_h, cy * self.height / native_h
            digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        camera = fx, fy, cx, cy
        # Preserve the sensor's natural missingness before synthetic corruption.
        original_valid = (depth > 0)[None]
        if self.corruption in ("holes", "noise", "dropout"):
            depth = corrupt_depth(depth, rng, self.corruption)
        rgb_np = np.array(rgb, dtype=np.float32) / 255.0
        if self.corruption == "dark":
            rgb_np = np.clip(rgb_np ** 2.0, 0, 1)
        data = pack_geometry(depth, camera)
        if not self.train:
            data["original_valid"] = original_valid
        data["luma"] = rgb_np.mean(2, keepdims=True).transpose(2, 0, 1)
        data["rgb"] = ((rgb_np - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
        if self.paired:
            damaged = corrupt_depth(depth, rng, "dropout" if rng.random() < 0.1 else "holes")
            for key, value in pack_geometry(damaged, camera).items():
                data["damaged_" + key] = value
            data["hidden"] = ((depth > 0) & (damaged == 0)).astype(np.float32)[None]
        # Encode geometry before mirroring: camera-up normal angles retain their meaning.
        if mirror:
            data = {k: np.flip(v, axis=-1) for k, v in data.items()}
            label = np.fliplr(label)
        result = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in data.items()}
        result.update(label=torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64)),
                      id=name, coverage=raw_coverage, brightness=raw_brightness)
        if not self.train:
            # Fingerprint physical inputs, allowing future baselines to encode
            # the same observed depth differently without changing the test data.
            result["sensor_fingerprint"] = array_fingerprint(rgb_np, depth)
        return result
