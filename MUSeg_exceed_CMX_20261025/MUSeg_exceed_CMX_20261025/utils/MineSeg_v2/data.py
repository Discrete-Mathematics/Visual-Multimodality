"""MUSeg splits from FTransUNet train.txt, val.txt and test.txt.

Raw label 0 is unlabeled background and is ignored. Labels 1..15 are the
fifteen semantic classes. The test list is not used for checkpoint selection.
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = Path("/root/autodl-tmp/MUSeg")
SPLIT_ROOT = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
TRAIN_LIST = SPLIT_ROOT / "train.txt"
VAL_LIST = SPLIT_ROOT / "val.txt"
TEST_LIST = SPLIT_ROOT / "test.txt"
IGNORE = 255
NUM_CLASSES = 15
RARE = (0, 9, 10, 11, 12)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_ids(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def build_index(root=DATA_ROOT):
    index = {}
    for mine in range(1, 7):
        folder = Path(root) / f"{mine:02d}-Mine"
        for image_path in (folder / "Image").glob("*.jpg"):
            stem = image_path.stem
            index[stem] = {
                "rgb": image_path,
                "depth": folder / "Depth" / f"{stem}.png",
                "label": folder / "Label" / f"{stem}_label.png",
                "mine": f"{mine:02d}",
            }
    return index


def group_key(sample_id):
    parts = sample_id.split("-")
    return f"{parts[0]}-{parts[3]}"


def holdout_split(train_ids, fraction=0.1, seed=2026):
    """Hold out whole capture groups so checkpoint selection does not see near-duplicates."""
    grouped = defaultdict(list)
    for sample_id in train_ids:
        grouped[group_key(sample_id)].append(sample_id)
    keys = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(keys)
    hold_n = max(1, int(round(len(keys) * fraction)))
    hold_keys = set(keys[:hold_n])
    holdout, fit = [], []
    for key, members in grouped.items():
        (holdout if key in hold_keys else fit).extend(members)
    return sorted(fit), sorted(holdout)


def remap_label(label, with_background=False):
    out = np.full(label.shape, IGNORE, dtype=np.int64)
    if with_background:
        valid = (label >= 0) & (label <= 15)
        out[valid] = label[valid]
        return out
    valid = (label >= 1) & (label <= NUM_CLASSES)
    out[valid] = label[valid] - 1
    return out


def class_weights_from_counts(counts, clip=(0.4, 8.0)):
    freq = counts / np.maximum(counts.sum(), 1.0)
    weights = np.median(freq) / np.maximum(freq, 1e-8)
    weights = np.clip(weights, clip[0], clip[1])
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def scan_labels(ids, index, with_background=False):
    classes = 16 if with_background else NUM_CLASSES
    rare = (1, 10, 11, 12, 13) if with_background else RARE
    counts = np.zeros(classes, dtype=np.float64)
    rare_ids = []
    for sample_id in ids:
        label = remap_label(np.asarray(Image.open(index[sample_id]["label"]), dtype=np.uint8), with_background)
        for class_id in range(classes):
            counts[class_id] += np.count_nonzero(label == class_id)
        if any(np.count_nonzero(label == class_id) > 40 for class_id in rare):
            rare_ids.append(sample_id)
    return counts, rare_ids


def _edge(log_depth):
    padded = np.pad(log_depth, 1, mode="edge")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    magnitude = np.sqrt(gx * gx + gy * gy)
    return magnitude / (magnitude.max() + 1e-6)


class MUSegRGBD(Dataset):
    def __init__(self, ids, index, height, width, train=False, paste_bank=None, depth_dropout=0.0, with_background=False):
        missing = [sample_id for sample_id in ids if sample_id not in index]
        if missing:
            raise FileNotFoundError(missing[:3])
        self.ids = list(ids)
        self.index = index
        self.height = height
        self.width = width
        self.train = train
        self.paste_bank = paste_bank or []
        self.depth_dropout = depth_dropout
        self.with_background = with_background

    def __len__(self):
        return len(self.ids)

    def _load(self, sample_id):
        record = self.index[sample_id]
        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32)
        depth = np.asarray(Image.open(record["depth"]), dtype=np.float32)
        label = remap_label(np.asarray(Image.open(record["label"]), dtype=np.uint8), self.with_background)
        return rgb, depth, label, record["mine"]

    def _resize(self, rgb, depth, label, height, width):
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        depth_t = torch.from_numpy(depth).view(1, 1, *depth.shape)
        label_t = torch.from_numpy(label).view(1, 1, *label.shape).float()
        rgb_t = F.interpolate(rgb_t, size=(height, width), mode="bilinear", align_corners=False)
        depth_t = F.interpolate(depth_t, size=(height, width), mode="bilinear", align_corners=False)
        label_t = F.interpolate(label_t, size=(height, width), mode="nearest")
        return rgb_t[0], depth_t[0, 0], label_t[0, 0]

    def _geometry(self, rgb, depth, label):
        # Evaluation resizes the whole frame, matching the MUSeg benchmark input.
        # Training does the same on most samples so large equipment is not cropped away.
        if not self.train or random.random() < 0.65:
            rgb, depth, label = self._resize(rgb, depth, label, self.height, self.width)
            if self.train and random.random() < 0.5:
                rgb = torch.flip(rgb, dims=[2])
                depth = torch.flip(depth, dims=[1])
                label = torch.flip(label, dims=[1])
            return rgb, depth, label
        scale = random.uniform(0.75, 1.5)
        height, width = depth.shape
        resized_h = max(self.height, int(round(height * scale)))
        resized_w = max(self.width, int(round(width * scale)))
        rgb, depth, label = self._resize(rgb, depth, label, resized_h, resized_w)
        top = 0 if resized_h == self.height else random.randint(0, resized_h - self.height)
        left = 0 if resized_w == self.width else random.randint(0, resized_w - self.width)
        rgb = rgb[:, top : top + self.height, left : left + self.width]
        depth = depth[top : top + self.height, left : left + self.width]
        label = label[top : top + self.height, left : left + self.width]
        if random.random() < 0.5:
            rgb = torch.flip(rgb, dims=[2])
            depth = torch.flip(depth, dims=[1])
            label = torch.flip(label, dims=[1])
        return rgb, depth, label

    def _jitter(self, rgb):
        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.5)
            rgb = 255.0 * torch.pow((rgb / 255.0).clamp(0, 1), gamma)
        if random.random() < 0.8:
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            mean = rgb.mean(dim=(1, 2), keepdim=True)
            rgb = (rgb - mean) * contrast + mean
            rgb = rgb * brightness
        return rgb.clamp(0, 255)

    def _paste(self, rgb, depth, label):
        for _ in range(random.randint(1, 2)):
            item = random.choice(self.paste_bank)
            crop_rgb = torch.from_numpy(item["rgb"]).permute(2, 0, 1).float()
            crop_depth = torch.from_numpy(item["depth"]).unsqueeze(0)
            mask = torch.from_numpy(item["mask"]).float().view(1, 1, *item["mask"].shape)
            scale = random.uniform(0.8, 1.2)
            new_h = max(8, int(round(crop_rgb.shape[1] * scale)))
            new_w = max(8, int(round(crop_rgb.shape[2] * scale)))
            if new_h >= self.height or new_w >= self.width:
                continue
            crop_rgb = F.interpolate(crop_rgb.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            crop_depth = F.interpolate(crop_depth.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask, size=(new_h, new_w), mode="nearest")[0, 0] > 0.5
            top = random.randint(0, self.height - new_h)
            left = random.randint(0, self.width - new_w)
            rgb_region = rgb[:, top : top + new_h, left : left + new_w].clone()
            depth_region = depth[top : top + new_h, left : left + new_w].clone()
            label_region = label[top : top + new_h, left : left + new_w].clone()
            rgb_region[:, mask] = crop_rgb[:, mask]
            depth_region[mask] = crop_depth[0][mask]
            label_region[mask] = float(item["class_id"])
            rgb[:, top : top + new_h, left : left + new_w] = rgb_region
            depth[top : top + new_h, left : left + new_w] = depth_region
            label[top : top + new_h, left : left + new_w] = label_region
        return rgb, depth, label

    def __getitem__(self, index):
        sample_id = self.ids[index]
        rgb, depth, label, mine = self._load(sample_id)
        rgb, depth, label = self._geometry(rgb, depth, label)
        if self.train and self.paste_bank and random.random() < 0.75:
            rgb, depth, label = self._paste(rgb, depth, label)
        if self.train:
            rgb = self._jitter(rgb)
        valid = (depth > 0).float().unsqueeze(0)
        log_depth = torch.log1p(depth.clamp_min(0)) / float(np.log1p(10000.0))
        log_depth = log_depth.clamp(0, 1)
        edge = torch.from_numpy(_edge(log_depth.numpy())).float().unsqueeze(0)
        if self.train and random.random() < 0.2:
            box_h = random.randint(self.height // 12, self.height // 3)
            box_w = random.randint(self.width // 12, self.width // 3)
            top = random.randint(0, self.height - box_h)
            left = random.randint(0, self.width - box_w)
            log_depth[top : top + box_h, left : left + box_w] = 0
            valid[:, top : top + box_h, left : left + box_w] = 0
            edge[:, top : top + box_h, left : left + box_w] = 0
        if self.train and self.depth_dropout > 0 and random.random() < self.depth_dropout:
            log_depth = torch.zeros_like(log_depth)
            valid = torch.zeros_like(valid)
            edge = torch.zeros_like(edge)
        if self.train and random.random() < 0.3:
            log_depth = (log_depth + torch.randn_like(log_depth) * 0.02).clamp(0, 1)
        depth_image = torch.stack([log_depth, edge[0], valid[0]], dim=0)
        depth_image = (depth_image - 0.5) / 0.25
        rgb = rgb / 255.0
        mean = rgb.new_tensor(MEAN).view(3, 1, 1)
        std = rgb.new_tensor(STD).view(3, 1, 1)
        rgb = (rgb - mean) / std
        return {
            "rgb": rgb.contiguous(),
            "depth": depth_image.contiguous(),
            "valid": valid.contiguous(),
            "edge": edge.contiguous(),
            "label": label.long().contiguous(),
            "id": sample_id,
            "mine": mine,
        }


def build_paste_bank(ids, index, per_class=40, seed=0, with_background=False):
    rng = random.Random(seed)
    order = list(ids)
    rng.shuffle(order)
    rare = (1, 10, 11, 12, 13) if with_background else RARE
    buckets = {class_id: [] for class_id in rare}
    for sample_id in order:
        if all(len(items) >= per_class for items in buckets.values()):
            break
        label = remap_label(np.asarray(Image.open(index[sample_id]["label"]), dtype=np.uint8), with_background)
        needed = [class_id for class_id in rare if len(buckets[class_id]) < per_class and np.any(label == class_id)]
        if not needed:
            continue
        rgb = np.asarray(Image.open(index[sample_id]["rgb"]).convert("RGB"))
        depth = np.asarray(Image.open(index[sample_id]["depth"]), dtype=np.float32)
        height, width = label.shape
        for class_id in needed:
            ys, xs = np.where(label == class_id)
            if ys.size < 80:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            if (y1 - y0) * (x1 - x0) > 0.35 * height * width:
                continue
            mask = label[y0:y1, x0:x1] == class_id
            if mask.mean() < 0.08:
                continue
            buckets[class_id].append(
                {
                    "rgb": rgb[y0:y1, x0:x1].copy(),
                    "depth": depth[y0:y1, x0:x1].copy(),
                    "mask": mask.copy(),
                    "class_id": class_id,
                }
            )
    bank = [item for items in buckets.values() for item in items]
    counts = {class_id: len(items) for class_id, items in buckets.items()}
    return bank, counts
