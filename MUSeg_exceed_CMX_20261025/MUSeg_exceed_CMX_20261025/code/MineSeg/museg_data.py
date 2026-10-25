"""MUSeg RGB-D dataset driven by the FTransUNet split lists."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = Path("/root/autodl-tmp/MUSeg")
SPLIT_DIR = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")

CLASS_NAMES = [
    "person",
    "cable",
    "tube",
    "indicator",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "rail area",
    "support equipment",
    "door",
    "tools and materials",
    "rescue equipment",
    "container",
    "metal fixture",
    "anchoring equipment",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

# Full-resolution foreground pixel counts on the training split (labels 1..15).
TRAIN_FG_COUNTS = np.array(
    [
        3493244,
        89515308,
        60298355,
        32005220,
        29382333,
        19568777,
        112456762,
        33088306,
        70204017,
        13423548,
        20353974,
        7407301,
        7394087,
        76925502,
        185589579,
    ],
    dtype=np.float64,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Foreground ids after subtracting 1. These are the long-tail classes on the train split.
RARE_CLASS_IDS = (0, 9, 10, 11, 12)


def class_weights(clip=(0.25, 6.0)):
    freq = TRAIN_FG_COUNTS / TRAIN_FG_COUNTS.sum()
    med = np.median(freq)
    weights = med / np.maximum(freq, 1e-8)
    weights = np.clip(weights, clip[0], clip[1])
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def read_ids(split):
    path = SPLIT_DIR / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_index(root=DATA_ROOT):
    index = {}
    for mine in range(1, 7):
        folder = Path(root) / f"{mine:02d}-Mine"
        for image_path in (folder / "Image").glob("*.jpg"):
            stem = image_path.stem
            index[stem] = {
                "rgb": str(image_path),
                "depth": str(folder / "Depth" / f"{stem}.png"),
                "label": str(folder / "Label" / f"{stem}_label.png"),
                "mine": mine,
            }
    return index


def remap_label(label):
    """Raw ids are 0=background and 1..15=classes. Train on 0..14 and ignore 255."""
    out = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
    valid = (label >= 1) & (label <= NUM_CLASSES)
    out[valid] = label[valid] - 1
    return out


def _pad(tensor, size, value):
    _, height, width = tensor.shape
    pad_h = max(0, size - height)
    pad_w = max(0, size - width)
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=value)
    return tensor


def build_paste_bank(ids, index, per_class=80, seed=0):
    """Crops of long-tail classes, copied together with the aligned depth."""
    rng = random.Random(seed)
    buckets = {class_id: [] for class_id in RARE_CLASS_IDS}
    order = list(ids)
    rng.shuffle(order)
    for sample_id in order:
        if all(len(items) >= per_class for items in buckets.values()):
            break
        label = remap_label(np.asarray(Image.open(index[sample_id]["label"]).convert("L"), dtype=np.uint8))
        needed = [class_id for class_id in RARE_CLASS_IDS if len(buckets[class_id]) < per_class and np.any(label == class_id)]
        if not needed:
            continue
        rgb = np.asarray(Image.open(index[sample_id]["rgb"]).convert("RGB"))
        depth = np.asarray(Image.open(index[sample_id]["depth"]))
        height, width = label.shape
        for class_id in needed:
            ys, xs = np.where(label == class_id)
            if ys.size < 80:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            box_area = (y1 - y0) * (x1 - x0)
            if box_area < 64 or box_area > 0.35 * height * width:
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


class MUSegDataset(Dataset):
    def __init__(self, ids, index, size=640, train=False, use_depth_aug=False, paste_bank=None):
        missing = [sample_id for sample_id in ids if sample_id not in index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} ids missing, e.g. {missing[:3]}")
        self.ids = list(ids)
        self.index = index
        self.size = size
        self.train = train
        self.use_depth_aug = use_depth_aug
        self.paste_bank = paste_bank or []

    def __len__(self):
        return len(self.ids)

    def _load(self, sample_id):
        record = self.index[sample_id]
        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32)
        depth = np.asarray(Image.open(record["depth"]), dtype=np.float32)
        label = remap_label(np.asarray(Image.open(record["label"]).convert("L"), dtype=np.uint8))
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        label_t = torch.from_numpy(label).unsqueeze(0).float()
        return rgb_t, depth_t, label_t, record["mine"]

    def _geometry(self, rgb, depth, label):
        _, height, width = depth.shape
        if self.train:
            scale = random.uniform(0.75, 1.25)
            new_h = max(self.size, int(round(height * scale)))
            new_w = max(self.size, int(round(width * scale)))
        else:
            new_h = new_w = self.size
        rgb = F.interpolate(rgb.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        label = F.interpolate(label.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
        rgb = _pad(rgb, self.size, 0.0)
        depth = _pad(depth, self.size, 0.0)
        label = _pad(label, self.size, float(IGNORE_INDEX))
        if self.train:
            _, height, width = depth.shape
            top = 0 if height == self.size else random.randint(0, height - self.size)
            left = 0 if width == self.size else random.randint(0, width - self.size)
            rgb = rgb[:, top : top + self.size, left : left + self.size]
            depth = depth[:, top : top + self.size, left : left + self.size]
            label = label[:, top : top + self.size, left : left + self.size]
            if random.random() < 0.5:
                rgb = torch.flip(rgb, dims=[2])
                depth = torch.flip(depth, dims=[2])
                label = torch.flip(label, dims=[2])
        else:
            rgb = rgb[:, : self.size, : self.size]
            depth = depth[:, : self.size, : self.size]
            label = label[:, : self.size, : self.size]
        return rgb, depth, label

    def _jitter(self, rgb):
        if random.random() < 0.8:
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            mean = rgb.mean(dim=(1, 2), keepdim=True)
            rgb = (rgb - mean) * contrast + mean
            rgb = rgb * brightness
        if random.random() < 0.2:
            gray = rgb.mean(dim=0, keepdim=True)
            rgb = gray.expand_as(rgb) * 0.85 + rgb * 0.15
        return rgb.clamp(0, 255)

    def _paste_rare(self, rgb, depth, label):
        for _ in range(random.randint(1, 2)):
            item = random.choice(self.paste_bank)
            crop_rgb = torch.from_numpy(item["rgb"]).permute(2, 0, 1).float()
            crop_depth = torch.from_numpy(np.asarray(item["depth"], dtype=np.float32)).unsqueeze(0)
            mask = torch.from_numpy(np.asarray(item["mask"]))
            scale = random.uniform(0.75, 1.25)
            new_h = max(8, int(round(crop_rgb.shape[1] * scale)))
            new_w = max(8, int(round(crop_rgb.shape[2] * scale)))
            crop_rgb = F.interpolate(crop_rgb.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
            crop_depth = F.interpolate(crop_depth.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.float().view(1, 1, *mask.shape), size=(new_h, new_w), mode="nearest").view(new_h, new_w) > 0.5
            _, height, width = depth.shape
            if new_h >= height or new_w >= width:
                continue
            top = random.randint(0, height - new_h)
            left = random.randint(0, width - new_w)
            region = mask
            rgb[:, top : top + new_h, left : left + new_w][:, region] = crop_rgb[:, region]
            depth[:, top : top + new_h, left : left + new_w][:, region] = crop_depth[:, region]
            label[0, top : top + new_h, left : left + new_w][region] = float(item["class_id"])
        return rgb, depth, label

    def __getitem__(self, index):
        sample_id = self.ids[index]
        rgb, depth, label, mine = self._load(sample_id)
        if self.train and self.paste_bank and random.random() < 0.8:
            rgb, depth, label = self._paste_rare(rgb, depth, label)
        rgb, depth, label = self._geometry(rgb, depth, label)
        if self.train:
            rgb = self._jitter(rgb)
        rgb = rgb / 255.0
        mean = rgb.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = rgb.new_tensor(IMAGENET_STD).view(3, 1, 1)
        rgb = (rgb - mean) / std

        depth = torch.log1p(depth.clamp_min(0)) / float(np.log1p(10000.0))
        depth = depth.clamp(0, 1)
        if self.train and self.use_depth_aug and random.random() < 0.4:
            _, height, width = depth.shape
            box_h = random.randint(max(8, height // 10), max(9, height // 3))
            box_w = random.randint(max(8, width // 10), max(9, width // 3))
            top = random.randint(0, height - box_h)
            left = random.randint(0, width - box_w)
            depth[:, top : top + box_h, left : left + box_w] = 0
        if self.train and self.use_depth_aug and random.random() < 0.3:
            depth = (depth + torch.randn_like(depth) * 0.02).clamp(0, 1)

        label = label.squeeze(0).long()
        return {
            "rgb": rgb.contiguous(),
            "depth": depth.contiguous(),
            "label": label.contiguous(),
            "mine": mine,
            "id": sample_id,
        }
