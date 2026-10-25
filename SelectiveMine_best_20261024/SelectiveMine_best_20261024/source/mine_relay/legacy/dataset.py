"""MUSeg loader.

Depth stays 16-bit until after geometric augmentation. Invalid Kinect zeros
are a separate channel, not a distance of zero cast down to 8-bit.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import NUM_CLASSES

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Robust range measured on the training split, in millimetres.
LOG_DEPTH_LOW = float(np.log(617.0))
LOG_DEPTH_HIGH = float(np.log(5658.0))
IGNORE_INDEX = 255


def _read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _resize_triplet(rgb, depth, label, size):
    width, height = size
    rgb = rgb.resize((width, height), Image.BILINEAR)
    depth = np.array(Image.fromarray(depth).resize((width, height), Image.NEAREST), dtype=np.uint16)
    label = np.array(Image.fromarray(label).resize((width, height), Image.NEAREST), dtype=np.uint8)
    return rgb, depth, label


def _pad_triplet(rgb, depth, label, crop_h, crop_w):
    width, height = rgb.size
    pad_h = max(0, crop_h - height)
    pad_w = max(0, crop_w - width)
    if pad_h == 0 and pad_w == 0:
        return rgb, depth, label
    canvas = Image.new("RGB", (width + pad_w, height + pad_h))
    canvas.paste(rgb, (0, 0))
    depth_canvas = np.zeros((height + pad_h, width + pad_w), dtype=np.uint16)
    depth_canvas[:height, :width] = depth
    label_canvas = np.full((height + pad_h, width + pad_w), IGNORE_INDEX, dtype=np.uint8)
    label_canvas[:height, :width] = label
    return canvas, depth_canvas, label_canvas


def encode_depth(depth):
    depth = depth.astype(np.float32)
    valid = depth > 0
    log_depth = np.zeros_like(depth, dtype=np.float32)
    log_depth[valid] = np.log(depth[valid])
    span = LOG_DEPTH_HIGH - LOG_DEPTH_LOW
    normalized = np.clip((log_depth - LOG_DEPTH_LOW) / span, 0.0, 1.0)
    normalized[~valid] = 0.0
    gradient_y, gradient_x = np.gradient(normalized)
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    peak = float(magnitude.max())
    if peak > 0:
        magnitude = magnitude / peak
    magnitude[~valid] = 0.0
    packed = np.stack([normalized, valid.astype(np.float32), magnitude.astype(np.float32)], axis=0)
    return packed, valid.astype(np.float32)[None]


class MUSegDataset(Dataset):
    def __init__(self, root, split, crop_size=(480, 640), train=False, paste_fn=None):
        self.root = Path(root)
        self.train = train
        self.paste_fn = paste_fn
        self.crop_h, self.crop_w = crop_size
        split_name = "train.txt" if split == "train" else "val.txt"
        self.ids = _read_ids(self.root / split_name)

    def __len__(self):
        return len(self.ids)

    def _load(self, name):
        rgb = Image.open(self.root / "RGB" / (name + ".jpg")).convert("RGB")
        depth = np.array(Image.open(self.root / "Depth" / (name + ".png")), dtype=np.uint16)
        label = np.array(Image.open(self.root / "Label" / (name + "_label.png")), dtype=np.uint8)
        return rgb, depth, label

    def _augment(self, rgb, depth, label):
        scale = random.uniform(0.7, 1.4)
        width, height = rgb.size
        rgb, depth, label = _resize_triplet(rgb, depth, label, (int(width * scale), int(height * scale)))
        rgb, depth, label = _pad_triplet(rgb, depth, label, self.crop_h, self.crop_w)
        width, height = rgb.size
        top = random.randint(0, height - self.crop_h)
        left = random.randint(0, width - self.crop_w)
        rgb = rgb.crop((left, top, left + self.crop_w, top + self.crop_h))
        depth = depth[top : top + self.crop_h, left : left + self.crop_w]
        label = label[top : top + self.crop_h, left : left + self.crop_w]
        if random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth = np.ascontiguousarray(np.fliplr(depth))
            label = np.ascontiguousarray(np.fliplr(label))
        if random.random() < 0.8:
            rgb = _color_jitter(rgb)
        return rgb, depth, label

    def __getitem__(self, index):
        name = self.ids[index]
        rgb, depth, label = self._load(name)
        if self.train:
            rgb, depth, label = self._augment(rgb, depth, label)
            if self.paste_fn is not None:
                rgb, depth, label = self.paste_fn(rgb, depth, label)
        else:
            rgb, depth, label = _resize_triplet(rgb, depth, label, (self.crop_w, self.crop_h))

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        luma = rgb_np.mean(axis=2, keepdims=True).transpose(2, 0, 1)
        rgb_np = (rgb_np - RGB_MEAN) / RGB_STD
        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1).copy())
        packed, valid = encode_depth(depth)
        return {
            "rgb": rgb_tensor,
            "depth": torch.from_numpy(packed),
            "valid": torch.from_numpy(valid),
            "luma": torch.from_numpy(luma.copy()),
            "label": torch.from_numpy(label.astype(np.int64)),
            "id": name,
        }


def _color_jitter(image):
    array = np.asarray(image, dtype=np.float32)
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)
    array = array * brightness
    mean = array.mean(axis=(0, 1), keepdims=True)
    array = (array - mean) * contrast + mean
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def class_pixel_counts(root, split="train"):
    dataset = MUSegDataset(root, split, train=False)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    has_person = np.zeros(len(dataset), dtype=np.bool_)
    for index, name in enumerate(dataset.ids):
        label = np.array(Image.open(Path(root) / "Label" / (name + "_label.png")), dtype=np.uint8)
        flat = label.reshape(-1)
        counts += np.bincount(flat[flat < NUM_CLASSES], minlength=NUM_CLASSES)
        has_person[index] = bool((label == 1).any())
    return counts, has_person


def class_weights_from_counts(counts, max_weight=6.0):
    frequency = counts.astype(np.float64)
    frequency = frequency / frequency.sum()
    weights = 1.0 / np.sqrt(np.clip(frequency, 1e-8, None))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.25, max_weight)
    return torch.tensor(weights, dtype=torch.float32)
