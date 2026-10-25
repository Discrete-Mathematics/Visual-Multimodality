"""MUSeg loader that turns aligned depth into HHA with the color intrinsics.

The released frames are the 1082×932 crop of the 2048×1536 color image,
box (462, 311, 1544, 1243). Depth is already on that color grid, so the
color intrinsics, shifted by the crop, are the ones that match the pixels.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


from .dataset import IGNORE_INDEX, RGB_MEAN, RGB_STD, encode_depth

NATIVE_W = 1082.0
NATIVE_H = 932.0
FX = 979.5962524414062
FY = 979.5027465820312
CX = 1022.30078125 - 462.0
CY = 773.655517578125 - 311.0
INV_NEAR = 1.0 / 0.617
INV_FAR = 1.0 / 5.658


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


def _color_jitter(image):
    array = np.asarray(image, dtype=np.float32)
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)
    array = array * brightness
    mean = array.mean(axis=(0, 1), keepdims=True)
    array = (array - mean) * contrast + mean
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _normalize_geometry(hha, valid):
    """Put the three HHA channels on the ImageNet scale the copied stem expects.

    Holes are set back to zero after the shift, so missing depth is not a
    large negative constant. The validity channel stays 0 or 1.
    """
    geo = np.array(hha, dtype=np.float32, copy=True)
    hole = valid[0] <= 0
    for channel in range(3):
        geo[channel] = (geo[channel] - np.float32(RGB_MEAN[channel])) / np.float32(RGB_STD[channel])
        geo[channel][hole] = 0.0
    return geo


def encode_hha(depth_mm, fx, fy, cx, cy):
    """Disparity, camera-up height, normal angle, validity. Holes stay zero."""
    depth_mm = np.ascontiguousarray(depth_mm)
    valid = depth_mm > 0
    z = np.zeros(depth_mm.shape, dtype=np.float32)
    z[valid] = depth_mm[valid].astype(np.float32) / 1000.0
    height, width = z.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    fx = np.float32(fx)
    fy = np.float32(fy)
    cx = np.float32(cx)
    cy = np.float32(cy)
    y = (v - cy) * z / fy

    inv = np.zeros_like(z)
    inv[valid] = 1.0 / np.maximum(z[valid], 1e-3)
    disparity = np.clip((inv - np.float32(INV_FAR)) / np.float32(INV_NEAR - INV_FAR), 0.0, 1.0)
    disparity[~valid] = 0.0
    height_n = np.clip((-y + 3.0) / 6.0, 0.0, 1.0)
    height_n[~valid] = 0.0

    dzdx = np.zeros_like(z)
    dzdy = np.zeros_like(z)
    dzdx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) * 0.5
    dzdy[1:-1, :] = (z[2:, :] - z[:-2, :]) * 0.5
    defined = np.zeros_like(valid)
    defined[:, 1:-1] = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    defined_y = np.zeros_like(valid)
    defined_y[1:-1, :] = valid[2:, :] & valid[1:-1, :] & valid[:-2, :]
    defined = defined & defined_y
    dzdx[~defined] = 0.0
    dzdy[~defined] = 0.0

    dxdu = (z + (u - cx) * dzdx) / fx
    dydu = ((v - cy) * dzdx) / fy
    dxdv = ((u - cx) * dzdy) / fx
    dydv = (z + (v - cy) * dzdy) / fy
    nx = dydu * dzdy - dzdx * dydv
    ny = dzdx * dxdv - dxdu * dzdy
    nz = dxdu * dydv - dydu * dxdv
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    angle = np.arccos(np.clip((-ny) / norm, -1.0, 1.0)) / np.pi
    angle = angle.astype(np.float32)
    angle[~defined] = 0.0

    valid_f = valid.astype(np.float32)
    packed = np.stack([disparity.astype(np.float32), height_n.astype(np.float32), angle, valid_f], axis=0)
    return packed, valid_f[None]


class MUSegHHA(torch.utils.data.Dataset):
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

    def _geometry(self, rgb, depth, label):
        width, height = rgb.size
        fx = FX * (width / NATIVE_W)
        fy = FY * (height / NATIVE_H)
        cx = CX * (width / NATIVE_W)
        cy = CY * (height / NATIVE_H)
        if self.train:
            scale = random.uniform(0.7, 1.4)
            rgb, depth, label = _resize_triplet(
                rgb, depth, label, (max(1, int(width * scale)), max(1, int(height * scale)))
            )
            fx *= scale
            fy *= scale
            cx *= scale
            cy *= scale
            rgb, depth, label = _pad_triplet(rgb, depth, label, self.crop_h, self.crop_w)
            cur_w, cur_h = rgb.size
            top = random.randint(0, cur_h - self.crop_h)
            left = random.randint(0, cur_w - self.crop_w)
            rgb = rgb.crop((left, top, left + self.crop_w, top + self.crop_h))
            depth = np.ascontiguousarray(depth[top : top + self.crop_h, left : left + self.crop_w])
            label = np.ascontiguousarray(label[top : top + self.crop_h, left : left + self.crop_w])
            cx -= left
            cy -= top
            if random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                depth = np.ascontiguousarray(np.fliplr(depth))
                label = np.ascontiguousarray(np.fliplr(label))
                cx = (self.crop_w - 1) - cx
            if self.paste_fn is not None:
                rgb, depth, label = self.paste_fn(rgb, depth, label)
                depth = np.ascontiguousarray(depth)
            if random.random() < 0.8:
                rgb = _color_jitter(rgb)
        else:
            rgb, depth, label = _resize_triplet(rgb, depth, label, (self.crop_w, self.crop_h))
            fx *= self.crop_w / width
            fy *= self.crop_h / height
            cx *= self.crop_w / width
            cy *= self.crop_h / height
        return rgb, depth, label, (fx, fy, cx, cy)

    def __getitem__(self, index):
        rgb, depth, label = self._load(self.ids[index])
        rgb, depth, label, camera = self._geometry(rgb, depth, label)
        hha, _hha_valid = encode_hha(depth, *camera)
        # Same RGB, depth pack and luma as MineGate, so the frozen checkpoint reproduces.
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        luma = rgb_np.mean(axis=2, keepdims=True).transpose(2, 0, 1)
        rgb_np = (rgb_np - RGB_MEAN) / RGB_STD
        packed, valid = encode_depth(depth)
        geo = _normalize_geometry(hha, valid)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb_np.transpose(2, 0, 1))),
            "depth": torch.from_numpy(np.ascontiguousarray(packed)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "luma": torch.from_numpy(np.ascontiguousarray(luma)),
            "hha": torch.from_numpy(np.ascontiguousarray(geo)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "id": self.ids[index],
        }
