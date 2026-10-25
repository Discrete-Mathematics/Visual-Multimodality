"""ImageNet Mix-Transformer B2 encoder.

The block layout matches the public MiT-B2 checkpoint so RGB features start
from a pretrained encoder. The segmentation model around it is MineGate.
"""

import math

import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def drop_path(x, drop_prob, training, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    if keep > 0.0 and scale_by_keep:
        mask = mask.div(keep)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).view(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)
        self.proj_drop = nn.Dropout(0.0)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        batch, length, channels = x.shape
        q = self.q(x).reshape(batch, length, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
            kv = self.kv(reduced)
        else:
            kv = self.kv(x)
        kv = kv.reshape(batch, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, drop_path_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class MixVisionTransformerB2(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dims = [64, 128, 320, 512]
        num_heads = [1, 2, 5, 8]
        depths = [3, 4, 6, 3]
        sr_ratios = [8, 4, 2, 1]
        mlp_ratios = [4, 4, 4, 4]
        drop_path_rate = 0.1
        dpr = [rate.item() for rate in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed1 = OverlapPatchEmbed(7, 4, 3, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        cursor = 0
        stages = []
        norms = []
        for index, depth in enumerate(depths):
            blocks = nn.ModuleList(
                [
                    Block(
                        embed_dims[index],
                        num_heads[index],
                        mlp_ratios[index],
                        sr_ratios[index],
                        dpr[cursor + block_index],
                    )
                    for block_index in range(depth)
                ]
            )
            stages.append(blocks)
            norms.append(nn.LayerNorm(embed_dims[index], eps=1e-6))
            cursor += depth
        self.block1, self.block2, self.block3, self.block4 = stages
        self.norm1, self.norm2, self.norm3, self.norm4 = norms

    def _stage(self, embed, blocks, norm, x):
        tokens, height, width = embed(x)
        for block in blocks:
            tokens = block(tokens, height, width)
        tokens = norm(tokens)
        batch = tokens.shape[0]
        return tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        feat1 = self._stage(self.patch_embed1, self.block1, self.norm1, x)
        feat2 = self._stage(self.patch_embed2, self.block2, self.norm2, feat1)
        feat3 = self._stage(self.patch_embed3, self.block3, self.norm3, feat2)
        feat4 = self._stage(self.patch_embed4, self.block4, self.norm4, feat3)
        return [feat1, feat2, feat3, feat4]

    def load_pretrained(self, path):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        cleaned = {}
        for key, value in checkpoint.items():
            name = key[9:] if key.startswith("backbone.") else key
            if name.startswith("head."):
                continue
            cleaned[name] = value
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "MiT-B2 checkpoint did not match. missing=%s unexpected=%s" % (missing, unexpected)
            )
