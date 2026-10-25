"""Load NVIDIA MiT classification weights into the installed SegFormer encoder."""

import re

import torch
from transformers import SegformerConfig, SegformerForImageClassification, SegformerModel

VARIANTS = {
    "b2": {"depths": [3, 4, 6, 3], "drop_path_rate": 0.1},
    "b5": {"depths": [3, 6, 40, 3], "drop_path_rate": 0.1},
}


def segformer_config(variant):
    spec = VARIANTS[variant]
    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=spec["depths"],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        drop_path_rate=spec["drop_path_rate"],
        reshape_last_stage=True,
        num_labels=1000,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def remap_legacy_hf(state):
    """Map transformers-4 SegFormer keys onto transformers-5 modules."""
    mapped = {}
    for key, value in state.items():
        if key.startswith("classifier."):
            mapped[key] = value
            continue
        match = re.match(r"segformer\.encoder\.patch_embeddings\.(\d+)\.(.+)", key)
        if match:
            mapped[f"segformer.stages.{match.group(1)}.patch_embeddings.{match.group(2)}"] = value
            continue
        match = re.match(r"segformer\.encoder\.layer_norm\.(\d+)\.(.+)", key)
        if match:
            mapped[f"segformer.stages.{match.group(1)}.layer_norm.{match.group(2)}"] = value
            continue
        match = re.match(r"segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)", key)
        if match:
            rest = match.group(3)
            rest = rest.replace("layer_norm_1.", "layernorm_before.")
            rest = rest.replace("layer_norm_2.", "layernorm_after.")
            rest = rest.replace("attention.self.query.", "attention.q_proj.")
            rest = rest.replace("attention.self.key.", "attention.k_proj.")
            rest = rest.replace("attention.self.value.", "attention.v_proj.")
            rest = rest.replace("attention.self.sr.", "attention.sequence_reduction.sequence_reduction.")
            rest = rest.replace("attention.self.layer_norm.", "attention.sequence_reduction.layer_norm.")
            rest = rest.replace("attention.output.dense.", "attention.o_proj.")
            rest = rest.replace("mlp.dense1.", "mlp.fc1.")
            rest = rest.replace("mlp.dense2.", "mlp.fc2.")
            mapped[f"segformer.stages.{match.group(1)}.blocks.{match.group(2)}.{rest}"] = value
            continue
        raise KeyError(key)
    return mapped


def load_encoder(variant, weight_path):
    classifier = SegformerForImageClassification(segformer_config(variant))
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    mapped = remap_legacy_hf(state)
    missing, unexpected = classifier.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"MiT load failed missing={missing[:8]} unexpected={unexpected[:8]}")
    encoder = classifier.segformer
    encoder.eval()
    return encoder


def encode(encoder, images):
    hidden = images
    features = []
    for stage in encoder.stages:
        hidden = stage(hidden)
        features.append(hidden)
    return features
