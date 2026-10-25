"""Average two SelectiveMine checkpoints on the official MUSeg test list."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("OMP_NUM_THREADS", "1")
SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "tools"))

from mine_relay.data import MineDataset, read_ids
from selective_inference_bundle_20261023 import load_for_inference
from stage2_selective import TEST_SPLIT, accumulate, pack, resize_input

INIT = "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth"
SECOND = "/root/autodl-tmp/MineSeg/v2/runs/selective_hi/best.pth"


def load_one(path):
    model, _ = load_for_inference(INIT, device="cuda")
    if path != INIT:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.eval()
    return model


@torch.no_grad()
def probabilities(model, batch):
    keys = ("rgb", "depth", "valid", "luma", "hha")
    tensors = [batch[key].cuda(non_blocking=True) for key in keys]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(*tensors)["logits"].float().exp()


def main():
    height, width, batch = 768, 896, 4
    first, second = load_one(INIT), load_one(SECOND)
    dataset = MineDataset("/root/autodl-tmp/MUSeg/flat", TEST_SPLIT, (height, width), seed=2026)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    paper = np.zeros((15, 15), dtype=np.float64)
    missed = np.zeros(15, dtype=np.float64)
    matrix16 = np.zeros((16, 16), dtype=np.float64)
    for batch_data in loader:
        probability = 0.5 * (probabilities(first, batch_data) + probabilities(second, batch_data))
        accumulate(paper, missed, matrix16, probability.argmax(1).cpu().numpy(), batch_data["label"].numpy())
    metrics = pack(paper, missed, matrix16)
    metrics.update(images=len(read_ids(TEST_SPLIT)), height=height, width=width, ensemble=["released-480-model", "selective_hi-epoch4"])
    out = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_ensemble/official_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({key: metrics[key] for key in ("holdout_miou", "holdout_mpa", "holdout_pa", "miou_16", "mpa_16", "pa_16")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
