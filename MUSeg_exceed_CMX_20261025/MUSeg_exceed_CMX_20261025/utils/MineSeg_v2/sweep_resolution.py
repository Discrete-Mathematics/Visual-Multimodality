"""Choose an inference resolution on the validation split only."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source/tools")

from finetune_selective import evaluate, prepare_model
from mine_relay.data import MineDataset

SIZES = ((704, 832), (736, 864), (768, 896), (800, 928), (832, 960))
VAL = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/val.txt")


def main():
    model = prepare_model("cuda")
    payload = torch.load(
        "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.eval()
    rows = []
    for height, width in SIZES:
        dataset = MineDataset("/root/autodl-tmp/MUSeg/flat", VAL, (height, width), train=False, seed=2026)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
        metrics = evaluate(model, loader, tta=False)
        row = {
            "height": height,
            "width": width,
            "miou16": metrics["miou_16"] * 100,
            "mpa16": metrics["mpa_16"] * 100,
            "pa16": metrics["pa_16"] * 100,
            "miou15": metrics["holdout_miou"] * 100,
            "mpa15": metrics["holdout_mpa"] * 100,
            "pa15": metrics["holdout_pa"] * 100,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path("/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/resolution_val.json").write_text(
        json.dumps(rows, indent=2)
    )


if __name__ == "__main__":
    main()
