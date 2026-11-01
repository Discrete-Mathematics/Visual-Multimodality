"""Val-selected probability ensemble of selective_pa3 checkpoints.

test.txt is read only for the subset chosen on val.txt. Epoch 12 alone is not
retested; its official single-scale file is already on disk.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source/tools")

from finetune_selective import prepare_model
from mine_relay.data import MineDataset
from mine_relay.engine import forward_batch
from sweep_pa3_prior import MIOU_FLOOR, MPA_FLOOR, summarize

RUN = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3")
ROOT = Path("/root/autodl-tmp/MUSeg/flat")
SPLIT = {
    "val": Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/val.txt"),
    "test": Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/test.txt"),
}
OUT = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3_ensemble/ensemble.json")
EPOCHS = (7, 9, 10, 11, 12)
GROUPS = {
    "e12": (12,),
    "e9+e12": (9, 12),
    "e7+e12": (7, 12),
    "e7+e9+e12": (7, 9, 12),
    "e10+e11+e12": (10, 11, 12),
}
CMX = {"pa": 86.30, "mpa": 72.72, "miou": 61.83}


def load_models(device):
    models = {}
    for epoch in EPOCHS:
        model = prepare_model(device)
        path = RUN / f"epoch_{epoch:02d}.pth"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.multimodal.configure_training(joint=True)
        model.multimodal.mix.fill_(1.0)
        model.eval()
        models[epoch] = model
        print(f"loaded epoch {epoch} from {path}", flush=True)
    return models


@torch.no_grad()
def run_split(models, split):
    dataset = MineDataset(ROOT, split, (768, 896), train=False, seed=2026)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    matrices = {name: np.zeros((16, 16), dtype=np.float64) for name in GROUPS}
    for batch in loader:
        probabilities = {}
        for epoch, model in models.items():
            probabilities[epoch] = forward_batch(model, batch)["logits"].float().exp()
        target = batch["label"].to(probabilities[12].device, non_blocking=True)
        valid = target != 255
        for name, members in GROUPS.items():
            prediction = torch.stack([probabilities[epoch] for epoch in members]).mean(0).argmax(1)
            gt = target[valid]
            pr = prediction[valid]
            ok = (gt >= 0) & (gt < 16) & (pr >= 0) & (pr < 16)
            if ok.any():
                index = (gt[ok] * 16 + pr[ok]).detach().cpu().numpy()
                matrices[name] += np.bincount(index, minlength=256).reshape(16, 16)
    return {name: summarize(matrix) for name, matrix in matrices.items()}


def beats(row):
    return row["pa16"] > CMX["pa"] and row["mpa16"] > CMX["mpa"] and row["miou16"] > CMX["miou"]


def main():
    models = load_models("cuda")
    print("val", flush=True)
    val = run_split(models, SPLIT["val"])
    for name, row in val.items():
        print(f"  {name} PA {row['pa16']:.2f} MPA {row['mpa16']:.2f} mIoU {row['miou16']:.2f}", flush=True)
    ranked = []
    for name, row in val.items():
        feasible = row["miou16"] >= MIOU_FLOOR and row["mpa16"] >= MPA_FLOOR
        ranked.append((feasible, row["pa16"], row["miou16"], -len(GROUPS[name]), name))
    ranked.sort(reverse=True)
    chosen = ranked[0][-1]
    print(f"chosen {chosen}", flush=True)
    tested = None
    if chosen != "e12":
        keep = set(GROUPS[chosen])
        models = {epoch: model for epoch, model in models.items() if epoch in keep}
        print("test", flush=True)
        tested = run_split(models, SPLIT["test"])[chosen]
        print(
            f"  test PA {tested['pa16']:.2f} MPA {tested['mpa16']:.2f} mIoU {tested['miou16']:.2f}",
            flush=True,
        )
    result = {
        "checkpoints": {str(epoch): str(RUN / f"epoch_{epoch:02d}.pth") for epoch in EPOCHS},
        "resolution": [768, 896],
        "selection_rule": "max validation 16-class PA among ensembles with mIoU>=61.20 and MPA>=71.80; fewer members win ties",
        "val": {name: {k: v for k, v in row.items() if not k.startswith("per_class")} for name, row in val.items()},
        "chosen": chosen,
        "members": list(GROUPS[chosen]),
        "test": None if tested is None else tested,
        "beats_cmx_16": None if tested is None else beats(tested),
        "cmx": CMX,
        "note": "e12 was not retested; use selective_pa3/official_test.json for that single checkpoint",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"chosen": chosen, "beats_cmx_16": result["beats_cmx_16"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
