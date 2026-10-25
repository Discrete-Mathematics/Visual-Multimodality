"""Evaluate a saved checkpoint once on the official 1,576-image MUSeg test list."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import TEST_LIST, MUSegRGBD, build_index, read_ids
from engine import CLASS_NAMES, evaluate_loader
from models import MineSegmentor

WEIGHTS = {"b2": "/root/autodl-tmp/weights/mit-b2.bin", "b5": "/root/autodl-tmp/weights/mit-b5.bin"}
BASELINES = {
    "DeeplabV3+": {"pa": 76.67, "mpa": 54.91, "miou": 43.64},
    "SegFormer RGB": {"pa": 78.75, "mpa": 58.54, "miou": 47.93},
    "SegFormer Depth": {"pa": 81.16, "mpa": 62.21, "miou": 51.28},
    "SegFormer RGB-D": {"pa": 83.00, "mpa": 65.25, "miou": 55.20},
    "CMNeXt": {"pa": 84.05, "mpa": 66.77, "miou": 56.06},
    "SA-Gate": {"pa": 84.77, "mpa": 71.34, "miou": 59.38},
    "DFormer": {"pa": 85.63, "mpa": 70.37, "miou": 59.74},
    "CMX": {"pa": 86.30, "mpa": 72.72, "miou": 61.83},
}


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload["args"]
    num_classes = 16 if args.get("with_background") else 15
    model = MineSegmentor(args["variant"], WEIGHTS[args["variant"]], args["model"], num_classes=num_classes)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, args, payload


@torch.no_grad()
def tta_predict(model, batch, device, scales=(0.75, 1.0, 1.25)):
    rgb = batch["rgb"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    edge = batch["edge"].to(device, non_blocking=True)
    accumulated = None
    views = 0
    base = rgb.shape[-2:]
    for scale in scales:
        height = max(32, int(round(base[0] * scale / 32.0) * 32))
        width = max(32, int(round(base[1] * scale / 32.0) * 32))
        for flip in (False, True):
            def view(tensor, mode):
                if mode == "bilinear":
                    resized = F.interpolate(tensor, size=(height, width), mode=mode, align_corners=False)
                else:
                    resized = F.interpolate(tensor, size=(height, width), mode=mode)
                return torch.flip(resized, dims=[-1]) if flip else resized

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(view(rgb, "bilinear"), view(depth, "bilinear"), view(valid, "nearest"), view(edge, "bilinear"))
            probs = output["logits"].exp() if output["log_space"] else torch.softmax(output["logits"].float(), dim=1)
            if flip:
                probs = torch.flip(probs, dims=[-1])
            if probs.shape[-2:] != base:
                probs = F.interpolate(probs, size=base, mode="bilinear", align_corners=False)
            accumulated = probs if accumulated is None else accumulated + probs
            views += 1
    return accumulated / views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=str(TEST_LIST))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", default="official_test")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    model, train_args, payload = load_model(args.checkpoint, device)
    index = build_index()
    ids = read_ids(args.split)
    dataset = MUSegRGBD(
        ids,
        index,
        train_args["height"],
        train_args["width"],
        train=False,
        with_background=bool(train_args.get("with_background")),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.tta:
        from engine import NUM_CLASSES, confusion_update, scores_from_confusion

        matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
        mines = {}
        for batch in loader:
            probs = tta_predict(model, batch, device)
            target = batch["label"].to(device, non_blocking=True)
            confusion_update(matrix, probs, target)
            for index, mine_id in enumerate(batch["mine"]):
                mines.setdefault(mine_id, torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device))
                confusion_update(mines[mine_id], probs[index : index + 1], target[index : index + 1])
        summary = scores_from_confusion(matrix)
        summary["per_mine"] = {mine_id: scores_from_confusion(conf) for mine_id, conf in sorted(mines.items())}
    else:
        summary = evaluate_loader(
            model,
            loader,
            device,
            log_space=(train_args["model"] == "selective"),
            num_classes=16 if train_args.get("with_background") else 15,
        )
        if "paper" in summary:
            summary = {**summary, **{key: summary["paper"][key] for key in ("miou", "mpa", "pa", "iou", "recall", "support") if key in summary["paper"]}}
    per_class = []
    for name, iou, recall, support in zip(CLASS_NAMES, summary["iou"], summary["recall"], summary["support"]):
        per_class.append({"name": name, "iou": iou, "recall": recall, "support": support})
    result = {
        "tag": args.tag,
        "checkpoint": str(args.checkpoint),
        "epoch": payload.get("epoch"),
        "images": len(ids),
        "split": str(args.split),
        "height": train_args["height"],
        "width": train_args["width"],
        "model": train_args["model"],
        "variant": train_args["variant"],
        "pa_percent": summary["pa"] * 100,
        "mpa_percent": summary["mpa"] * 100,
        "miou_percent": summary["miou"] * 100,
        "per_class": per_class,
        "per_mine": {
            mine: {"pa": scores["pa"], "mpa": scores["mpa"], "miou": scores["miou"]}
            for mine, scores in summary["per_mine"].items()
        },
        "baselines_percent": BASELINES,
        "beats_cmx": {
            "pa": summary["pa"] * 100 > BASELINES["CMX"]["pa"],
            "mpa": summary["mpa"] * 100 > BASELINES["CMX"]["mpa"],
            "miou": summary["miou"] * 100 > BASELINES["CMX"]["miou"],
        },
    }
    out = Path(args.checkpoint).parent / f"{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("images", "pa_percent", "mpa_percent", "miou_percent", "beats_cmx")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
