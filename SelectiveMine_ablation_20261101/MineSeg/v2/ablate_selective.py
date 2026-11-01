"""Inference ablations of the frozen selective_pa3 checkpoint.

Weights are not retrained. Each variant removes one trained path at evaluation
time and, where the fused expert changes, recomputes the router on that pair.
The checkpoint was fit on FTransUNet train.txt and selected on val.txt.
Val and test are both scored. The published single-scale protocol is 768x896,
no test-time augmentation, label 255 ignored, label 0 kept in the 16-class scores.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")

from finetune_selective import PAPER_CLASSES, prepare_model, seed_everything
from mine_relay.data import MineDataset, read_ids

CHECKPOINT = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3/best.pth")
OFFICIAL_TEST = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3/official_test.json")
OUTDIR = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3_ablation")
SPLIT_DIR = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
SPLITS = {
    "train": SPLIT_DIR / "train.txt",
    "val": SPLIT_DIR / "val.txt",
    "test": SPLIT_DIR / "test.txt",
}
ROOT = Path("/root/autodl-tmp/MUSeg/flat")
HEIGHT, WIDTH = 768, 896
BATCH_SIZE = 4

VARIANTS = [
    ("full", "完整模型", "不移除。路由器混合外观对数概率和融合对数概率。"),
    ("equal_mix", "常数 0.5 混合", "路由器换成 0.5，两路专家仍是完整的外观头和融合头。"),
    ("fused", "只用融合专家", "去掉路由器和外观头，对融合 logits 做 argmax。"),
    ("rgb", "只用外观解码器", "去掉深度、几何、融合解码器和路由器。"),
    ("geo", "只用几何解码器", "只保留 HHA 几何头。"),
    ("no_geo_head", "去掉几何 logits", "融合解码器仍吃 relay 特征，不再加上几何头；路由器按新的融合专家重算。"),
    ("no_geo_head_fused", "去掉几何 logits 和路由器", "对去掉几何头之后的融合解码器直接 argmax。"),
    ("no_relay", "去掉 relay", "RGB-D 门控特征直接进融合解码器，几何头仍相加；路由器重算。"),
    ("no_relay_fused", "去掉 relay 和路由器", "对无 relay 的融合专家直接 argmax。"),
    ("no_geometry", "去掉几何支路", "去掉 relay 和几何头，路由器仍把该解码器与外观头混合。"),
    ("no_geometry_fused", "只留 RGB-D 门控解码器", "去掉 relay、几何头和路由器。"),
    ("no_depth_gate", "去掉深度门控", "可靠度门控的有效掩码置零，深度残差不注入；HHA、relay、几何头和路由器保留。"),
    ("no_hha", "去掉 HHA", "HHA 输入置零。深度门控、relay、几何头和路由器仍运行。"),
    ("no_depth", "去掉深度和 HHA", "深度门控关闭且 HHA 置零，路由器仍混合外观头。"),
    ("rgb_fusion_decoder", "融合解码器只吃 RGB", "融合解码器接收未经深度注入、未经 relay 的 RGB 特征，不经过路由器。"),
]

FOREGROUND = [
    "人员", "线缆", "管路", "指示装置", "电气设备", "电子设备", "采掘设备",
    "轨道区域", "支护设备", "门", "工具材料", "救援设备", "容器", "金属构件", "锚固设备",
]


def _scores(paper, missed, matrix16):
    true_positive = np.diag(paper)
    false_positive = paper.sum(0) - true_positive
    false_negative = paper.sum(1) - true_positive + missed
    support = paper.sum(1) + missed
    union = true_positive + false_positive + false_negative
    present = support > 0
    iou = true_positive / np.maximum(union, 1.0)
    recall = true_positive / np.maximum(support, 1.0)
    tp16 = np.diag(matrix16)
    support16 = matrix16.sum(1)
    pred16 = matrix16.sum(0)
    union16 = support16 + pred16 - tp16
    present16 = support16 > 0
    iou16 = tp16 / np.maximum(union16, 1.0)
    recall16 = tp16 / np.maximum(support16, 1.0)
    return {
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "mpa": float(recall[present].mean()) if present.any() else 0.0,
        "pa": float(true_positive.sum() / max(support.sum(), 1.0)),
        "per_class_iou": iou.tolist(),
        "per_class_support": support.astype(np.int64).tolist(),
        "miou_16": float(iou16[present16].mean()) if present16.any() else 0.0,
        "mpa_16": float(recall16[present16].mean()) if present16.any() else 0.0,
        "pa_16": float(tp16.sum() / max(matrix16.sum(), 1.0)),
        "per_class_iou_16": iou16.tolist(),
        "per_class_support_16": support16.astype(np.int64).tolist(),
    }


def _accumulate(paper, missed, matrix16, prediction, target):
    for index in range(prediction.shape[0]):
        pred = prediction[index]
        gt = target[index]
        valid = (gt >= 1) & (gt <= PAPER_CLASSES)
        pred_v, gt_v = pred[valid], gt[valid] - 1
        mapped = pred_v - 1
        recognized = (mapped >= 0) & (mapped < PAPER_CLASSES)
        if recognized.any():
            bins = np.bincount(
                gt_v[recognized] * PAPER_CLASSES + mapped[recognized],
                minlength=PAPER_CLASSES ** 2,
            )
            paper += bins.reshape(PAPER_CLASSES, PAPER_CLASSES)
        if (~recognized).any():
            missed += np.bincount(gt_v[~recognized], minlength=PAPER_CLASSES)
        labeled = gt != 255
        pred_l, gt_l = pred[labeled], gt[labeled]
        ok = (gt_l >= 0) & (gt_l < 16) & (pred_l >= 0) & (pred_l < 16)
        if ok.any():
            matrix16 += np.bincount(gt_l[ok] * 16 + pred_l[ok], minlength=256).reshape(16, 16)


def _decode(decoder, features, size):
    return F.interpolate(decoder(features), size, mode="bilinear", align_corners=False)


def _mix_features(relay, features, geometry, valid, luma, depth, use_relay):
    mixed = []
    for index, feature in enumerate(features):
        if use_relay:
            geo = geometry[index]
            if geo.shape[-2:] != feature.shape[-2:]:
                geo = F.interpolate(geo, feature.shape[-2:], mode="bilinear", align_corners=False)
            delta, _state = relay[index](feature, geo, valid, luma, depth[:, :1])
        else:
            delta = 0.0
        mixed.append(feature + delta)
    return mixed


def _route(model, rgb_logits, fused_logits, rgb_features, valid, luma, present, gate):
    log_rgb = F.log_softmax(rgb_logits.float(), dim=1)
    log_fused = F.log_softmax(fused_logits.float(), dim=1)
    if gate == "rgb":
        return log_rgb, None
    if gate == "fused":
        return torch.where(present, log_fused, log_rgb), None
    if gate == "half":
        half = log_rgb.new_tensor(0.5).log()
        mixture = torch.logaddexp(log_rgb + half, log_fused + half)
        return torch.where(present, mixture, log_rgb), None
    rgb_p = log_rgb.detach().exp()
    fused_p = log_fused.detach().exp()
    entropy_rgb = -(rgb_p * log_rgb.detach()).sum(1, keepdim=True)
    entropy_fused = -(fused_p * log_fused.detach()).sum(1, keepdim=True)
    grid = rgb_features[0].shape[-2:]
    evidence = torch.cat(
        [rgb_p, fused_p, (rgb_p - fused_p).abs(), entropy_rgb, entropy_fused, valid.float(), luma.float()],
        dim=1,
    )
    router_logit = _decode_map(model.router(F.adaptive_avg_pool2d(evidence, grid)), rgb_logits.shape[-2:])
    mixture = torch.logaddexp(
        log_rgb + F.logsigmoid(-router_logit.float()),
        log_fused + F.logsigmoid(router_logit.float()),
    )
    return torch.where(present, mixture, log_rgb), router_logit


def _decode_map(tensor, size):
    return F.interpolate(tensor, size, mode="bilinear", align_corners=False)


def variant_predictions(model, rgb, depth, valid, luma, hha):
    net = model.multimodal
    size = rgb.shape[-2:]
    rgb_features = net.base.rgb_encoder(rgb)
    depth_features = net.base.depth_encoder(depth)
    gated = [gate(r, d, valid, luma)[0] for gate, r, d in zip(net.base.gates, rgb_features, depth_features)]
    valid0 = torch.zeros_like(valid)
    gated_rgb = [gate(r, d, valid0, luma)[0] for gate, r, d in zip(net.base.gates, rgb_features, depth_features)]
    geometry = net.geo_encoder(net.stem(hha))
    geometry0 = net.geo_encoder(net.stem(torch.zeros_like(hha)))
    present = (valid.flatten(1).sum(1) > 0).view(-1, 1, 1, 1)
    scale = torch.tanh(net.open_raw) * net.mix * present.to(rgb.dtype)

    mixed = _mix_features(net.relay, gated, geometry, valid, luma, depth, True)
    mixed_norel = _mix_features(net.relay, gated, geometry, valid, luma, depth, False)
    mixed_nodepth = _mix_features(net.relay, gated_rgb, geometry, valid, luma, depth, True)
    mixed_nohha = _mix_features(net.relay, gated, geometry0, valid, luma, depth, True)
    mixed_nodepth_nohha = _mix_features(net.relay, gated_rgb, geometry0, valid, luma, depth, True)

    base = _decode(net.base.decoder, mixed, size)
    base_norel = _decode(net.base.decoder, mixed_norel, size)
    base_nodepth = _decode(net.base.decoder, mixed_nodepth, size)
    base_nohha = _decode(net.base.decoder, mixed_nohha, size)
    base_neither = _decode(net.base.decoder, mixed_nodepth_nohha, size)
    rgb_only_decoder = _decode(net.base.decoder, gated_rgb, size)
    geo = _decode(net.geo_decoder, geometry, size)
    geo0 = _decode(net.geo_decoder, geometry0, size)
    rgb_logits = _decode(model.appearance_decoder, rgb_features, size)

    fused = base + scale * geo
    fused_nogeo = base
    fused_norel = base_norel + scale * geo
    fused_nogeom = base_norel
    fused_nodepth = base_nodepth + scale * geo
    fused_nohha = base_nohha + scale * geo0
    fused_neither = base_neither + scale * geo0

    routed = {}
    router_logit = None
    specs = {
        "full": (rgb_logits, fused, "learned"),
        "equal_mix": (rgb_logits, fused, "half"),
        "fused": (rgb_logits, fused, "fused"),
        "rgb": (rgb_logits, fused, "rgb"),
        "no_geo_head": (rgb_logits, fused_nogeo, "learned"),
        "no_geo_head_fused": (rgb_logits, fused_nogeo, "fused"),
        "no_relay": (rgb_logits, fused_norel, "learned"),
        "no_relay_fused": (rgb_logits, fused_norel, "fused"),
        "no_geometry": (rgb_logits, fused_nogeom, "learned"),
        "no_geometry_fused": (rgb_logits, fused_nogeom, "fused"),
        "no_depth_gate": (rgb_logits, fused_nodepth, "learned"),
        "no_hha": (rgb_logits, fused_nohha, "learned"),
        "no_depth": (rgb_logits, fused_neither, "learned"),
    }
    predictions = {}
    for name, (left, right, gate) in specs.items():
        log_prob, logit = _route(model, left, right, rgb_features, valid, luma, present, gate)
        predictions[name] = log_prob.argmax(1)
        if name == "full":
            router_logit = logit
            routed["full_log"] = log_prob
    predictions["geo"] = geo.argmax(1)
    predictions["rgb_fusion_decoder"] = rgb_only_decoder.argmax(1)
    gate_mean = router_logit.float().sigmoid().mean()
    return predictions, gate_mean, present.flatten().sum()


def _split_audit():
    lists = {name: read_ids(path) for name, path in SPLITS.items()}
    sets = {name: set(values) for name, values in lists.items()}
    overlap = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    if any(overlap.values()):
        raise SystemExit(f"FTransUNet splits overlap: {overlap}")
    return {
        "files": {name: str(path) for name, path in SPLITS.items()},
        "counts": {name: len(values) for name, values in lists.items()},
        "overlap": overlap,
    }


def _evaluate_split(model, split_name, split_path):
    dataset = MineDataset(
        ROOT, split_path, (HEIGHT, WIDTH), train=False, paired=False, rare_crop=0.0, corruption="none", seed=2026,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    names = [item[0] for item in VARIANTS]
    paper = {name: np.zeros((PAPER_CLASSES, PAPER_CLASSES), dtype=np.float64) for name in names}
    missed = {name: np.zeros(PAPER_CLASSES, dtype=np.float64) for name in names}
    matrix16 = {name: np.zeros((16, 16), dtype=np.float64) for name in names}
    gate_sum = 0.0
    gate_count = 0
    images_with_depth = 0
    disagree_fused = 0
    disagree_rgb = 0
    disagree_relay = 0
    labeled_pixels = 0
    checked = False
    started = time.time()
    seen = 0
    print(f"evaluating {split_name} {split_path}", flush=True)
    for step, batch in enumerate(loader):
        rgb = batch["rgb"].cuda(non_blocking=True)
        depth = batch["depth"].cuda(non_blocking=True)
        valid = batch["valid"].cuda(non_blocking=True)
        luma = batch["luma"].cuda(non_blocking=True)
        hha = batch["hha"].cuda(non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions, gate_mean, present_count = variant_predictions(model, rgb, depth, valid, luma, hha)
            if not checked:
                reference = model(rgb, depth, valid, luma, hha)["logits"].argmax(1)
                if not torch.equal(reference, predictions["full"]):
                    mismatch = int((reference != predictions["full"]).sum())
                    raise SystemExit(f"full variant diverges from SelectiveMine.forward on {mismatch} pixels")
                checked = True
                print(f"{split_name}: full variant matches SelectiveMine.forward", flush=True)
        target = batch["label"].numpy()
        arrays = {name: pred.detach().cpu().numpy() for name, pred in predictions.items()}
        labeled = target != 255
        labeled_pixels += int(labeled.sum())
        disagree_fused += int((arrays["full"][labeled] != arrays["fused"][labeled]).sum())
        disagree_rgb += int((arrays["full"][labeled] != arrays["rgb"][labeled]).sum())
        disagree_relay += int((arrays["full"][labeled] != arrays["no_relay"][labeled]).sum())
        images_with_depth += int(present_count.item())
        gate_sum += float(gate_mean.item()) * rgb.shape[0]
        gate_count += rgb.shape[0]
        for name in names:
            _accumulate(paper[name], missed[name], matrix16[name], arrays[name], target)
        seen += rgb.shape[0]
        if step % 20 == 0:
            print(f"{split_name} step {step} images {seen}", flush=True)
    elapsed = time.time() - started
    full = _scores(paper["full"], missed["full"], matrix16["full"])
    rows = []
    for name, title, removed in VARIANTS:
        scores = _scores(paper[name], missed[name], matrix16[name])
        scores.update({
            "id": name,
            "title": title,
            "removed": removed,
            "delta_pa_16": scores["pa_16"] - full["pa_16"],
            "delta_mpa_16": scores["mpa_16"] - full["mpa_16"],
            "delta_miou_16": scores["miou_16"] - full["miou_16"],
            "delta_miou": scores["miou"] - full["miou"],
        })
        rows.append(scores)
        print(
            f"{split_name} {name:22s} PA {scores['pa_16']*100:.2f} MPA {scores['mpa_16']*100:.2f} "
            f"mIoU {scores['miou_16']*100:.2f} dPA {scores['delta_pa_16']*100:+.2f}",
            flush=True,
        )
    return {
        "split_name": split_name,
        "split": str(split_path),
        "images": len(read_ids(split_path)),
        "seconds": round(elapsed, 1),
        "mean_router_gate": gate_sum / max(gate_count, 1),
        "images_with_any_depth": images_with_depth,
        "labeled_pixels": labeled_pixels,
        "fraction_disagree_fused": disagree_fused / max(labeled_pixels, 1),
        "fraction_disagree_rgb": disagree_rgb / max(labeled_pixels, 1),
        "fraction_disagree_relay": disagree_relay / max(labeled_pixels, 1),
        "variants": rows,
    }


def _check_official(test_report):
    official = json.loads(OFFICIAL_TEST.read_text())
    full = next(row for row in test_report["variants"] if row["id"] == "full")
    compared = {
        "pa_16": (full["pa_16"], official["pa_16"]),
        "mpa_16": (full["mpa_16"], official["mpa_16"]),
        "miou_16": (full["miou_16"], official["miou_16"]),
    }
    gaps = {key: left - right for key, (left, right) in compared.items()}
    test_report["official_test"] = {key: right for key, (_, right) in compared.items()}
    test_report["official_gap"] = gaps
    worst = max(abs(value) for value in gaps.values())
    print(f"official gap {gaps} worst {worst:.8f}", flush=True)
    if worst > 1e-4:
        raise SystemExit(f"full test variant is too far from official_test.json: {gaps}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seed_everything(2026)
    audit = _split_audit()
    print("split audit", json.dumps(audit, ensure_ascii=False), flush=True)
    device = "cuda"
    print("loading", CHECKPOINT, flush=True)
    model = prepare_model(device)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    model.eval()
    epoch = int(payload.get("epoch", -1))
    open_scale = float(torch.tanh(model.multimodal.open_raw).detach() * model.multimodal.mix)
    print(f"epoch {epoch} open_scale {open_scale:.4f}", flush=True)

    evaluated = {}
    for split_name in ("val", "test"):
        evaluated[split_name] = _evaluate_split(model, split_name, SPLITS[split_name])
    _check_official(evaluated["test"])
    report = {
        "checkpoint": str(CHECKPOINT),
        "epoch": epoch,
        "height": HEIGHT,
        "width": WIDTH,
        "batch_size": BATCH_SIZE,
        "protocol": (
            "frozen-weight inference ablation of selective_pa3. "
            "Train.txt fits the checkpoint, val.txt is the selection split, test.txt is the report split. "
            "Weights are not retrained. Single scale, no test-time augmentation. "
            "16-class scores count label 0 and ignore 255."
        ),
        "open_scale": open_scale,
        "foreground_names": FOREGROUND,
        "splits": audit,
        "val": evaluated["val"],
        "test": evaluated["test"],
    }
    destination = OUTDIR / "ablation.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (OUTDIR / "ablation_val.json").write_text(json.dumps(evaluated["val"], indent=2, ensure_ascii=False))
    (OUTDIR / "ablation_test.json").write_text(json.dumps(evaluated["test"], indent=2, ensure_ascii=False))
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
