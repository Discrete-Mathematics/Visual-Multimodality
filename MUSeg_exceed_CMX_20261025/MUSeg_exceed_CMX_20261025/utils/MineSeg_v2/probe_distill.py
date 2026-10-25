"""One validation batch: teacher accuracy and the distillation terms."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/MineSeg/v2")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source")
sys.path.insert(0, "/root/autodl-tmp/SelectiveMine_best_20261024/source/tools")

from distill_debf import PairedDataset, SPLIT_ROOT
from models import MineSegmentor
from selective_inference_bundle_20261023 import load_for_inference

WEIGHTS = "/root/autodl-tmp/weights/mit-b2.bin"


def accuracy(pred, label):
    valid = label != 255
    return float((pred[valid] == label[valid]).float().mean())


def main():
    student = MineSegmentor("b2", WEIGHTS, "debf", num_classes=16).cuda().eval()
    payload = torch.load(
        "/root/autodl-tmp/MineSeg/v2/runs/debf_distill_b2/best.pth",
        map_location="cpu",
        weights_only=False,
    )
    student.load_state_dict(payload["model"], strict=True)
    teacher, _ = load_for_inference(
        "/root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth",
        device="cuda",
    )
    teacher.load_state_dict(
        torch.load(
            "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
            map_location="cpu",
            weights_only=False,
        )["model"],
        strict=True,
    )
    teacher.eval()
    loader = DataLoader(
        PairedDataset(SPLIT_ROOT / "val.txt", 768, 896, train=False, rare_crop=0.0),
        batch_size=2,
        shuffle=False,
        num_workers=2,
    )
    batch = next(iter(loader))
    rgb = batch["rgb"].cuda()
    label = batch["label"].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        student_logits = student(
            rgb,
            batch["debf_depth"].cuda(),
            batch["valid"].cuda(),
            batch["edge"].cuda(),
        )["logits"].float()
        teacher_log = teacher(
            rgb,
            batch["depth"].cuda(),
            batch["valid"].cuda(),
            batch["luma"].cuda(),
            batch["hha"].cuda(),
        )["logits"].float()
    teacher_prob = teacher_log.exp()
    teacher_prob = teacher_prob / teacher_prob.sum(1, keepdim=True).clamp_min(1e-6)
    valid = label != 255
    kl = F.kl_div(
        F.log_softmax(student_logits, 1).permute(0, 2, 3, 1)[valid],
        teacher_prob.permute(0, 2, 3, 1)[valid],
        reduction="batchmean",
    )
    agree = (student_logits.argmax(1) == teacher_prob.argmax(1))[valid].float().mean()
    print(
        {
            "student_pa": round(accuracy(student_logits.argmax(1), label), 4),
            "teacher_pa": round(accuracy(teacher_prob.argmax(1), label), 4),
            "agree": round(float(agree), 4),
            "kl": round(float(kl), 4),
            "student_logit_std": round(float(student_logits.std()), 3),
            "teacher_log_std": round(float(teacher_log.std()), 3),
            "teacher_prob_max": round(float(teacher_prob.max(1).values[valid].mean()), 4),
        }
    )


if __name__ == "__main__":
    main()
