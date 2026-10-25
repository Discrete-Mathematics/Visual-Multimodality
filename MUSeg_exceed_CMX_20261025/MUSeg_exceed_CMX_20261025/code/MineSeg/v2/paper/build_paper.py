#!/usr/bin/env python3
"""Render the v2-only MUSeg report to Markdown, LaTeX, Word, and a number record."""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("/root/autodl-tmp/MineSeg/v2/paper")
RUNS = Path("/root/autodl-tmp/MineSeg/v2/runs")

CLASSES = [
    "人员 person",
    "线缆 cable",
    "管路 tube",
    "指示装置 indicator",
    "电气设备 electrical equipment",
    "电子设备 electronic equipment",
    "采掘设备 mining equipment",
    "轨道区域 rail area",
    "支护设备 support equipment",
    "门 door",
    "工具材料 tools and materials",
    "救援设备 rescue equipment",
    "容器 container",
    "金属构件 metal fixture",
    "锚固设备 anchoring equipment",
]


def load(path):
    return json.loads(Path(path).read_text())


def rows_jsonl(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def pct(value):
    value = float(value)
    if abs(value) <= 1.5:
        value *= 100.0
    return f"{value:.2f}"


def best_row(rows, key):
    usable = [row for row in rows if row.get(key) is not None]
    return max(usable, key=lambda row: row[key])


def tex_escape(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def main():
    ftrans_hist = rows_jsonl(RUNS / "selective_ftrans/history.jsonl")
    ftrans_test = load(RUNS / "selective_ftrans_test_single/official_test.json")
    ftrans_cfg = load(RUNS / "selective_ftrans/config.json")
    prior = load(RUNS / "selective_ftrans_test_single/prior_sweep.json")
    resolution = load(RUNS / "selective_ftrans_test_single/resolution_val.json")
    debf_test = load(RUNS / "debf_b5_ftrans/test_ftrans.json")
    debf_hist = rows_jsonl(RUNS / "debf_b5_ftrans/history.jsonl")
    debf_cfg = load(RUNS / "debf_b5_ftrans/config.json")
    debf_b2 = load(RUNS / "debf_b2/holdout_eval.json")
    debf_b2_cfg = load(RUNS / "debf_b2/config.json")
    debf_b2_hist = rows_jsonl(RUNS / "debf_b2/history.jsonl")
    debf_hi_hist = rows_jsonl(RUNS / "debf_b2_hi/history.jsonl")
    debf_hi_cfg = load(RUNS / "debf_b2_hi/config.json")
    pa2_hist = rows_jsonl(RUNS / "selective_pa2/history.jsonl")
    pa2_cfg = load(RUNS / "selective_pa2/config.json")
    pa_hist = rows_jsonl(RUNS / "selective_pa/history.jsonl")

    saved7 = next(row for row in ftrans_hist if row["epoch"] == 7)
    best16 = best_row(ftrans_hist, "miou_16")
    last_f = ftrans_hist[-1]
    pa2_best_pa = best_row(pa2_hist, "pa_16")
    pa2_best_mpa = best_row(pa2_hist, "mpa_16")
    pa2_met = any(row["miou_16"] >= 0.610 and row["mpa_16"] >= 0.728 for row in pa2_hist)

    val_runs = []
    specs = [
        ("selective_hi", "开发集 314", "已保存"),
        ("selective_hard", "开发集 314", "已保存"),
        ("selective_lowlr", "开发集 314", "已保存"),
        ("selective_ftrans", "FTrans 788", "已保存"),
        ("selective_margin", "FTrans 788", "已保存"),
        ("selective_pa", "FTrans 788", "未保存"),
        ("selective_pa2", "FTrans 788", "未保存"),
    ]
    for name, split, saved in specs:
        hist = rows_jsonl(RUNS / name / "history.jsonl")
        cfg = load(RUNS / name / "config.json")
        key = "pa_16" if cfg["args"].get("select_pa") else "holdout_miou"
        chosen = best_row(hist, key)
        val_runs.append(
            [
                name,
                f"{len(hist)}/{cfg['args']['epochs']}",
                split,
                str(chosen["epoch"]),
                pct(chosen["holdout_miou"]),
                pct(chosen["pa_16"]),
                pct(chosen["mpa_16"]),
                pct(chosen["miou_16"]),
                saved,
            ]
        )

    distill_rows = []
    for name, planned in (
        ("debf_distill_b2", 16),
        ("debf_distill_b2_v2", 12),
        ("debf_distill_b5", 10),
    ):
        hist = rows_jsonl(RUNS / name / "history.jsonl")
        chosen = best_row(hist, "miou_16")
        distill_rows.append(
            [
                name,
                f"{len(hist)}/{planned}",
                str(chosen["epoch"]),
                pct(chosen["pa_16"]),
                pct(chosen["mpa_16"]),
                pct(chosen["miou_16"]),
                pct(chosen["pa_15"]),
                pct(chosen["miou_15"]),
            ]
        )

    scale_rows = []
    for height, width in ((480, 640), (640, 768), (768, 896), (896, 1024)):
        data = load(RUNS / f"scale_{height}x{width}/dev_val.json")
        scale_rows.append(
            [
                f"{height}×{width}",
                pct(data["pa_16"]),
                pct(data["mpa_16"]),
                pct(data["miou_16"]),
                pct(data["holdout_pa"]),
                pct(data["holdout_miou"]),
            ]
        )

    wide_rows = []
    for label, path in (
        ("发布权重 480×640", RUNS / "selective_480_test/official_test.json"),
        ("hi 第4轮 640×768", RUNS / "selective_test_640/official_test.json"),
        ("hi 第4轮 768×896", RUNS / "selective_hi/official_test.json"),
        ("hi 第4轮 768×896 TTA", RUNS / "selective_hi/official_test_tta.json"),
        ("两权重平均 768×896", RUNS / "selective_ensemble/official_test.json"),
    ):
        data = load(path)
        wide_rows.append(
            [
                label,
                str(data["images"]),
                pct(data["pa_16"]),
                pct(data["mpa_16"]),
                pct(data["miou_16"]),
                pct(data["holdout_pa"]),
                pct(data["holdout_mpa"]),
                pct(data["holdout_miou"]),
            ]
        )

    resolution_rows = [
        [
            f"{row['height']}×{row['width']}",
            f"{row['pa16']:.2f}",
            f"{row['mpa16']:.2f}",
            f"{row['miou16']:.2f}",
            f"{row['pa15']:.2f}",
            f"{row['miou15']:.2f}",
        ]
        for row in resolution
    ]
    prior_val = []
    for bias, row in prior["val"].items():
        prior_val.append(
            [
                f"{float(bias):.2f}",
                f"{row['pa16']:.2f}",
                f"{row['mpa16']:.2f}",
                f"{row['miou16']:.2f}",
            ]
        )

    class_rows = []
    for name, iou_s, support_s, item in zip(
        CLASSES,
        ftrans_test["per_class_iou"],
        ftrans_test["per_class_support"],
        debf_test["per_class"],
    ):
        class_rows.append(
            [
                name,
                pct(iou_s),
                str(int(support_s)),
                pct(item["iou"]),
                str(int(item["support"])),
            ]
        )

    mine_rows = []
    for mine, scores in debf_test["per_mine"].items():
        mine_rows.append([mine, pct(scores["pa"]), pct(scores["mpa"]), pct(scores["miou"])])
    b2_mines = []
    for mine, scores in debf_b2["per_mine"].items():
        b2_mines.append([mine, pct(scores["pa"]), pct(scores["mpa"]), pct(scores["miou"])])

    chosen_test = prior["test_at_chosen_bias"]
    zero_test = prior["test_at_zero"]

    blocks = []

    def h(level, text):
        blocks.append(("h", level, text))

    def p(text):
        blocks.append(("p", text))

    def table(caption, headers, rows):
        blocks.append(("t", caption, headers, rows))

    h(1, "MineSeg v2 实验报告：MUSeg 上的 DEBFNet 与 SelectiveMine")
    p(
        "日期：2026-10-25。本文只整理 /root/autodl-tmp/MineSeg/v2 目录中已经写盘的配置、训练历史和评测 JSON。"
        "百分比保留两位小数。没有把 v3 目录中的深度补全结果写进来，也没有把验证集上的最高像素准确率写成测试结论。"
        "像素准确率、平均像素准确率和平均交并比没有同时超过 MUSeg 论文 Table 5 的 CMX。"
    )

    h(2, "摘要")
    p(
        "v2 是 2026-10-25 这一轮实验的代码和运行目录。里面有两条模型线。"
        "DEBFNet 定义在 v2/models.py，使用 MiT RGB 编码器和轻量深度编码器，在每个尺度做深度边缘双向门控，只在最粗尺度加双向交叉注意力。"
        "SelectiveMine 的微调不使用这个文件里的同名分支，而是加载 2026-10-24 发布的检查点："
        "MineRelay 融合、独立的 RGB 外观解码器，以及一个空间路由器；深度缺失时回退到 RGB 头。"
        "v3 是随后加上的深度补全前端，训练记录在 /root/autodl-tmp/MineSeg/v3/runs，对应论文在 v3/paper。那一版把 v3 的补全结果和 v2 的测试结果写在一起，不能当作 v2 目录本身的记录。"
    )
    p(
        f"和 MUSeg Table 5 对齐的协议是 16 类：原始标签 0 算一类，只忽略 255。CMX（MiT-B2，RGB-D HHA）为像素准确率 86.30%、平均像素准确率 72.72%、平均交并比 61.83%。"
        f"v2 里写进 FTransUNet 测试集 788 张、并且对应已保存权重的 SelectiveMine 主结果，是 selective_ftrans 第 {saved7['epoch']} 轮、768×896、关闭测试时增强："
        f"16 类像素准确率 {pct(ftrans_test['pa_16'])}%、平均像素准确率 {pct(ftrans_test['mpa_16'])}%、平均交并比 {pct(ftrans_test['miou_16'])}%。"
        f"后两项高于 CMX，像素准确率低 0.37 个百分点。"
        f"同一文件里的 15 类数字（同时忽略 0 和 255）是 {pct(ftrans_test['holdout_pa'])}%、{pct(ftrans_test['holdout_mpa'])}%、{pct(ftrans_test['holdout_miou'])}%，不能和 86.30% 比较。"
        f"DEBF-B5 计划 30 轮，实际停在第 {debf_test['epoch']} 轮。六视图测试时增强下的 15 类测试结果是 {pct(debf_test['pa_percent'])}%、{pct(debf_test['mpa_percent'])}%、{pct(debf_test['miou_percent'])}%，三项都低于 Table 5，也低于上面的 SelectiveMine。"
    )

    h(2, "1  v2 与 v3")
    p(
        "v2 目录包含模型、训练和评测脚本，以及 runs 下的每次运行。DEBFNet 的训练入口是 v2/train.py，评测入口是 v2/evaluate.py。"
        "SelectiveMine 的微调入口是 v2/finetune_selective.py 和 v2/stage2_selective.py，权重来自 /root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth。"
        "v2/models.py 里也有一个名叫 selective 的 MiT 分支，带 RGB 头和空间路由器，但 selective_ftrans、selective_hi 这些目录里的 best.pth 不是它训出来的。"
    )
    p(
        "v3 在 SelectiveMine 前面加了一个补全模块：多尺度归一化卷积填深度空洞，再预测深度、HHA 和低照度 RGB 残差。代码在 v3/complement.py，运行目录是 v3/runs/selective_complement 和 v3/runs/debf_complement。"
        "v3/paper 里 SelectiveMine-C 的 86.10%、76.64%、62.41% 来自那次补全实验的测试时增强，不属于 v2。"
    )

    h(2, "2  数据与协议")
    p(
        "图像来自 MUSeg，六个矿井的 Azure Kinect DK，裁剪后为 1082×932。语义类是标签 1 至 15，标签 0 是大片未标注区域，255 为忽略。"
        "本文使用两套划分，数字不混在一张主表里。"
    )
    p(
        "FTransUNet 划分：train.txt 1595 张、val.txt 788 张、test.txt 788 张，路径在 /root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/。"
        "selective_ftrans、selective_margin、selective_pa、selective_pa2 和 debf_b5_ftrans 使用这一划分。检查点按验证集选择，测试集不参与保存。"
    )
    p(
        "另一套是发布权重自带的开发划分，外加 MUSeg 论文给出的 1576 张列表。"
        "开发训练 1281 张、开发验证 314 张，文件在 SelectiveMine_best_20261024/source/splits/。"
        "1576 张列表是 /root/autodl-tmp/Experiment/DatasetSplit/val.txt，等于 FTransUNet 的验证集加测试集。"
        "selective_hi、selective_hard、selective_lowlr 和早期的分辨率扫描使用开发划分做选择，再在这 1576 张上评测。"
        "因为 1576 张包含后来 selective_ftrans 用作验证集的 788 张，所以 selective_ftrans 没有在这 1576 张上做测试。"
    )
    p(
        "16 类协议把标签 0 计入混淆矩阵，只忽略 255。它的像素准确率落在 86% 附近，用来和 Table 5 比较。"
        "15 类协议同时忽略 0 和 255，像素准确率大约 79% 到 81%，平均交并比通常更高，因为未标注区域不再拉低平均。"
        "部分 JSON 里的 beats_cmx 用 15 类像素准确率去比 86.30%，这个布尔值会一律为假，本文不采用它。"
        "Table 5 的其余数字是引用，不是这次重跑：DFormer 85.63 / 70.37 / 59.74，SA-Gate 84.77 / 71.34 / 59.38，CMNeXt 84.05 / 66.77 / 56.06，"
        "SegFormer RGB-D 83.00 / 65.25 / 55.20，SegFormer 深度 HHA 81.16 / 62.21 / 51.28，SegFormer RGB 78.75 / 58.54 / 47.93，DeeplabV3+ RGB 76.67 / 54.91 / 43.64。"
    )

    h(2, "3  模型")
    h(3, "3.1  SelectiveMine")
    p(
        "微调所用的 SelectiveMine 复用 MineRelay 的特征提取和融合，另有一个独立的 RGB 外观解码器。"
        "路由器的输入是两个专家的概率、两者的差异、熵、覆盖和亮度，不读真值。深度缺失时混合权重落到 RGB 头。"
        "评测前把 multimodal.mix 填成 1，并按 joint 模式配置。输入顺序是 RGB、深度、有效掩码、亮度和 HHA。"
        f"selective_ftrans 从 selective_hi 的 best.pth 继续，配置写明该文件是第 4 轮。训练 16 轮，批大小 4，768×896，学习率 1e-5，种子 2026，稀有类裁剪概率 0.45，Lovász 权重 0.25。"
        f"保存标准是 15 类验证平均交并比。最高值在第 7 轮：15 类平均交并比 {pct(saved7['holdout_miou'])}%、平均像素准确率 {pct(saved7['holdout_mpa'])}%、像素准确率 {pct(saved7['holdout_pa'])}%；"
        f"对应 16 类为 {pct(saved7['miou_16'])}%、{pct(saved7['mpa_16'])}%、{pct(saved7['pa_16'])}%。"
        f"第 {best16['epoch']} 轮的 16 类平均交并比更高，是 {pct(best16['miou_16'])}%，但 15 类没有超过第 7 轮，权重没有被留下。"
        f"第 16 轮的 16 类像素准确率是本组验证里最高的 {pct(last_f['pa_16'])}%，同样没有保存。测试数字只对应该目录的 best.pth。"
    )
    h(3, "3.2  DEBFNet")
    p(
        f"DEBFNet 的 RGB 编码器是 MiT，B2 参数量 {debf_b2_cfg['parameters'] / 1e6:.2f}M，B5 参数量 {debf_cfg['parameters'] / 1e6:.2f}M。"
        "深度编码器是四级卷积，步长依次为 4、2、2、2，不使用第二个 MiT。"
        "每个尺度用深度边缘双向门控：深度边缘投影后加进深度流，再乘有效掩码；RGB 和深度互相生成门控。"
        "只有最粗一层、且特征图不超过 4096 个位置时，才做双向交叉注意力，残差投影初始化为 0。"
        "解码头是 SegFormer 的 MLP 融合，另有一个边界输出。训练时标签 1–15 映到 0–14，原始 0 和 255 都忽略，因此 DEBF 的主指标是 15 类协议。"
        f"debf_b5_ftrans 计划 30 轮，批大小 4，768×896，骨干学习率 4e-5，头学习率 4e-4，种子 2026，开启粘贴和稀有类采样。日志停在第 14 轮，该轮也是验证集最高的一轮："
        f"15 类平均交并比 {pct(debf_hist[-1]['holdout_miou'])}%、平均像素准确率 {pct(debf_hist[-1]['holdout_mpa'])}%、像素准确率 {pct(debf_hist[-1]['holdout_pa'])}%。"
        "测试命令带有 --tta，即尺度 0.75、1.0、1.25 再加水平翻转的六个视图；结果 JSON 本身没有保存这个开关。"
    )

    h(2, "4  运行清单")
    p(
        "下表的最佳轮按该次运行自己的保存标准选取。selective_pa 和 selective_pa2 的标准是：在 16 类平均交并比不低于 61.0% 且平均像素准确率不低于 72.8% 的轮次里，取 16 类像素准确率最高的一轮。"
        "没有任何一轮同时越过这两条线，所以表中给出的是像素准确率最高的一轮，只作记录，没有对应权重。"
        "15 类三列里的平均交并比是训练脚本的 holdout_miou；16 类三列来自同一轮的 miou_16、mpa_16、pa_16。"
    )
    table(
        "表 1  SelectiveMine 各次运行的验证集峰值。单位是百分比。",
        ["运行", "完成/计划", "验证集", "轮次", "15类 mIoU", "16类 PA", "16类 MPA", "16类 mIoU", "权重"],
        val_runs,
    )
    p(
        "selective_hard 和 selective_lowlr 从 selective_hi 继续，验证集仍是 314 张开发集，都只跑了 3 轮。"
        "selective_margin 从 selective_ftrans 的 best.pth 继续，6 轮都跑完，验证集是 FTransUNet 的 788 张，但没有再做测试评测。"
        f"selective_pa 只写出了 1 轮历史。selective_pa2 跑满 12 轮，16 类像素准确率最高在第 {pa2_best_pa['epoch']} 轮，为 {pct(pa2_best_pa['pa_16'])}%，"
        f"同轮平均像素准确率 {pct(pa2_best_pa['mpa_16'])}%、平均交并比 {pct(pa2_best_pa['miou_16'])}%。"
        f"全程最高的 16 类平均像素准确率是 {pct(pa2_best_mpa['mpa_16'])}%，仍低于 72.80% 的保存门槛。"
        "因此 pa2 没有 best.pth，也没有测试数字。"
    )

    h(2, "5  FTransUNet 测试集")
    p(
        "这一节只使用 test.txt 的 788 张。SelectiveMine 是单尺度前向，路由后的 logits 做 argmax。"
        "同一 JSON 还记录了融合头单独的 16 类结果："
        f"像素准确率 {pct(ftrans_test['fused_pa_16'])}%、平均像素准确率 {pct(ftrans_test['fused_mpa_16'])}%、平均交并比 {pct(ftrans_test['fused_miou_16'])}%。"
        "主结果用路由输出，不用融合头单独的数字。"
        "DEBF-B5 只有 15 类输出，表中 16 类三列留空。"
    )
    table(
        "表 2  FTransUNet 测试集。PA、MPA、mIoU 的单位是百分比。CMX 一行引自 MUSeg Table 5。",
        ["方法", "协议", "分辨率", "TTA", "PA", "MPA", "mIoU"],
        [
            ["CMX (MiT-B2, RGB-D HHA)", "Table 5", "—", "—", "86.30", "72.72", "61.83"],
            ["DFormer", "Table 5", "—", "—", "85.63", "70.37", "59.74"],
            ["SA-Gate", "Table 5", "—", "—", "84.77", "71.34", "59.38"],
            ["CMNeXt", "Table 5", "—", "—", "84.05", "66.77", "56.06"],
            [
                "SelectiveMine 第 7 轮",
                "16 类",
                "768×896",
                "关闭",
                pct(ftrans_test["pa_16"]),
                pct(ftrans_test["mpa_16"]),
                pct(ftrans_test["miou_16"]),
            ],
            [
                "SelectiveMine 第 7 轮",
                "15 类",
                "768×896",
                "关闭",
                pct(ftrans_test["holdout_pa"]),
                pct(ftrans_test["holdout_mpa"]),
                pct(ftrans_test["holdout_miou"]),
            ],
            [
                "DEBF-B5 第 14 轮",
                "15 类",
                "768×896",
                "六视图",
                pct(debf_test["pa_percent"]),
                pct(debf_test["mpa_percent"]),
                pct(debf_test["miou_percent"]),
            ],
        ],
    )
    p(
        f"16 类比较下，SelectiveMine 的平均像素准确率比 CMX 高 {float(pct(ftrans_test['mpa_16'])) - 72.72:.2f} 个百分点，"
        f"平均交并比高 {float(pct(ftrans_test['miou_16'])) - 61.83:.2f} 个百分点，"
        f"像素准确率低 {86.30 - float(pct(ftrans_test['pa_16'])):.2f} 个百分点。"
        "DEBF-B5 的 15 类像素准确率是 71.71%，即使不谈协议差异，也低于表中任何一个 RGB-D 基线的 Table 5 像素准确率。"
        "v2 没有 SelectiveMine 的六视图测试时增强文件；那次评测写在 v3/runs/selective_ftrans_test/test_tta.json，不计入本文。"
    )

    h(2, "6  类别与矿井")
    p(
        "表 3 的交并比都是前景类。SelectiveMine 这一列来自 15 类混淆矩阵，15 个数的平均等于表 2 的 15 类平均交并比 65.10%，不是 16 类的 62.42%。"
        "DEBF 这一列同样是 15 类，平均等于 50.35%。两列的支持像素数不同，因为两次评测的插值实现并不共用同一套数据读取代码，不能把支持数的差别解释成类别面积变了。"
        "SelectiveMine 最弱的是容器 37.12% 和工具材料 41.00%。DEBF 最弱的是工具材料 20.42% 和人员 27.33%。"
    )
    table(
        "表 3  FTransUNet 测试集前景类交并比（%）。",
        ["类别", "SelectiveMine IoU", "SelectiveMine 支持", "DEBF-B5 IoU", "DEBF-B5 支持"],
        class_rows,
    )
    p(
        "DEBF-B5 按矿井分开的 15 类结果见表 4。05 号矿井平均交并比最低，为 35.11%；02 号矿井最高，为 60.08%。"
        "SelectiveMine 的测试 JSON 没有按矿井汇总。"
    )
    table(
        "表 4  DEBF-B5 测试集按矿井的 15 类指标（%）。",
        ["矿井", "PA", "MPA", "mIoU"],
        mine_rows,
    )
    p(
        "验证集混淆矩阵在 selective_ftrans_test_single/val_confusion.json，对应第 7 轮、768×896，788 张，16 类像素准确率 86.84%。"
        "错误像素里占比最高的几对是：背景预测成锚固设备、金属构件预测成背景、锚固设备预测成背景、背景预测成金属构件、背景预测成线缆。"
        "像素准确率的缺口主要落在未标注区域和金属、锚固、线缆之间，而不是稀有类单独造成的。"
    )

    h(2, "7  验证集上的分辨率和背景偏置")
    p(
        "分辨率扫描只读 FTransUNet 的 val.txt，检查点是 selective_ftrans 第 7 轮，关闭测试时增强。测试集没有按这张表重跑。"
        "16 类平均交并比最高的是 704×832，为 61.41%；训练分辨率 768×896 是 61.23%。"
        "16 类像素准确率最高的是 800×928，为 86.92%。这些都是验证集数字。"
    )
    table(
        "表 5  selective_ftrans 第 7 轮在验证集上的分辨率。单位是百分比。",
        ["分辨率", "16类 PA", "16类 MPA", "16类 mIoU", "15类 PA", "15类 mIoU"],
        resolution_rows,
    )
    p(
        "背景 logit 偏置也先在验证集上选。候选是 -0.6、-0.3、-0.15、0、0.15、0.3、0.6、0.9，加在 16 类矩阵的第 0 类上。"
        "排序键是“是否三项都超过 CMX，其次是 16 类平均交并比”。验证集上没有任何偏置三项都超过，因此实际选中的是 16 类平均交并比最高的 +0.60。"
        "偏置冻结之后，测试集只再读了一次。这里的 15 类平均交并比是 16 类矩阵里前景类交并比的平均，背景仍然参与每个前景类的并集，所以低于表 2 那种把标签 0 直接忽略的 65.10%。"
    )
    table(
        "表 6  验证集上的背景 logit 偏置。单位是百分比。",
        ["偏置", "16类 PA", "16类 MPA", "16类 mIoU"],
        prior_val,
    )
    table(
        "表 7  偏置冻结后的一次测试。单位是百分比。",
        ["设置", "16类 PA", "16类 MPA", "16类 mIoU"],
        [
            ["偏置 0", f"{zero_test['pa16']:.2f}", f"{zero_test['mpa16']:.2f}", f"{zero_test['miou16']:.2f}"],
            [
                "验证集选中的 +0.60",
                f"{chosen_test['pa16']:.2f}",
                f"{chosen_test['mpa16']:.2f}",
                f"{chosen_test['miou16']:.2f}",
            ],
            ["CMX Table 5", "86.30", "72.72", "61.83"],
        ],
    )
    p(
        f"+0.60 把测试集 16 类像素准确率从 {zero_test['pa16']:.2f}% 提到 {chosen_test['pa16']:.2f}%，"
        f"平均交并比从 {zero_test['miou16']:.2f}% 提到 {chosen_test['miou16']:.2f}%，"
        f"平均像素准确率从 {zero_test['mpa16']:.2f}% 降到 {chosen_test['mpa16']:.2f}%。"
        "像素准确率仍然低于 86.30%。偏置 0 的 16 类三列与表 2 的单尺度测试一致，说明这是同一次条件的重复计算，不是另一个检查点。"
    )

    h(2, "8  1576 张列表上的早期检查点")
    p(
        "下表的划分是 DatasetSplit/val.txt，1576 张。selective_hi 的 best.pth 在日志里对应第 4 轮，选择用的是 314 张开发验证集，不是这 1576 张，也不是 FTransUNet 的 788 张验证集。"
        "发布权重在 480×640 下评测。概率平均使用发布权重和 selective_hi，分辨率 768×896。"
        "这张表的 16 类三项都没有超过 CMX。它不能和第 5 节的 788 张测试混成一个平均值。"
    )
    table(
        "表 8  DatasetSplit/val.txt（1576 张）。单位是百分比。",
        ["设置", "图像数", "16类 PA", "16类 MPA", "16类 mIoU", "15类 PA", "15类 MPA", "15类 mIoU"],
        wide_rows,
    )
    p(
        "同一检查点 selective_hi 在开发验证集 314 张上做过分辨率扫描，关闭测试时增强。JSON 没有写入检查点路径；写出这些文件的命令使用的是 selective_hi/best.pth。"
    )
    table(
        "表 9  selective_hi 在开发验证集 314 张上的分辨率。单位是百分比。",
        ["分辨率", "16类 PA", "16类 MPA", "16类 mIoU", "15类 PA", "15类 mIoU"],
        scale_rows,
    )

    h(2, "9  蒸馏、内部留出和未完成的运行")
    p(
        "三次蒸馏都用 selective_ftrans 第 7 轮做教师，学生是 DEBFNet，验证集是 FTransUNet 的 788 张。配置写明测试集在训练期间不读，目录里也没有测试 JSON。"
        "debf_distill_b2 从随机的 16 类头开始，温度 2，KL 权重 3，交叉熵权重 0.25。"
        "debf_distill_b2_v2 从上一份 best.pth 继续，温度 1，KL 和交叉熵权重都是 1，头学习率倍率 10。"
        "debf_distill_b5 从 debf_b5_ftrans 的 15 类头扩成 16 类后继续，头学习率倍率 8。"
        "三份权重都在，但验证平均交并比停在 36.98% 到 45.99%，没有进入主表。"
    )
    table(
        "表 10  蒸馏的验证集峰值。单位是百分比。没有测试评测。",
        ["运行", "完成/计划", "轮次", "16类 PA", "16类 MPA", "16类 mIoU", "15类 PA", "15类 mIoU"],
        distill_rows,
    )
    p(
        f"debf_b2 是更早的 MiT-B2 运行，480×640，计划 120 轮，实际 42 轮后停止。它的留出集是训练捕获组里划出的 158 张，不是论文测试集。"
        f"第 42 轮 15 类像素准确率 {pct(debf_b2['pa_percent'])}%、平均像素准确率 {pct(debf_b2['mpa_percent'])}%、平均交并比 {pct(debf_b2['miou_percent'])}%。"
        f"debf_b2_hi 计划 {debf_hi_cfg['args']['epochs']} 轮、640×768，历史只有 {len(debf_hi_hist)} 轮，最后一轮 15 类平均交并比 {pct(debf_hi_hist[-1]['holdout_miou'])}%。"
        "debf_full 只写出了 config.json 和划分文件，没有 history.jsonl，也没有权重。计划是从 debf_b2 的 best.pth 以学习率 3e-5 再训 80 轮。"
        "下列三个评测 JSON 没有记录检查点路径，本文不把它们放进比较表：selective_probe/dev_val.json、selective_ftrans_eval/dev_val.json、selective_val_480/dev_val.json。"
    )
    table(
        "表 11  debf_b2 在 158 张内部留出集上按矿井的 15 类指标（%）。不是论文测试集。",
        ["矿井", "PA", "MPA", "mIoU"],
        b2_mines,
    )

    h(2, "10  结论")
    p(
        "v2 目录里，能够和 MUSeg Table 5 放在同一协议下的测试结果只有 selective_ftrans 第 7 轮这一次 16 类评测。"
        f"它在 FTransUNet 测试集上的像素准确率是 {pct(ftrans_test['pa_16'])}%，平均像素准确率是 {pct(ftrans_test['mpa_16'])}%，平均交并比是 {pct(ftrans_test['miou_16'])}%。"
        "平均像素准确率和平均交并比超过 CMX，像素准确率没有超过。验证集上把背景 logit 加 0.60 之后，测试像素准确率升到 86.05%，仍然低 0.25 个百分点，而且平均像素准确率下降。"
        "DEBF-B5 停在第 14 轮，15 类测试平均交并比 50.35%，蒸馏也没有把它补到 Table 5 的量级。"
        "以像素准确率为保存目标的 selective_pa2 跑完了 12 轮，但没有一轮同时满足平均交并比和平均像素准确率的门槛，因此 v2 里不存在那条轨迹的测试结果。"
    )

    h(2, "11  文件")
    p("主测试：/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/official_test.json")
    p("背景偏置：/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/prior_sweep.json")
    p("验证分辨率：/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/resolution_val.json")
    p("SelectiveMine 权重与历史：/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth 与 history.jsonl")
    p("DEBF-B5 测试：/root/autodl-tmp/MineSeg/v2/runs/debf_b5_ftrans/test_ftrans.json")
    p("1576 张列表：/root/autodl-tmp/MineSeg/v2/runs/selective_hi/official_test.json 与 official_test_tta.json")
    p("本文数字记录：/root/autodl-tmp/MineSeg/v2/paper/results.json")

    record = {
        "scope": "v2 runs only; v3 complement results are excluded",
        "date": "2026-10-25",
        "primary_test": {
            "checkpoint": "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
            "epoch": 7,
            "split": ftrans_test["split"],
            "images": ftrans_test["images"],
            "height": ftrans_test["height"],
            "width": ftrans_test["width"],
            "tta": False,
            "pa16": ftrans_test["pa_16"],
            "mpa16": ftrans_test["mpa_16"],
            "miou16": ftrans_test["miou_16"],
            "pa15": ftrans_test["holdout_pa"],
            "mpa15": ftrans_test["holdout_mpa"],
            "miou15": ftrans_test["holdout_miou"],
        },
        "debf_b5_test": {
            "checkpoint": debf_test["checkpoint"],
            "epoch": debf_test["epoch"],
            "tta_command": True,
            "pa15": debf_test["pa_percent"] / 100.0,
            "mpa15": debf_test["mpa_percent"] / 100.0,
            "miou15": debf_test["miou_percent"] / 100.0,
        },
        "prior_test": {
            "chosen_bias": prior["chosen_bias"],
            "zero": zero_test,
            "chosen": chosen_test,
            "beats_cmx_16": prior["beats_cmx_16"],
        },
        "pa2_checkpoint_saved": False,
        "pa2_any_epoch_met_floors": pa2_met,
        "claim": {
            "all_three_metrics_exceed_cmx_on_test": False,
            "reason": "16-class test PA of the saved SelectiveMine checkpoint is 85.93, below CMX 86.30. MPA and mIoU exceed CMX. DEBF-B5 does not.",
        },
        "ftrans_config_init_epoch_logged": ftrans_cfg["args"]["checkpoint"],
    }
    (ROOT / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    write_markdown(blocks)
    write_latex(blocks)
    write_docx(blocks)


def write_markdown(blocks):
    lines = []
    for block in blocks:
        if block[0] == "h":
            level, text = block[1], block[2]
            lines.append("#" * level + " " + text)
            lines.append("")
        elif block[0] == "p":
            lines.append(block[1])
            lines.append("")
        else:
            _, caption, headers, rows = block
            lines.append(caption)
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    (ROOT / "paper.md").write_text("\n".join(lines))


def write_latex(blocks):
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        r"\setCJKmainfont{WenQuanYi Micro Hei}[AutoFakeBold=2.5]",
        r"\setmainfont{TeX Gyre Termes}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\usepackage{setspace}",
        r"\setstretch{1.15}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\begin{document}",
    ]
    for block in blocks:
        if block[0] == "h":
            level, text = block[1], block[2]
            if level == 1:
                parts.append(r"{\centering\Large " + tex_escape(text) + r"\par}")
                parts.append(r"\vspace{0.8em}")
            else:
                parts.append(r"\section*{" + tex_escape(text) + "}")
        elif block[0] == "p":
            parts.append(tex_escape(block[1]))
            parts.append("")
        else:
            _, caption, headers, rows = block
            spec = "l" + "c" * (len(headers) - 1)
            parts.append(r"{\footnotesize\setlength{\tabcolsep}{3pt}")
            parts.append(r"\begin{longtable}{" + spec + "}")
            parts.append(r"\caption*{" + tex_escape(caption) + r"}\\")
            parts.append(r"\toprule")
            parts.append(" & ".join(tex_escape(item) for item in headers) + r"\\")
            parts.append(r"\midrule")
            parts.append(r"\endfirsthead")
            parts.append(r"\toprule")
            parts.append(" & ".join(tex_escape(item) for item in headers) + r"\\")
            parts.append(r"\midrule")
            parts.append(r"\endhead")
            for row in rows:
                parts.append(" & ".join(tex_escape(item) for item in row) + r"\\")
            parts.append(r"\bottomrule")
            parts.append(r"\end{longtable}")
            parts.append("}")
    parts.append(r"\end{document}")
    (ROOT / "paper.tex").write_text("\n".join(parts) + "\n")


def set_run_font(run, size=11, bold=False):
    run.font.size = Pt(size)
    run.bold = bold
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "WenQuanYi Micro Hei")


def write_docx(blocks):
    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "WenQuanYi Micro Hei")
    for block in blocks:
        if block[0] == "h":
            level, text = block[1], block[2]
            paragraph = document.add_paragraph()
            if level == 1:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = paragraph.add_run(text)
                set_run_font(run, 16, True)
            else:
                run = paragraph.add_run(text)
                set_run_font(run, 14 if level == 2 else 12, True)
                run.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
        elif block[0] == "p":
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(8)
            paragraph.paragraph_format.first_line_indent = Cm(0.74)
            run = paragraph.add_run(block[1])
            set_run_font(run, 11, False)
        else:
            _, caption, headers, rows = block
            cap = document.add_paragraph()
            run = cap.add_run(caption)
            set_run_font(run, 10, True)
            table = document.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = "Table Grid"
            for index, header in enumerate(headers):
                cell = table.rows[0].cells[index]
                cell.text = ""
                run = cell.paragraphs[0].add_run(header)
                set_run_font(run, 8, True)
            for r_index, row in enumerate(rows):
                for c_index, value in enumerate(row):
                    cell = table.rows[r_index + 1].cells[c_index]
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(value)
                    set_run_font(run, 8, False)
            document.add_paragraph()
    document.save(ROOT / "paper.docx")


if __name__ == "__main__":
    main()
