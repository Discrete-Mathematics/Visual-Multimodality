#!/usr/bin/env python3
"""Record finished MUSeg runs and render the same paper to md, tex, and docx."""

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("/root/autodl-tmp/MineSeg/v3/paper")
RUNS = Path("/root/autodl-tmp/MineSeg")

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
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def pct(value, digits=2):
    return round(float(value) * 100.0, digits)


def fmt(value, digits=2):
    return f"{float(value):.{digits}f}"


def main():
    ftrans = load(RUNS / "v3/runs/selective_ftrans_test/test_tta.json")
    comp = load(RUNS / "v3/runs/selective_complement/test_tta.json")
    single = load(RUNS / "v2/runs/selective_ftrans_test_single/official_test.json")
    debf = load(RUNS / "v2/runs/debf_b5_ftrans/test_ftrans.json")
    pa2_cfg = load(RUNS / "v2/runs/selective_pa2/config.json")
    pa2_hist = rows_jsonl(RUNS / "v2/runs/selective_pa2/history.jsonl")
    ftrans_cfg = load(RUNS / "v2/runs/selective_ftrans/config.json")
    comp_cfg = load(RUNS / "v3/runs/selective_complement/config.json")
    debf_cfg = load(RUNS / "v2/runs/debf_b5_ftrans/config.json")
    debf_hist = rows_jsonl(RUNS / "v2/runs/debf_b5_ftrans/history.jsonl")
    comp_hist = rows_jsonl(RUNS / "v3/runs/selective_complement/history.jsonl")

    record = {
        "date": "2026-10-25",
        "dataset": {
            "name": "MUSeg",
            "root": "/root/autodl-tmp/MUSeg",
            "flat": "/root/autodl-tmp/MUSeg/flat",
            "splits": {
                "train": "/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/train.txt",
                "val": "/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/val.txt",
                "test": "/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/test.txt",
            },
            "counts": {"train": 1595, "val": 788, "test": 788},
            "citation": "Scientific Data 2025, doi:10.1038/s41597-025-05493-9",
        },
        "protocol": {
            "sixteen_class": "raw label 0 is a class; label 255 is ignored. PA is on the same scale as MUSeg Table 5.",
            "fifteen_class": "raw labels 0 and 255 are both ignored. PA is about 80 and is not comparable to Table 5.",
            "table5_cmx": {"pa": 86.30, "mpa": 72.72, "miou": 61.83, "backbone": "MiT-B2", "input": "RGB-D HHA"},
            "table5_others": {
                "DFormer": {"pa": 85.63, "mpa": 70.37, "miou": 59.74},
                "SA-Gate": {"pa": 84.77, "mpa": 71.34, "miou": 59.38},
                "CMNeXt": {"pa": 84.05, "mpa": 66.77, "miou": 56.06},
                "SegFormer RGB-D": {"pa": 83.00, "mpa": 65.25, "miou": 55.20},
                "SegFormer Depth(HHA)": {"pa": 81.16, "mpa": 62.21, "miou": 51.28},
                "SegFormer RGB": {"pa": 78.75, "mpa": 58.54, "miou": 47.93},
                "DeeplabV3+ RGB": {"pa": 76.67, "mpa": 54.91, "miou": 43.64},
            },
        },
        "test": {
            "selective_ftrans_tta": {
                "checkpoint": "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth",
                "saved_epoch": 7,
                "selection": "15-class holdout mIoU during finetune_selective.py",
                "resolution": "768x896",
                "tta": "scales 0.75/1.0/1.25 and horizontal flip, six probability views",
                "images": ftrans["images"],
                "split": ftrans["split"],
                "pa16": ftrans["pa_16"],
                "mpa16": ftrans["mpa_16"],
                "miou16": ftrans["miou_16"],
                "pa15": ftrans["holdout_pa"],
                "mpa15": ftrans["holdout_mpa"],
                "miou15": ftrans["holdout_miou"],
                "per_class_iou_15": ftrans["per_class_iou"],
                "per_class_support_15": ftrans["per_class_support"],
                "beats": ftrans["beats"],
                "json": "/root/autodl-tmp/MineSeg/v3/runs/selective_ftrans_test/test_tta.json",
            },
            "selective_complement_tta": {
                "checkpoint": "/root/autodl-tmp/MineSeg/v3/runs/selective_complement/best.pth",
                "saved_epoch": 5,
                "selection": "16-class validation mIoU",
                "resolution": "640x768",
                "tta": "scales 0.75/1.0/1.25 and horizontal flip, six probability views",
                "images": comp["images"],
                "split": comp["split"],
                "pa16": comp["pa_16"],
                "mpa16": comp["mpa_16"],
                "miou16": comp["miou_16"],
                "pa15": comp["holdout_pa"],
                "mpa15": comp["holdout_mpa"],
                "miou15": comp["holdout_miou"],
                "per_class_iou_15": comp["per_class_iou"],
                "per_class_support_15": comp["per_class_support"],
                "beats": comp["beats"],
                "json": "/root/autodl-tmp/MineSeg/v3/runs/selective_complement/test_tta.json",
            },
            "selective_ftrans_single": {
                "resolution": f"{single.get('height')}x{single.get('width')}",
                "tta": False,
                "images": single["images"],
                "split": single["split"],
                "pa16": single["pa_16"],
                "mpa16": single["mpa_16"],
                "miou16": single["miou_16"],
                "pa15": single["holdout_pa"],
                "mpa15": single["holdout_mpa"],
                "miou15": single["holdout_miou"],
                "seconds": single.get("seconds"),
                "json": "/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/official_test.json",
            },
            "debf_b5": {
                "checkpoint": debf["checkpoint"],
                "epoch": debf["epoch"],
                "variant": debf["variant"],
                "protocol": "15-class, labels 0 and 255 ignored",
                "resolution": f"{debf['height']}x{debf['width']}",
                "tta": True,
                "images": debf["images"],
                "split": debf["split"],
                "pa": debf["pa_percent"],
                "mpa": debf["mpa_percent"],
                "miou": debf["miou_percent"],
                "per_class": debf["per_class"],
                "per_mine": debf["per_mine"],
                "beats_cmx": debf["beats_cmx"],
                "json": "/root/autodl-tmp/MineSeg/v2/runs/debf_b5_ftrans/test_ftrans.json",
            },
        },
        "configs": {
            "selective_ftrans": ftrans_cfg,
            "selective_complement": comp_cfg,
            "debf_b5_ftrans": debf_cfg,
            "selective_pa2": pa2_cfg,
        },
        "validation_histories": {
            "selective_pa2": pa2_hist,
            "selective_complement": comp_hist,
            "debf_b5_ftrans_last": debf_hist[-1] if debf_hist else None,
            "debf_b5_epochs_run": len(debf_hist),
        },
        "selective_pa2_outcome": {
            "weights_saved": False,
            "reason": "select_pa saves only when miou_16>=0.610 and mpa_16>=0.728; no epoch met the MPA floor, so best.pth was not written and no test evaluation of this run exists.",
            "directory": "/root/autodl-tmp/MineSeg/v2/runs/selective_pa2",
        },
        "claim": {
            "all_three_metrics_exceed_cmx_on_test": False,
            "reason": "16-class test PA is 86.10 at best, below CMX 86.30. MPA and mIoU of SelectiveMine exceed CMX. DEBF-B5 does not.",
        },
    }
    (ROOT / "results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def cell16(blob, key):
        return fmt(pct(blob[key]))

    pa2_table = []
    for row in pa2_hist:
        pa2_table.append([
            str(row["epoch"]),
            fmt(row["loss"], 3),
            fmt(pct(row["miou_16"])),
            fmt(pct(row["mpa_16"])),
            fmt(pct(row["pa_16"])),
            fmt(pct(row["holdout_miou"])),
            fmt(row["seconds"], 1),
        ])
    best_pa_row = max(pa2_hist, key=lambda row: row["pa_16"])
    best_mpa_row = max(pa2_hist, key=lambda row: row["mpa_16"])

    fg_headers = ["类别", "F-TTA IoU", "C-TTA IoU", "DEBF-B5 IoU"]
    fg_rows = []
    for index, name in enumerate(CLASSES):
        fg_rows.append([
            name,
            fmt(pct(ftrans["per_class_iou"][index])),
            fmt(pct(comp["per_class_iou"][index])),
            fmt(debf["per_class"][index]["iou"] * 100),
        ])

    mine_rows = []
    for mine, stats in debf["per_mine"].items():
        mine_rows.append([
            mine,
            fmt(stats["pa"] * 100),
            fmt(stats["mpa"] * 100),
            fmt(stats["miou"] * 100),
        ])

    debf_last = debf_hist[-1]
    blocks = []

    def h(level, text):
        blocks.append(("h", level, text))

    def p(text):
        blocks.append(("p", text))

    def t(caption, headers, rows):
        blocks.append(("t", caption, headers, rows))

    h(1, "MUSeg 官方划分上的 RGB-D 语义分割实验报告")
    p("日期：2026-10-25。数据来自已结束的训练与一次测试评测，数值取自对应 JSON，未作填补。本文不声称像素准确率、平均像素准确率和平均交并比三项同时超过 MUSeg 论文 Table 5 的 CMX。")

    h(2, "摘要")
    p(
        "本文在 MUSeg 地下矿山 RGB-D 语义分割数据集上，使用给定的 FTransUNet 划分（训练 1595、验证 788、测试 788）评估两条模型线：SelectiveMine 及其深度补全前端 SelectiveMine-C，以及 MiT-B5 的 DEBFNet。比较对象是原论文 Table 5 中的 CMX（MiT-B2，RGB-D HHA）：像素准确率 86.30%，平均像素准确率 72.72%，平均交并比 61.83%。与该表同一量级的协议把原始标签 0 计为一个类、忽略 255，记为 16 类协议。忽略 0 和 255 的 15 类协议像素准确率大约在 80%，不能与 86.30% 直接比较。"
    )
    p(
        f"测试集 788 张上，SelectiveMine 高分辨率检查点（保存于第 7 轮，768×896，尺度 0.75/1.0/1.25 加水平翻转的六视图测试时增强）16 类结果为像素准确率 {cell16(ftrans,'pa_16')}%、平均像素准确率 {cell16(ftrans,'mpa_16')}%、平均交并比 {cell16(ftrans,'miou_16')}%。SelectiveMine-C（保存于第 5 轮，640×768，同样的测试时增强）为 {cell16(comp,'pa_16')}%、{cell16(comp,'mpa_16')}%、{cell16(comp,'miou_16')}%。两条结果的平均像素准确率和平均交并比都高于 CMX，像素准确率分别低 0.26 和 0.20 个百分点。DEBF-B5 在第 14 轮停止（计划 30 轮），15 类测试时增强结果为像素准确率 {fmt(debf['pa_percent'])}%、平均像素准确率 {fmt(debf['mpa_percent'])}%、平均交并比 {fmt(debf['miou_percent'])}%，三项都低于 Table 5。"
    )
    p(
        f"以验证集 16 类像素准确率为选择目标的 12 轮微调（selective_pa2）已经跑完。验证集最高像素准确率出现在第 {best_pa_row['epoch']} 轮，为 {fmt(pct(best_pa_row['pa_16']))}%，对应平均像素准确率 {fmt(pct(best_pa_row['mpa_16']))}%、平均交并比 {fmt(pct(best_pa_row['miou_16']))}%。保存条件要求 16 类平均交并比至少 61.0% 且平均像素准确率至少 72.8%。全程没有任何一轮同时满足，因此没有写出 best.pth，也没有这条轨迹的测试集数字。验证集像素准确率不能当作测试集结论。"
    )

    h(2, "1 问题与协议")
    p("MUSeg 采集自六个矿井的 Azure Kinect DK，裁剪后的 RGB-D 分辨率为 1082×932，语义标签为 1 至 15。标签 0 覆盖大片未标注区域，255 为忽略。原论文指标公式的类别下标从 0 写到 C，和“只评估 15 个前景类”的实现并不相同。本仓库里两种实现都存在，数值差一个明显的台阶。")
    p("16 类协议把标签 0 算进混淆矩阵，只忽略 255。强模型的像素准确率落在 86% 附近，和 Table 5 的 86.30% 处于同一量级，因此和 CMX、DFormer、SA-Gate、CMNeXt 的比较使用这一协议。15 类协议同时忽略 0 和 255，像素准确率约 80%，平均交并比往往高于 16 类，因为未标注区域的交并比会把 16 类平均拉低。两种数字在正文中分列，不混用。")
    p("划分文件固定为 FTransUNet 的 train.txt、val.txt 和 test.txt，没有重新随机划分。检查点只按验证集选择，测试集每个已保存检查点只评测一次（另有一次关闭测试时增强的对照）。按测试集扫分辨率或反复挑选检查点不在本报告的结果之内。")

    h(2, "2 方法")
    h(3, "2.1 SelectiveMine")
    p("SelectiveMine 使用 RGB 外观解码器、MineRelay 融合（MineGate 与 HHA 几何）和空间效用路由。输入为 ImageNet 归一化的 RGB、打包深度（对数深度、有效掩码、梯度）、有效掩码、亮度以及 HHA。对数深度的上下界沿用已发布实现中的 log(617) 与 log(5658)。输出 logits 在对数空间，缺失深度时回退到 RGB 头。发布权重位于 /root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth。后续微调从该权重开始，再经过 selective_hi，进入 selective_ftrans。")
    p("selective_ftrans 的设置是：16 轮，批大小 4，768×896，学习率 1e-5，随机种子 2026，稀有类裁剪概率 0.45，Lovász 权重 0.25，4 个数据进程。该脚本按 15 类验证平均交并比保存检查点。实际写出的 best.pth 是第 7 轮，验证集 16 类像素准确率 86.84%、平均像素准确率 73.49%、平均交并比 61.22%，15 类平均交并比 63.42%。历史里更晚的轮次在 16 类上更高，但 15 类没有超过第 7 轮，所以那些权重没有被保存。测试数字对应该文件，而不是历史中 16 类最高的那一轮。")

    h(3, "2.2 SelectiveMine-C")
    p("补全前端接在 SelectiveMine 之前。深度和 HHA 先做多尺度归一化卷积填洞，核大小为奇数 15、9、5、3，已观测像素保持不变。一个三层卷积预测深度残差、HHA 残差、RGB 残差和置信度。最后一层卷积权重初始化为 0，置信度通道的偏置初始化为 -4，因此加载旧检查点的第一步几乎是恒等映射。训练时随机遮住 30% 的有效深度像素，用 L1 重建被遮住的值，用来监督 Kinect 空洞。低照度残差乘以亮度低于 0.45 的暗区比例，只在暗处改 RGB。补全损失权重 0.25，分割损失为原 selective loss 加 0.35 的 Lovász。")
    p("该次训练 6 轮，批大小 2，640×768，学习率 1.2e-5，种子 2026，初始化为 selective_ftrans 的 best.pth。显存上限设为设备的 0.32，以避免和同时占用 GPU 的其他进程冲突。检查点按验证集 16 类平均交并比保存，最佳是第 5 轮：16 类像素准确率 86.98%、平均像素准确率 74.36%、平均交并比 61.19%。第 6 轮回落到 61.01%，未覆盖该检查点。测试评测使用训练分辨率 640×768，因此它不是与 768×896 的 SelectiveMine 严格对齐的消融。")

    h(3, "2.3 DEBFNet")
    p("DEBFNet 使用 MiT RGB 编码器和轻量深度编码器，只在最粗尺度做深度边缘双向门控，分割头沿用 SegFormer。本报告中的 B5 运行把标签 1–15 映到 0–14，忽略 0 和 255，所以它的指标是 15 类协议。深度通道为 log1p、边缘和有效掩码，再按 (x-0.5)/0.25 归一化。训练计划 30 轮，批大小 4，768×896，骨干学习率 4e-5，头学习率 4e-4，种子 2026，开启稀有类采样和粘贴。实际停在第 14 轮。验证集最后一轮 15 类像素准确率 "
      f"{fmt(pct(debf_last.get('holdout_pa', debf_last.get('pa', 0))))}%，平均像素准确率 {fmt(pct(debf_last.get('holdout_mpa', debf_last.get('mpa', 0))))}%，平均交并比 {fmt(pct(debf_last.get('holdout_miou', debf_last.get('miou', 0))))}%。测试使用同一分辨率和与 SelectiveMine 相同的六视图测试时增强。")
    p("把该检查点扩成 16 类头并蒸馏到 SelectiveMine 的尝试使验证平均交并比掉到 40% 多，已经中止，不纳入主表。B2 蒸馏轨迹最高 16 类平均交并比约 41.7%，同样不作为主结果。")

    h(2, "3 实验设置")
    p("硬件为一张 NVIDIA RTX PRO 6000 Blackwell，显存 97887 MiB。评测时训练进程已经退出。优化器、损失和数据增强以各运行目录下的 config.json 为准，路径列在第 6 节。类别权重由训练集像素频率计算，人员、救援设备和容器等稀有类权重大于 1，背景权重大幅压低。selective_pa2 额外把选择分数改成条件像素准确率，其余超参与 selective_ftrans 相同，初始化为其 best.pth，共 12 轮。")
    p("测试时增强对 RGB、深度、有效掩码、亮度和 HHA 同步缩放并可选水平翻转。缩放后的高宽对齐到 32 的倍数。logits 先取指数得到概率，六个视图平均后再 argmax。DEBF 的测试时增强实现相同，尺度同样是 0.75、1.0 和 1.25。")

    h(2, "4 测试集结果")
    p("下表的前三行是 16 类协议，最后一行 DEBF-B5 是 15 类协议。Table 5 的数字引自 MUSeg 论文，不是本次重跑。超过 CMX 的格子只有 SelectiveMine 两条结果的平均像素准确率和平均交并比。像素准确率没有超过。")
    t(
        "表 1  FTransUNet 测试集与 MUSeg Table 5。PA、MPA、mIoU 的单位是百分比。",
        ["方法", "协议", "分辨率", "TTA", "PA", "MPA", "mIoU"],
        [
            ["CMX (MiT-B2, RGB-D HHA)", "论文 Table 5", "—", "—", "86.30", "72.72", "61.83"],
            ["DFormer", "论文 Table 5", "—", "—", "85.63", "70.37", "59.74"],
            ["SA-Gate", "论文 Table 5", "—", "—", "84.77", "71.34", "59.38"],
            ["CMNeXt", "论文 Table 5", "—", "—", "84.05", "66.77", "56.06"],
            ["SegFormer RGB-D", "论文 Table 5", "—", "—", "83.00", "65.25", "55.20"],
            ["SegFormer 深度 HHA", "论文 Table 5", "—", "—", "81.16", "62.21", "51.28"],
            ["SegFormer RGB", "论文 Table 5", "—", "—", "78.75", "58.54", "47.93"],
            ["DeeplabV3+ RGB", "论文 Table 5", "—", "—", "76.67", "54.91", "43.64"],
            ["SelectiveMine-F", "16 类", "768×896", "六视图", cell16(ftrans, "pa_16"), cell16(ftrans, "mpa_16"), cell16(ftrans, "miou_16")],
            ["SelectiveMine-C", "16 类", "640×768", "六视图", cell16(comp, "pa_16"), cell16(comp, "mpa_16"), cell16(comp, "miou_16")],
            ["SelectiveMine-F", "16 类", f"{single.get('height')}×{single.get('width')}", "关闭", cell16(single, "pa_16"), cell16(single, "mpa_16"), cell16(single, "miou_16")],
            ["DEBF-B5 第 14 轮", "15 类", f"{debf['height']}×{debf['width']}", "六视图", fmt(debf["pa_percent"]), fmt(debf["mpa_percent"]), fmt(debf["miou_percent"])],
        ],
    )
    p(
        f"同一检查点的 15 类测试数字如下，只作协议对照。SelectiveMine-F 加测试时增强：像素准确率 {cell16(ftrans,'holdout_pa')}%、平均像素准确率 {cell16(ftrans,'holdout_mpa')}%、平均交并比 {cell16(ftrans,'holdout_miou')}%。SelectiveMine-C：{cell16(comp,'holdout_pa')}%、{cell16(comp,'holdout_mpa')}%、{cell16(comp,'holdout_miou')}%。关闭测试时增强的 SelectiveMine-F：{cell16(single,'holdout_pa')}%、{cell16(single,'holdout_mpa')}%、{cell16(single,'holdout_miou')}%。15 类像素准确率低于 80%，若拿去和 86.30% 比较会得到错误结论。"
    )
    p("测试时增强对 SelectiveMine-F 的 16 类平均交并比从 62.42% 提到 62.61%，像素准确率从 85.93% 提到 86.04%，平均像素准确率从 75.46% 提到 75.73%。增益存在，但补不齐相对 CMX 的像素准确率缺口。SelectiveMine-C 的平均像素准确率是本组测试里最高的 76.64%，平均交并比 62.41%，略低于不加补全前端、但使用更高分辨率的 SelectiveMine-F。在分辨率不同的前提下，不能把这 0.20 个百分点的平均交并比差异解释成补全模块有害或有益。")

    h(2, "5 类别与矿井")
    p("表 2 是 15 个前景类的交并比，单位百分比。SelectiveMine 的列表对应标签 1 至 15，不包含作为第 16 类的标签 0，因此这些交并比的平均是 15 类平均交并比，不是表 1 的 16 类平均交并比。支持像素数随分辨率变化：768×896 上人员类约 58.1 万像素，640×768 上约 41.5 万，面积比与分辨率比一致。DEBF 的支持数与 768×896 的 SelectiveMine 同量级。")
    t("表 2  测试集前景类交并比（%）", fg_headers, fg_rows)
    p("三条模型都在容器、工具材料和门上偏弱，容器交并比分别是 37.23%、37.46% 和 30.89%。SelectiveMine 的锚固设备、轨道区域和指示装置高于 78%。DEBF-B5 的人员类只有 27.33%，工具材料 20.42%，是它和 SelectiveMine 差距的主要来源。补全前端相对高分辨率 SelectiveMine，在人员、电气设备、电子设备、采掘设备、支护设备和门上更高，在线缆、管路、救援设备和锚固设备上更低。这组差异同时包含模块和分辨率。")
    p("表 3 只来自 DEBF-B5 的 15 类测试时增强评测，按矿井编号聚合。矿井 05 的平均交并比 35.11% 明显低于矿井 02 的 60.08%。SelectiveMine 的按矿井分解没有写进本次测试 JSON，本文不补。")
    t("表 3  DEBF-B5 按矿井的 15 类测试指标（%）", ["矿井", "PA", "MPA", "mIoU"], mine_rows)

    h(2, "6 验证集上的像素准确率微调")
    p("selective_pa2 从 selective_ftrans 的 best.pth 继续训练 12 轮，批大小 4，768×896，学习率 1e-5，稀有类裁剪 0.45，Lovász 权重 0.25，种子 2026。选择规则写在 config.json：只有 16 类平均交并比不低于 0.610 且平均像素准确率不低于 0.728 时，才用 16 类像素准确率作为保存分数，否则分数为 -1，不保存。运行目录在结束后只有 config.json 和 history.jsonl。")
    t(
        "表 4  selective_pa2 验证集轨迹。16 类指标与 15 类平均交并比的单位是百分比。",
        ["轮次", "损失", "mIoU-16", "MPA-16", "PA-16", "mIoU-15", "秒"],
        pa2_table,
    )
    p(
        f"第 {best_pa_row['epoch']} 轮验证像素准确率最高，为 {fmt(pct(best_pa_row['pa_16']))}%，但平均像素准确率只有 {fmt(pct(best_pa_row['mpa_16']))}%，低于 72.80 的保存门槛，平均交并比 {fmt(pct(best_pa_row['miou_16']))}% 虽然过了 61.0。平均像素准确率最高的是第 {best_mpa_row['epoch']} 轮的 {fmt(pct(best_mpa_row['mpa_16']))}%，仍未达到 72.80，像素准确率 {fmt(pct(best_mpa_row['pa_16']))}%，平均交并比 {fmt(pct(best_mpa_row['miou_16']))}%。因此这次运行没有可评测的新权重。表 4 全部是验证集，不能写入表 1。"
    )
    p("这次轨迹说明，在已经较高的像素准确率上继续用很小的学习率微调，验证集像素准确率可以到 87% 附近，但平均像素准确率同步掉到 CMX 门槛以下。若放宽门槛去保存第 12 轮，再用该权重做一次测试，才可能知道测试集像素准确率是否跟上。该测试没有发生。用测试集反复试门槛属于对测试集的选择，本文不做。")

    h(2, "7 讨论")
    p("和 Table 5 对齐之后，SelectiveMine 已经在平均像素准确率和平均交并比上超过 CMX，差距大约是 3 到 4 个百分点和 0.6 到 0.8 个百分点。像素准确率差 0.20 到 0.26 个百分点。这个缺口小于一次分辨率或测试时增强带来的像素准确率变化（关闭测试时增强时为 85.93%），但现有已保存检查点的测试结果都没有跨过 86.30%。")
    p("15 类和 16 类的混用会直接改变结论。selective_ftrans 按 15 类保存，丢掉了验证集上 16 类更高的后续轮次。若只报告 15 类平均交并比 65.35%，会显得大幅超过 61.83%，但这不是 Table 5 的协议。本文同时给出两种协议，主结论只用 16 类。")
    p("DEBF-B5 没有形成有竞争力的结果。第 14 轮验证平均交并比约 50.21%，测试 50.35%，继续剩下的 16 轮最多改变若干个百分点，从曲线上看不到通向 61.83% 的趋势。人员、工具材料和容器是短板。深度补全蒸馏到 16 类头的尝试进一步变差，该方向在当前初始化下不值得按同一配方重跑。")
    p("SelectiveMine-C 的设计目标是模态失衡、深度补全和低照度增强。它在自己的验证协议上稳定，测试平均像素准确率最高。由于分辨率低于对照模型，补全模块的净效应仍然未知。要做这个消融，需要在 768×896 上用同一选择规则各训一版带前端和不带前端的模型，再各做一次测试。本次没有这组配对实验。")

    h(2, "8 局限")
    p("Table 5 的基线没有在本机重训，比较依赖于原论文报告值及其与 16 类协议的量级一致性，而不是逐像素复现 CMX。捕获组是否在给定划分中泄漏，本文没有重新审计，只保证没有改写这三个列表。selective_pa2 没有权重。DEBF 按矿井的结果不能代表 SelectiveMine。补全前端与分辨率绑在一起。论文没有定性可视化。")

    h(2, "9 结论")
    p(
        f"在 FTransUNet 官方测试列表的 16 类协议下，已保存的 SelectiveMine-F 达到像素准确率 {cell16(ftrans,'pa_16')}%、平均像素准确率 {cell16(ftrans,'mpa_16')}%、平均交并比 {cell16(ftrans,'miou_16')}%，SelectiveMine-C 达到 {cell16(comp,'pa_16')}%、{cell16(comp,'mpa_16')}%、{cell16(comp,'miou_16')}%。平均像素准确率和平均交并比超过 MUSeg Table 5 的 CMX，像素准确率没有超过。DEBF-B5 的 15 类测试平均交并比为 {fmt(debf['miou_percent'])}%，低于全部列出的 Table 5 方法。selective_pa2 已结束，验证集像素准确率最高 {fmt(pct(best_pa_row['pa_16']))}%，因为平均像素准确率门槛没有写入新检查点。三项指标在测试集上同时超过 CMX 这一目标，以现有实测为准，尚未达到。"
    )

    h(2, "10 记录位置")
    p("机器可读汇总：/root/autodl-tmp/MineSeg/v3/paper/results.json。训练配置抄录在该文件的 configs 字段。各次评测文件如下。")
    p("SelectiveMine-F 测试：/root/autodl-tmp/MineSeg/v3/runs/selective_ftrans_test/test_tta.json")
    p("SelectiveMine-C 测试：/root/autodl-tmp/MineSeg/v3/runs/selective_complement/test_tta.json")
    p("单尺度对照：/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans_test_single/official_test.json")
    p("DEBF-B5 测试：/root/autodl-tmp/MineSeg/v2/runs/debf_b5_ftrans/test_ftrans.json")
    p("selective_pa2 配置与轨迹：/root/autodl-tmp/MineSeg/v2/runs/selective_pa2/")

    write_md(blocks)
    write_tex(blocks)
    write_docx(blocks)
    print("wrote", ROOT)


def write_md(blocks):
    lines = []
    for block in blocks:
        if block[0] == "h":
            lines.append("#" * block[1] + " " + block[2])
            lines.append("")
        elif block[0] == "p":
            lines.append(block[1])
            lines.append("")
        else:
            _, caption, headers, rows = block
            lines.append(caption)
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    (ROOT / "paper.md").write_text("\n".join(lines).rstrip() + "\n")


def tex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_tex(blocks):
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
        r"\usepackage{hyperref}",
        r"\usepackage{setspace}",
        r"\setstretch{1.15}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\begin{document}",
    ]
    for block in blocks:
        if block[0] == "h":
            level, text = block[1], tex_escape(block[2])
            if level == 1:
                parts.append(r"{\centering\LARGE " + text + r"\par}")
                parts.append(r"\vspace{0.8em}")
            elif level == 2:
                parts.append(r"\section*{" + text + "}")
            else:
                parts.append(r"\subsection*{" + text + "}")
        elif block[0] == "p":
            text = tex_escape(block[1]).replace("/", "/\\allowbreak ").replace(r"\_", r"\_\allowbreak ")
            parts.append(text)
            parts.append("")
        else:
            _, caption, headers, rows = block
            parts.append(r"\begin{small}")
            parts.append(tex_escape(caption))
            parts.append(r"\end{small}")
            if headers and headers[0] == "方法":
                spec = "p{4.6cm}" + "c" * (len(headers) - 1)
            elif len(headers) > 4:
                spec = "c" * len(headers)
            else:
                spec = "l" + "c" * (len(headers) - 1)
            if len(headers) == 4 and len(headers[0]) > 4:
                spec = "p{5.6cm}ccc"
            parts.append(r"{\small")
            parts.append(r"\begin{longtable}{" + spec + "}")
            parts.append(r"\toprule")
            parts.append(" & ".join(tex_escape(cell) for cell in headers) + r" \\")
            parts.append(r"\midrule")
            parts.append(r"\endhead")
            for row in rows:
                parts.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
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
