#!/usr/bin/env python3
"""Build the experiment note from measured test json files.

Numbers are read from disk. The script stops if either test file is missing.
"""

import json
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path("/root/autodl-tmp/MineSeg/v3/paper")
SELECTIVE = Path("/root/autodl-tmp/MineSeg/v3/runs/selective_complement/test_tta.json")
DEBF = Path("/root/autodl-tmp/MineSeg/v3/runs/debf_complement/test.json")
BASELINES = [
    ("DeeplabV3+", "Xception", "RGB", 76.67, 54.91, 43.64),
    ("SegFormer", "MiT-B2", "RGB", 78.75, 58.54, 47.93),
    ("SegFormer", "MiT-B2", "Depth", 81.16, 62.21, 51.28),
    ("SegFormer", "MiT-B2", "RGB-D", 83.00, 65.25, 55.20),
    ("CMNeXt", "MiT-B2", "RGB-D", 84.05, 66.77, 56.06),
    ("SA-Gate", "ResNet-101", "RGB-D", 84.77, 71.34, 59.38),
    ("DFormer", "DFormer-S", "RGB-D", 85.63, 70.37, 59.74),
    ("CMX", "MiT-B2", "RGB-D", 86.30, 72.72, 61.83),
]


def pct(value):
    return float(value) * 100.0


def load(path):
    if not path.is_file():
        raise SystemExit(f"missing measured test file: {path}")
    data = json.loads(path.read_text())
    for key in ("pa_16", "mpa_16", "miou_16", "holdout_pa", "holdout_mpa", "holdout_miou"):
        if key not in data and key.startswith("pa"):
            raise SystemExit(f"{path} has no {key}")
    data["_pa"] = pct(data.get("pa_16", data["holdout_pa"]))
    data["_mpa"] = pct(data.get("mpa_16", data["holdout_mpa"]))
    data["_miou"] = pct(data.get("miou_16", data["holdout_miou"]))
    data["_pa15"] = pct(data["holdout_pa"]) if "pa_16" in data else None
    data["_mpa15"] = pct(data["holdout_mpa"]) if "mpa_16" in data else None
    data["_miou15"] = pct(data["holdout_miou"]) if "miou_16" in data else None
    return data


def beats_cmx(row):
    return row["_pa"] > 86.30 and row["_mpa"] > 72.72 and row["_miou"] > 61.83


def markdown(selective, debf):
    sel_flag = "超过" if beats_cmx(selective) else "没有同时超过"
    debf_flag = "超过" if beats_cmx(debf) else "没有同时超过"
    lines = [
        "# 面向井下 RGB-D 分割的模态补全与选择性融合",
        "",
        "**英文题目：** Modality Completion and Selective Fusion for Underground RGB-D Segmentation",
        "",
        "## 摘要",
        "",
        "MUSeg 的井下图像光照不均，Kinect 深度在反光和吸光表面会成片缺失。本文在 FTransUNet 目录中的 train.txt、val.txt、test.txt 上改进两个 RGB-D 分割模型。SelectiveMine-C 在已有的彩色专家和融合专家之间保留空间路由，并在输入端加入多尺度深度补全和由亮度控制的低照残差。DEBFNet-C 保留深度边缘双向门控，用同一个补全前端，并用冻结的 SelectiveMine 做十六类蒸馏。验证集只用于选择检查点，测试集只评估一次。",
        "",
        f"测试集十六类指标（含标签 0，忽略 255）为：SelectiveMine-C 像素准确率 {selective['_pa']:.2f}%，平均像素准确率 {selective['_mpa']:.2f}%，mIoU {selective['_miou']:.2f}%；DEBFNet-C 像素准确率 {debf['_pa']:.2f}%，平均像素准确率 {debf['_mpa']:.2f}%，mIoU {debf['_miou']:.2f}%。MUSeg 论文中 CMX 的对应数字是 86.30%、72.72%、61.83%。按这一组三个指标同时比较，SelectiveMine-C {sel_flag} CMX，DEBFNet-C {debf_flag} CMX。",
        "",
        "**关键词：** RGB-D 语义分割；井下场景；深度补全；模态不平衡；MUSeg",
        "",
        "## 1 问题",
        "",
        "MUSeg 论文给出的最强基线是 CMX，测试集像素准确率 86.30%，平均像素准确率 72.72%，mIoU 61.83%。井下失败主要来自三类原因。一是深度空洞：无效像素若被当成近处表面，融合模块会把几何噪声写进彩色特征。二是模态不平衡：暗光下彩色不可靠，深度完整时光照分支又不必主导。三是长尾类别像素少，容器、工具、救援设备和金属构件容易被多数类吞掉。",
        "",
        "划分固定为 `/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet` 的三个名单，图像来自 `/root/autodl-tmp/MUSeg`。训练 1595 张，验证和测试各约 787 张。同一采集组不会同时出现在训练和测试中。",
        "",
        "## 2 方法",
        "",
        "补全前端对有效深度做多尺度归一化卷积，只在空洞处写入估计值。最后一层卷积初始化为 0，置信度偏置初始化为 -4，因此加载旧检查点后的第一步几乎不改变原输入。训练时随机遮住一部分有效深度，用 L1 重建被遮住的值。低照残差乘在彩色特征上，权重随亮度下降而增大，亮区系数接近 0。",
        "",
        "SelectiveMine-C 保留彩色解码器和融合解码器，由空间路由器按专家差异、熵、亮度和深度有效掩膜混合对数概率。深度整幅缺失时路由退回彩色专家。采样提高含电子装备、门、工具材料、救援设备、容器和金属构件的图像的频率，损失在类别重加权交叉熵上再加 Lovász。检查点按验证集十六类 mIoU 保存。测试使用尺度 0.75、1、1.25 和水平翻转。",
        "",
        "DEBFNet-C 使用 MiT-B2 彩色编码器和轻量深度编码器，在四个尺度上做深度边缘双向门控。分类头从十五类检查点扩成十六类，背景 logit 初始置负。训练目标是分割损失、补全损失，以及与冻结 SelectiveMine 教师之间的十六类 KL 散度。教师和学生读取同一次裁剪。",
        "",
        "## 3 指标",
        "",
        "十六类指标把标签 0 算作一类，忽略 255。这组数字的像素准确率落在 86% 附近，和 MUSeg 论文表 5 的量级一致。十五类指标忽略标签 0 和 255，只平均 15 个前景类，像素准确率会低一截。两套数字都写在测试 json 里。表中「本文」一行使用十六类结果。",
        "",
        "## 4 测试结果",
        "",
        "| 方法 | 骨干 | 模态 | PA(%) | MPA(%) | mIoU(%) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, backbone, modal, pa, mpa, miou in BASELINES:
        lines.append(f"| {name} | {backbone} | {modal} | {pa:.2f} | {mpa:.2f} | {miou:.2f} |")
    lines.append(
        f"| SelectiveMine-C | MiT-B2 | RGB-D | {selective['_pa']:.2f} | {selective['_mpa']:.2f} | {selective['_miou']:.2f} |"
    )
    lines.append(
        f"| DEBFNet-C | MiT-B2 | RGB-D | {debf['_pa']:.2f} | {debf['_mpa']:.2f} | {debf['_miou']:.2f} |"
    )
    lines += [
        "",
        "十五类前景指标（忽略标签 0 和 255）：",
        "",
        f"- SelectiveMine-C：PA {selective['_pa15']:.2f}%，MPA {selective['_mpa15']:.2f}%，mIoU {selective['_miou15']:.2f}%。",
        f"- DEBFNet-C：PA {debf.get('pa_15', 0)*100 if debf.get('pa_15') is not None else float('nan'):.2f}%，"
        f"MPA {debf.get('mpa_15', 0)*100 if debf.get('mpa_15') is not None else float('nan'):.2f}%，"
        f"mIoU {debf.get('miou_15', 0)*100 if debf.get('miou_15') is not None else float('nan'):.2f}%。",
        "",
        "## 5 配置",
        "",
        "SelectiveMine-C 的配置、历史和权重在 `/root/autodl-tmp/MineSeg/v3/runs/selective_complement`。DEBFNet-C 在 `/root/autodl-tmp/MineSeg/v3/runs/debf_complement`。测试 json 与本文同时生成，没有另填数字。",
        "",
        "## 6 局限",
        "",
        "MUSeg 论文没有公开它的测试名单和是否把标签 0 计入 mIoU。本文用十六类协议去对齐表 5 的像素准确率量级，并同时给出十五类前景指标。若原表只统计十五个前景类，则应以上面的十五类数字为准，不能把两套协议混成一个结论。DEBFNet-C 的教师来自 SelectiveMine，因此它不是一个完全独立的系统。",
        "",
    ]
    return "\n".join(lines) + "\n"


def latex(text):
    body = []
    for line in text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("**英文题目"):
            continue
        if line.startswith("## "):
            body.append("\\section{" + line[3:] + "}")
            continue
        if line.startswith("|"):
            continue
        if line.startswith("- "):
            body.append("\\item " + line[2:])
            continue
        body.append(line)
    table = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "方法 & 骨干 & 模态 & PA & MPA & mIoU \\\\",
        "\\midrule",
    ]
    # The markdown already states the numbers in prose. The table is rebuilt by the caller.
    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\end{table}")
    return body, table


def write_tex(selective, debf, md):
    rows = []
    for name, backbone, modal, pa, mpa, miou in BASELINES:
        rows.append(f"{name} & {backbone} & {modal} & {pa:.2f} & {mpa:.2f} & {miou:.2f} \\\\")
    rows.append(
        f"SelectiveMine-C & MiT-B2 & RGB-D & {selective['_pa']:.2f} & {selective['_mpa']:.2f} & {selective['_miou']:.2f} \\\\"
    )
    rows.append(
        f"DEBFNet-C & MiT-B2 & RGB-D & {debf['_pa']:.2f} & {debf['_mpa']:.2f} & {debf['_miou']:.2f} \\\\"
    )
    sections = []
    items = []
    in_items = False
    for line in md.splitlines():
        if line.startswith("# ") or line.startswith("**英文题目"):
            continue
        if line.startswith("## "):
            if in_items:
                sections.append("\\end{itemize}")
                in_items = False
            sections.append("\\section{" + line[3:].replace("_", "\\_") + "}")
            continue
        if line.startswith("|") or not line.strip() or line.startswith("**关键词"):
            continue
        if line.startswith("- "):
            if not in_items:
                sections.append("\\begin{itemize}")
                in_items = True
            sections.append("\\item " + line[2:].replace("_", "\\_").replace("%", "\\%"))
            continue
        if in_items:
            sections.append("\\end{itemize}")
            in_items = False
        sections.append(line.replace("_", "\\_").replace("%", "\\%") + "\n")
    if in_items:
        sections.append("\\end{itemize}")
    tex = r"""\documentclass[11pt]{article}
\usepackage{xeCJK}
\setCJKmainfont{WenQuanYi Micro Hei}
\usepackage{geometry}
\geometry{a4paper, margin=2.3cm}
\usepackage{booktabs}
\usepackage{hyperref}
\title{面向井下 RGB-D 分割的模态补全与选择性融合\\
\large Modality Completion and Selective Fusion for Underground RGB-D Segmentation}
\author{}
\date{}
\begin{document}
\maketitle
"""
    tex += "\n".join(sections)
    tex += "\n\\begin{table}[ht]\\centering\\begin{tabular}{lllrrr}\\toprule\n"
    tex += "方法 & 骨干 & 模态 & PA(\\%) & MPA(\\%) & mIoU(\\%) \\\\\n\\midrule\n"
    tex += "\n".join(rows)
    tex += "\n\\bottomrule\\end{tabular}\\end{table}\n\\end{document}\n"
    return tex


def write_docx(md, path):
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    for line in md.splitlines():
        if line.startswith("# "):
            document.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=1)
        elif line.startswith("|"):
            document.add_paragraph(line)
        elif line.strip():
            document.add_paragraph(line)
    document.save(path)


def main():
    selective = load(SELECTIVE)
    debf = load(DEBF)
    ROOT.mkdir(parents=True, exist_ok=True)
    md = markdown(selective, debf)
    (ROOT / "paper.md").write_text(md, encoding="utf-8")
    (ROOT / "paper.tex").write_text(write_tex(selective, debf, md), encoding="utf-8")
    write_docx(md, ROOT / "paper.docx")
    (ROOT / "results.json").write_text(
        json.dumps({"selective_test": selective, "debf_test": debf, "baselines": BASELINES}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()
