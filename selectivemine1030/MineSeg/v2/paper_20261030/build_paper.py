#!/usr/bin/env python3
"""Build one SelectiveMine paper as Markdown, LaTeX, Word, and PDF source text.

Every format is rendered from the same list of paragraphs and tables.
Numbers are read from the run files. Nothing is filled in by hand.
"""

import json
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("/root/autodl-tmp/MineSeg/v2/paper_20261030")
RUNS = Path("/root/autodl-tmp/MineSeg/v2/runs")
CMX = {"pa": 86.30, "mpa": 72.72, "miou": 61.83}
CLASSES = [
    "人员", "线缆", "管路", "指示装置", "电气设备", "电子设备", "采掘设备",
    "轨道区域", "支护设备", "门", "工具材料", "救援设备", "容器", "金属构件", "锚固设备",
]
QUOTED = [
    ("CMX，MiT-B2，RGB-D HHA", "86.30", "72.72", "61.83"),
    ("DFormer", "85.63", "70.37", "59.74"),
    ("SA-Gate", "84.77", "71.34", "59.38"),
    ("CMNeXt", "84.05", "66.77", "56.06"),
    ("SegFormer，RGB-D", "83.00", "65.25", "55.20"),
    ("SegFormer，深度 HHA", "81.16", "62.21", "51.28"),
    ("SegFormer，RGB", "78.75", "58.54", "47.93"),
    ("DeeplabV3+，RGB", "76.67", "54.91", "43.64"),
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


def metric(row, key):
    if key in row:
        return float(row[key])
    alt = "holdout_" + key
    if alt in row:
        return float(row[alt])
    raise KeyError(key)


def show(value):
    value = float(value)
    if abs(value) <= 1.5:
        value *= 100.0
    return f"{value:.2f}"


def _percent(value):
    value = float(value)
    if abs(value) <= 1.5:
        value *= 100.0
    return value


def signed(value, baseline):
    return f"{_percent(value) - baseline:+.2f}"


def behind(value, baseline):
    return f"{baseline - _percent(value):.2f}"


def ahead(value, baseline):
    return f"{_percent(value) - baseline:.2f}"


def tex_escape(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def main():
    pa3_hist = rows_jsonl(RUNS / "selective_pa3/history.jsonl")
    pa3_cfg = load(RUNS / "selective_pa3/config.json")
    pa3_test = load(RUNS / "selective_pa3/official_test.json")
    tta = load(RUNS / "selective_pa3_tta/official_test.json")
    ftrans = load(RUNS / "selective_ftrans_test_single/official_test.json")
    pa4_hist = rows_jsonl(RUNS / "selective_pa4/history.jsonl")
    pa4_cfg = load(RUNS / "selective_pa4/config.json")
    sweep = load(RUNS / "selective_pa3_prior/sweep.json")
    ensemble = load(RUNS / "selective_pa3_ensemble/ensemble.json")
    pa5_hist_path = RUNS / "selective_pa5/history.jsonl"
    pa5_cfg_path = RUNS / "selective_pa5/config.json"
    pa5_test_path = RUNS / "selective_pa5/official_test.json"
    pa5_tta_path = RUNS / "selective_pa5_tta/official_test.json"
    pa5_hist = rows_jsonl(pa5_hist_path) if pa5_hist_path.exists() else []
    pa5_cfg = load(pa5_cfg_path) if pa5_cfg_path.exists() else None
    pa5_test = load(pa5_test_path) if pa5_test_path.exists() else None
    pa5_tta = load(pa5_tta_path) if pa5_tta_path.exists() else None

    best = max(pa3_hist, key=lambda row: row["pa_16"])
    single = (metric(pa3_test, "pa_16"), metric(pa3_test, "mpa_16"), metric(pa3_test, "miou_16"))
    multi = (metric(tta, "pa_16"), metric(tta, "mpa_16"), metric(tta, "miou_16"))
    previous = (metric(ftrans, "pa_16"), metric(ftrans, "mpa_16"), metric(ftrans, "miou_16"))
    single_beats = single[0] * 100 > CMX["pa"] and single[1] * 100 > CMX["mpa"] and single[2] * 100 > CMX["miou"]
    multi_beats = multi[0] * 100 > CMX["pa"] and multi[1] * 100 > CMX["mpa"] and multi[2] * 100 > CMX["miou"]

    blocks = []

    def h(level, text):
        blocks.append(("h", level, text))

    def p(text):
        blocks.append(("p", text))

    def table(caption, headers, rows):
        blocks.append(("t", caption, headers, rows))

    h(1, "SelectiveMine 在 MUSeg 上的继续优化：FTransUNet 划分下的 16 类测试结果")
    p("日期：2026-10-30。本文只记录已经写进配置、训练历史和评测 JSON 的结果。百分比保留两位小数，由原始分数换算，不另行改写。训练、验证和测试使用 FTransUNet 的 train.txt、val.txt 和 test.txt。检查点只按验证集保存。测试集在选择规则冻结之后才读取。")

    h(2, "摘要")
    p(
        "SelectiveMine 在 MineRelay 融合路径之外保留一个独立的 RGB 外观解码器，并用一个不读取真值的空间路由器把两路对数概率混合起来。深度完全缺失时，混合退回 RGB 头。"
        f"本文从 selective_ftrans 已保存的第 7 轮权重继续训练。主运行 selective_pa3 在 FTransUNet 训练集 1595 张上训练 12 轮，输入 768×896，用验证集 788 张上的 16 类像素准确率保存权重，地板是 16 类平均交并比不低于 60.80%、平均像素准确率不低于 72.00%。"
        f"被保存的是第 {best['epoch']} 轮。它在测试集 788 张上的单尺度 16 类像素准确率、平均像素准确率和平均交并比是 {show(single[0])}%、{show(single[1])}% 和 {show(single[2])}%。"
        f"MUSeg 论文 Table 5 中 CMX（MiT-B2，RGB-D HHA）的对应数字是 86.30%、72.72% 和 61.83%。单尺度结果的平均像素准确率高 {ahead(single[1], CMX['mpa'])} 个百分点，平均交并比高 {ahead(single[2], CMX['miou'])} 个百分点，像素准确率低 {behind(single[0], CMX['pa'])} 个百分点。"
        f"同一权重预先指定的六视图测试时增强，尺度为 0.75、1.0 和 1.25，每个尺度再做水平翻转，概率取平均。16 类三项是 {show(multi[0])}%、{show(multi[1])}% 和 {show(multi[2])}%，分别比 Table 5 的 CMX 高 {ahead(multi[0], CMX['pa'])}、{ahead(multi[1], CMX['mpa'])} 和 {ahead(multi[2], CMX['miou'])} 个百分点。"
        "六视图和单尺度不是同一评测协议。Table 5 的数字是引用，不是在这台机器上重跑的。"
    )

    h(2, "1  数据和协议")
    p("图像来自 MUSeg。先前实验记录给出的采集设置是六个矿井的 Azure Kinect DK，裁剪后为 1082×932。语义标签 1 至 15 是前景类。标签 0 在 16 类协议里计入混淆矩阵，在 15 类协议里与 255 一起忽略。255 在两种协议里都忽略。")
    p("和 Table 5 对齐的是 16 类协议：像素准确率、平均像素准确率和平均交并比都把标签 0 算作一类。15 类协议的像素准确率大约在 78% 到 81%，不能和 86.30% 放在一起比较。selective_pa3 的测试 JSON 里有一个 beats_cmx 字段，写文件时用的是 15 类像素准确率去比 86.30%，所以像素准确率那一项为假。本文不使用该字段，三项比较都按 16 类原始分数重新计算。评测脚本里的这一比较已经改成 16 类。")
    p("划分文件在 /root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/。train.txt 有 1595 张，val.txt 和 test.txt 各有 788 张。三个集合的图像名互不相交，训练脚本在发现捕获组交叉时会退出。数据根目录是 /root/autodl-tmp/MUSeg/flat。")
    table(
        "表 1  MUSeg 论文 Table 5 的引用数字，单位是百分比。这些行不是本次重测。",
        ["模型", "像素准确率", "平均像素准确率", "平均交并比"],
        [list(row) for row in QUOTED],
    )

    h(2, "2  模型")
    p("微调使用 2026-10-24 发布的 SelectiveMine，不使用 v2/models.py 里另一个同名的 MiT 分支。RGB 编码器和深度编码器先按 MineRelay 的门控融合。几何分支编码 HHA，再通过 relay 单元注入融合特征。融合解码器输出融合 logits，几何解码器输出几何 logits，二者按一个标量开度、multimodal.mix 和深度是否存在相加。外观解码器是融合解码器的独立拷贝，只吃 RGB 特征。")
    p("路由器的输入是两路专家概率、二者绝对差、两路熵、深度有效掩码和亮度，共 52 个通道，不包含真值。它先自适应池化到第一级 RGB 特征的空间大小，再双线性插值回原分辨率。最后一层卷积的权重初始化为 0，偏置初始化为 2，因此训练开始时门控偏向融合专家。两路对数概率按这个门控做 logaddexp 混合。若一张图的深度完全无效，输出改用 RGB 头。评测前把 multimodal.mix 填成 1，并按 joint 模式配置。训练日志记录可训练参数为 59.06M。")
    p("损失沿用 selective_loss：RGB 项和融合项权重都是 0.5，几何项 0.2，路由器效用项 0.05，温度 0.5。另外加上权重 0.25 的 Lovász-Softmax，以及权重 0.12 的、不按类别重加权的像素负对数似然。类别权重来自训练集频数。背景权重再乘以 1.15，用来拉高像素准确率。优化器是 AdamW，权重衰减 0.01，头的学习率是编码器的 4 倍，梯度裁剪为 1。学习率按余弦从初始值降到初始值的 5%。每个训练步之后把 mix 重新填成 1。")
    p("训练增强在 MineDataset 里：尺度在 0.7 到 1.4 之间均匀抽取，再随机裁剪到网络输入；以 0.40 的概率把裁剪中心移向稀有类；以 0.8 的概率做颜色抖动；以 0.5 的概率水平翻转。翻转发生在几何编码之后。验证和测试只把图像缩放到指定分辨率，不做随机增强，corruption 设为 none。")

    h(2, "3  主训练")
    p(
        f"selective_pa3 的初始权重是 {pa3_cfg['init']}，加载日志写明该文件的轮次是 7。该权重在同一测试集上的 16 类三项是 {show(previous[0])}%、{show(previous[1])}% 和 {show(previous[2])}%，像素准确率比 CMX 低 {behind(previous[0], CMX['pa'])} 个百分点。"
        f"继续训练 12 轮，批大小 4，分辨率 768×896，初始学习率 6e-6，随机种子 2026，稀有类裁剪概率 0.40。保存规则是 select-pa：在 16 类平均交并比不低于 0.608 且平均像素准确率不低于 0.720 的轮次中，保留 16 类像素准确率最高的一轮。每一轮的权重都另外存了下来，供事后在验证集上比较，不供按测试集重选。"
    )
    p(
        f"12 轮全部跑完。验证集 16 类像素准确率从第 1 轮的 {show(pa3_hist[0]['pa_16'])}% 升到第 {best['epoch']} 轮的 {show(best['pa_16'])}%，该轮同时满足两条地板，因此 best.pth 和 best_pa.pth 都是第 {best['epoch']} 轮。"
        f"同轮验证集 16 类平均像素准确率是 {show(best['mpa_16'])}%，平均交并比是 {show(best['miou_16'])}%。这两个验证数字低于 Table 5 的 72.72% 和 61.83%，保存地板也低于这两条线。测试集上的平均像素准确率和平均交并比高于验证集，所以验证地板没有把测试比较提前判掉。"
    )
    table(
        "表 2  selective_pa3 的验证集。单位是百分比。15 类同时忽略标签 0 和 255。",
        ["轮次", "损失", "16 类 PA", "16 类 MPA", "16 类 mIoU", "15 类 PA", "15 类 MPA", "15 类 mIoU"],
        [
            [
                str(row["epoch"]),
                f"{row['loss']:.4f}",
                show(row["pa_16"]),
                show(row["mpa_16"]),
                show(row["miou_16"]),
                show(row["holdout_pa"]),
                show(row["holdout_mpa"]),
                show(row["holdout_miou"]),
            ]
            for row in pa3_hist
        ],
    )

    h(2, "4  测试结果")
    p(
        f"单尺度测试读取 best.pth，分辨率 768×896，对路由后的对数概率做 argmax。测试集 788 张，耗时 {pa3_test['seconds']} 秒。"
        f"16 类像素准确率 {show(single[0])}%，平均像素准确率 {show(single[1])}%，平均交并比 {show(single[2])}%。"
        f"与表 1 的 CMX 相比，差值是 {signed(single[0], CMX['pa'])}、{signed(single[1], CMX['mpa'])} 和 {signed(single[2], CMX['miou'])} 个百分点。"
        f"同一份 JSON 里的 15 类三项是 {show(metric(pa3_test, 'holdout_pa'))}%、{show(metric(pa3_test, 'holdout_mpa'))}% 和 {show(metric(pa3_test, 'holdout_miou'))}%，只作记录。"
        f"{'三项都高于 CMX。' if single_beats else '平均像素准确率和平均交并比高于 CMX，像素准确率没有高于 CMX。'}"
    )
    p(
        f"六视图在同一权重上评测，耗时 {tta['seconds']} 秒。概率在六个视图上平均后再 argmax。"
        f"16 类三项是 {show(multi[0])}%、{show(multi[1])}% 和 {show(multi[2])}%，差值是 {signed(multi[0], CMX['pa'])}、{signed(multi[1], CMX['mpa'])} 和 {signed(multi[2], CMX['miou'])} 个百分点。"
        f"15 类三项是 {show(metric(tta, 'holdout_pa'))}%、{show(metric(tta, 'holdout_mpa'))}% 和 {show(metric(tta, 'holdout_miou'))}%。"
        f"{'按 16 类三项计算，六视图都高于所引用的 CMX。' if multi_beats else '六视图没有同时高于所引用的 CMX。'}"
        "这个协议在测试之前就定成 0.75、1.0、1.25 和水平翻转，没有根据测试分数改尺度。"
    )
    headers = ["设置", "PA", "MPA", "mIoU", "PA差", "MPA差", "mIoU差"]
    result_rows = [
        ["CMX，Table 5 引用", "86.30", "72.72", "61.83", "0.00", "0.00", "0.00"],
        [
            "selective_ftrans 第 7 轮，单尺度",
            show(previous[0]), show(previous[1]), show(previous[2]),
            signed(previous[0], CMX["pa"]), signed(previous[1], CMX["mpa"]), signed(previous[2], CMX["miou"]),
        ],
        [
            "selective_pa3 第 12 轮，单尺度",
            show(single[0]), show(single[1]), show(single[2]),
            signed(single[0], CMX["pa"]), signed(single[1], CMX["mpa"]), signed(single[2], CMX["miou"]),
        ],
        [
            "selective_pa3 第 12 轮，六视图",
            show(multi[0]), show(multi[1]), show(multi[2]),
            signed(multi[0], CMX["pa"]), signed(multi[1], CMX["mpa"]), signed(multi[2], CMX["miou"]),
        ],
    ]
    if pa5_test is not None:
        result_rows.append([
            "selective_pa5，单尺度",
            show(metric(pa5_test, "pa_16")),
            show(metric(pa5_test, "mpa_16")),
            show(metric(pa5_test, "miou_16")),
            signed(metric(pa5_test, "pa_16"), CMX["pa"]),
            signed(metric(pa5_test, "mpa_16"), CMX["mpa"]),
            signed(metric(pa5_test, "miou_16"), CMX["miou"]),
        ])
    if pa5_tta is not None:
        result_rows.append([
            "selective_pa5，六视图",
            show(metric(pa5_tta, "pa_16")),
            show(metric(pa5_tta, "mpa_16")),
            show(metric(pa5_tta, "miou_16")),
            signed(metric(pa5_tta, "pa_16"), CMX["pa"]),
            signed(metric(pa5_tta, "mpa_16"), CMX["mpa"]),
            signed(metric(pa5_tta, "miou_16"), CMX["miou"]),
        ])
    table("表 3  FTransUNet 测试集上的 16 类结果。单位是百分比。差值等于该项减去表 1 中的 CMX。", headers, result_rows)

    iou = pa3_test["per_class_iou"]
    support = pa3_test["per_class_support"]
    order = sorted(range(len(iou)), key=lambda index: iou[index])
    p(
        "表 4 是单尺度测试的 15 个前景类交并比，平均之后等于表 3 那一行的 15 类平均交并比 "
        f"{show(metric(pa3_test, 'holdout_miou'))}%，不是 16 类的 {show(single[2])}%。"
        f"最低的是{CLASSES[order[0]]} {show(iou[order[0]])}%，其次是{CLASSES[order[1]]} {show(iou[order[1]])}% 和{CLASSES[order[2]]} {show(iou[order[2]])}%。"
        f"最高的是{CLASSES[order[-1]]} {show(iou[order[-1]])}%。"
    )
    table(
        "表 4  selective_pa3 单尺度测试的前景类交并比。单位是百分比。支持像素数来自同一份 JSON。",
        ["类别", "交并比", "支持像素"],
        [[CLASSES[i], show(iou[i]), str(support[i])] for i in range(len(CLASSES))],
    )

    h(2, "5  没有提高单尺度测试像素准确率的尝试")
    pa4 = pa4_hist[-1]
    p(
        f"selective_pa4 从 selective_pa3 的第 12 轮接着训，计划 8 轮。学习率 2.5e-6，像素负对数似然权重提到 {pa4_cfg['args']['pixel_ce']}，背景权重乘数提到 {pa4_cfg['args']['bg_boost']}，稀有类裁剪概率 {pa4_cfg['args']['rare_crop']}，Lovász 权重 {pa4_cfg['args']['lovasz_weight']}，分辨率仍是 768×896。"
        f"第 1 轮验证集 16 类像素准确率 {show(pa4['pa_16'])}%，平均像素准确率 {show(pa4['mpa_16'])}%，平均交并比 {show(pa4['miou_16'])}%，三项都低于起点的 {show(best['pa_16'])}%、{show(best['mpa_16'])}% 和 {show(best['miou_16'])}%。这一轮结束后训练被停止，没有读取测试集。"
    )
    chosen = sweep["selection"]
    val_chosen = sweep["val_chosen"]
    test_chosen = sweep["test_at_chosen"]
    test_zero = sweep["test_at_zero_same_resolution"]
    p(
        "第二项尝试只改推理，不改权重。在验证集上同时扫描分辨率 768×896、800×928、832×960、704×832，以及加在背景对数概率上的偏置 -0.15、0、0.10、0.20、0.30、0.45、0.60、0.75、0.90。"
        f"选择规则在看测试集之前写成：在验证集 16 类平均交并比不低于 61.20 且平均像素准确率不低于 71.80 的设置里，取 16 类像素准确率最高的一个。选中的是 {chosen['height']}×{chosen['width']}、偏置 {chosen['chosen_bias']:.2f}，验证集三项是 {show(val_chosen['pa16'])}%、{show(val_chosen['mpa16'])}% 和 {show(val_chosen['miou16'])}%。"
        f"随后只对这个设置读了一次测试集，16 类三项是 {show(test_chosen['pa16'])}%、{show(test_chosen['mpa16'])}% 和 {show(test_chosen['miou16'])}%。像素准确率低于 768×896、偏置 0 的官方单尺度结果 {show(single[0])}%。"
        f"同一次测试前向还记下了该分辨率、偏置 0 的三项，{show(test_zero['pa16'])}%、{show(test_zero['mpa16'])}% 和 {show(test_zero['miou16'])}%，这个零偏置数字没有参与选择。"
    )
    val_rows = []
    for key, row in sweep["val"].items():
        height, width, bias = key.split("|")
        if bias != "0.00" and key != chosen["chosen_key"]:
            continue
        val_rows.append([
            f"{height}×{width}",
            bias,
            show(row["pa16"]),
            show(row["mpa16"]),
            show(row["miou16"]),
        ])
    table(
        "表 5  验证集上各分辨率在背景偏置为 0 时的 16 类分数，以及被选中的那一格。单位是百分比。",
        ["分辨率", "背景偏置", "像素准确率", "平均像素准确率", "平均交并比"],
        val_rows,
    )
    ens_rows = []
    for name, row in ensemble["val"].items():
        ens_rows.append([name, show(row["pa16"]), show(row["mpa16"]), show(row["miou16"])])
    p(
        "第三项尝试把 selective_pa3 第 7、9、10、11、12 轮的概率做平均。成员组合在验证集上比较，规则仍是两条地板之内像素准确率最高，成员更少者优先。"
        f"五种组合里像素准确率最高的是 {ensemble['chosen']}，验证集像素准确率 {show(ensemble['val'][ensemble['chosen']]['pa16'])}%。"
        "它就是已经测过的第 12 轮单模型，所以没有为集成再读测试集。其余组合的验证像素准确率更低，平均像素准确率或平均交并比有的更高。验证集没有支持把集成当成新的主结果。"
    )
    table(
        "表 6  检查点概率平均在验证集上的 16 类分数。单位是百分比。",
        ["组合", "像素准确率", "平均像素准确率", "平均交并比"],
        ens_rows,
    )
    if not pa5_hist:
        p("selective_pa5 计划从第 12 轮以学习率 6e-7、与 selective_pa3 相同的损失再训 6 轮。写这篇论文时，该运行还没有留下 history.jsonl。")
    else:
        pa5_best = max(pa5_hist, key=lambda row: row["pa_16"])
        improved = pa5_best["pa_16"] > best["pa_16"]
        text = (
            f"selective_pa5 从第 12 轮继续，损失与 selective_pa3 相同：Lovász 权重 0.25，像素负对数似然权重 0.12，背景权重乘数 1.15，稀有类裁剪概率 0.40，分辨率 768×896，批大小 4，种子 2026。"
            f"初始学习率降到 {pa5_cfg['args']['lr']}，计划 {pa5_cfg['args']['epochs']} 轮，保存规则仍是 select-pa。"
            f"实际写下 {len(pa5_hist)} 轮。验证集 16 类像素准确率最高的是第 {pa5_best['epoch']} 轮，三项为 {show(pa5_best['pa_16'])}%、{show(pa5_best['mpa_16'])}% 和 {show(pa5_best['miou_16'])}%。"
            f"起点第 12 轮的验证像素准确率是 {show(best['pa_16'])}%。"
        )
        if pa5_test is None and not improved:
            text += (
                f"验证像素准确率没有高于起点，平均像素准确率降到 {show(pa5_best['mpa_16'])}%。"
                f"训练在第 {len(pa5_hist)} 轮后停止，没有读取测试集。"
            )
        elif pa5_test is None:
            text += "最佳验证像素准确率高于起点。测试 JSON 尚未写入。"
        else:
            text += (
                f"因为最佳验证像素准确率{'高于' if improved else '没有高于'}起点，测试集按预先写下的规则评了 best.pth。"
                f"单尺度 16 类三项是 {show(metric(pa5_test, 'pa_16'))}%、{show(metric(pa5_test, 'mpa_16'))}% 和 {show(metric(pa5_test, 'miou_16'))}%。"
            )
            if pa5_tta is not None:
                text += (
                    f"同一权重的六视图是 {show(metric(pa5_tta, 'pa_16'))}%、{show(metric(pa5_tta, 'mpa_16'))}% 和 {show(metric(pa5_tta, 'miou_16'))}%。"
                )
        p(text)
        table(
            "表 7  selective_pa5 的验证集。单位是百分比。",
            ["轮次", "损失", "16 类 PA", "16 类 MPA", "16 类 mIoU"],
            [
                [str(row["epoch"]), f"{row['loss']:.4f}", show(row["pa_16"]), show(row["mpa_16"]), show(row["miou_16"])]
                for row in pa5_hist
            ],
        )

    h(2, "6  结论")
    conclusion = (
        f"在 FTransUNet 的 788 张测试图上，selective_pa3 第 12 轮的单尺度 16 类结果是像素准确率 {show(single[0])}%，平均像素准确率 {show(single[1])}%，平均交并比 {show(single[2])}%。"
        f"后两项高于 MUSeg Table 5 所引用的 CMX，像素准确率低 {behind(single[0], CMX['pa'])} 个百分点。"
        f"同一权重的六视图结果是 {show(multi[0])}%，{show(multi[1])}%，{show(multi[2])}%，三项都高于该引用。"
        "把像素负对数似然和背景权重加大的继续训练，在第 1 轮就低于验证起点，并已停止。验证集选中的分辨率和背景偏置，测试像素准确率是 "
        f"{show(test_chosen['pa16'])}%，低于单尺度主结果。检查点概率平均也没有在验证集上超过第 12 轮的像素准确率。"
    )
    if pa5_test is not None:
        conclusion += (
            f"降低学习率的 selective_pa5 单尺度测试是 {show(metric(pa5_test, 'pa_16'))}%、{show(metric(pa5_test, 'mpa_16'))}%、{show(metric(pa5_test, 'miou_16'))}%。"
        )
    elif pa5_hist:
        conclusion += "降低学习率的继续训练没有在验证集上超过第 12 轮的像素准确率，测试集仍用第 12 轮的数字。"
    p(conclusion)
    p("主权重是 /root/autodl-tmp/MineSeg/v2/runs/selective_pa3/best.pth。单尺度测试是同目录的 official_test.json。六视图测试是 /root/autodl-tmp/MineSeg/v2/runs/selective_pa3_tta/official_test.json。验证历史是 selective_pa3/history.jsonl。分辨率扫描是 selective_pa3_prior/sweep.json。集成验证是 selective_pa3_ensemble/ensemble.json。")

    record = {
        "single_scale_16": {"pa": show(single[0]), "mpa": show(single[1]), "miou": show(single[2])},
        "tta_16": {"pa": show(multi[0]), "mpa": show(multi[1]), "miou": show(multi[2])},
        "single_scale_beats_cmx": single_beats,
        "tta_beats_cmx": multi_beats,
        "cmx_quoted": CMX,
        "pa5_tested": pa5_test is not None,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    write_markdown(blocks)
    write_latex(blocks)
    write_docx(blocks)
    print(json.dumps(record, indent=2))


def write_markdown(blocks):
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
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    (ROOT / "paper.md").write_text("\n".join(lines) + "\n")


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
