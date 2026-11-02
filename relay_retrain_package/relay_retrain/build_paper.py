#!/usr/bin/env python3
"""Write the Geometry Relay retrain report in Markdown, LaTeX, Word, and chart form.

PDF is produced by xelatex from the LaTeX file. Every format uses the same blocks.
"""

import json
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    APPEARANCE_IDS,
    CLASS_ZH,
    FROZEN_ABLATION,
    GEOMETRY_IDS,
    GROUP_ORDER,
    GROUP_TITLE,
    PAPER,
    RUN,
    SEEDS,
    VARIANTS,
    judge,
    mean_std,
    rule_text,
    variant_title,
)

RESULT = RUN / "results.json"
PROTOCOL = RUN / "protocol.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value):
    return f"{float(value) * 100:.2f}"


def pm(mean, std):
    return f"{float(mean) * 100:.2f}±{float(std) * 100:.2f}"


def spm(mean, std):
    return f"{float(mean) * 100:+.2f}±{float(std) * 100:.2f}"


def tex_escape(text):
    return (
        str(text).replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def records_by(result):
    found = {}
    for record in result["records"]:
        found[(record["variant"], int(record["seed"]))] = record
    for variant in result["variants"]:
        for seed in SEEDS:
            if (variant, int(seed)) not in found:
                raise SystemExit(f"missing {variant} seed {seed}")
    return found


def metric_pair(found, variant, key):
    values = [found[(variant, int(seed))]["test"][key] for seed in SEEDS]
    return mean_std(values)


def delta_pair(found, variant, key):
    values = [
        found[(variant, int(seed))]["test"][key] - found[("full", int(seed))]["test"][key]
        for seed in SEEDS
    ]
    return values, mean_std(values)


def frozen_sentence():
    if not FROZEN_ABLATION.is_file():
        return "先前的冻结权重消融文件不在原路径，本文不引用它的数字。"
    data = load(FROZEN_ABLATION)
    row = next(item for item in data["test"]["variants"] if item["id"] == "no_relay")
    return (
        f"先前对 selective_pa3 第 {data['epoch']} 轮冻结权重做推理消融时，去掉 Relay 后测试集 16 类像素准确率、"
        f"平均像素准确率和平均交并比的差值都是 {row['delta_pa_16'] * 100:.2f}、{row['delta_mpa_16'] * 100:.2f} 和 "
        f"{row['delta_miou_16'] * 100:.2f} 个百分点。那个结果只说明当前权重和当前推理代码没有使用 Relay 改变输出，"
        "不能说明重新训练之后 Relay 没有作用，也不能说明它有作用。"
    )


def judgement_sentence(name, kind, mu, sd):
    title = variant_title(name)
    common = f"{title}相对完整模型的 16 类 mIoU 差均值是 {mu * 100:+.2f} 个百分点，样本标准差是 {sd * 100:.2f} 个百分点。"
    if kind == "small":
        return common + "按事先写定的阈值，这是差异很小。"
    if kind == "stable_negative":
        return common + "三种子同号且均值绝对值大于标准差，方向稳定为负，即该变体低于完整模型。"
    if kind == "stable_positive":
        return common + "三种子同号且均值绝对值大于标准差，方向稳定为正，即该变体高于完整模型。"
    return common + "它不满足差异很小，也不满足方向稳定，记为不稳定。"


def class_mean(found, variant, ids):
    values = []
    for seed in SEEDS:
        iou = found[(variant, int(seed))]["test"]["per_class_iou_16"]
        support = found[(variant, int(seed))]["test"]["per_class_support_16"]
        chosen = [iou[index] for index in ids if support[index] > 0]
        values.append(sum(chosen) / len(chosen))
    return values, mean_std(values)


def build_blocks(result, protocol, found):
    blocks = []
    h = lambda level, text: blocks.append(("h", level, text))
    p = lambda text: blocks.append(("p", text))
    table = lambda caption, headers, rows: blocks.append(("table", caption, headers, rows))
    fig = lambda path, caption: blocks.append(("fig", path, caption))

    h(1, "SelectiveMine Geometry Relay 的重新训练消融")
    p(
        "本文报告 Geometry Relay 在同一训练预算下的重新训练对照、机制对照、场景分组、参数量、显存、单张耗时和可视化。"
        "主指标是测试集 16 类像素准确率、平均像素准确率和平均交并比。16 类把标签 0 计入，标签 255 忽略。"
        "15 类同时忽略标签 0 和 255，只作对照。百分比由 JSON 中的原始分数换算，均值旁边的数是三种子的样本标准差。"
        + frozen_sentence()
    )
    h(2, "1  协议")
    p(
        f"数据根目录是 /root/autodl-tmp/MUSeg/flat。划分文件是 FTransUNet 的 train.txt、val.txt 和 test.txt，"
        f"图像数分别是 1595、788 和 788，三个名单交集为 0。所有变体从 selective_ftrans 的同一份检查点出发，"
        f"训练 {12} 轮，输入 {768}×{896}，批大小 4，初始学习率 6e-6，余弦调度下限为 0.05 倍。"
        "损失与 selective_pa3 相同：选择性损失的外观、融合、几何和路由器权重是 0.5、0.5、0.2 和 0.05，"
        "另加权重 0.25 的 Lovász，以及权重 0.12 的像素负对数似然。背景类权重乘数是 1.15。稀有类裁剪概率是 0.40。"
        "随机种子是 2026、2027 和 2028。cudnn.benchmark 与原先的微调脚本一样打开，所以同一种子不保证逐位复现，"
        "三种子用来观察方向是否稳定。"
        "发行推理包把融合方式建成 logits，Relay 的输出卷积在 selective_ftrans 第 7 轮里仍是全 0，所以把它接进前向之前不会改变预测。"
        "本次完整模型把融合方式设为 relay，让这块模块真正接受梯度。"
        "为了在批大小仍是 4 时放进 32 GB 显存，反向时重算 RGB 编码器的激活。损失、优化器和选择规则不变。"
        "十八次训练在 RTX 5090 上于 2026-11-02 12:07 全部结束。主流程当时还没有进入测试评估，脚本在循环结束后读到了被改写的内容，"
        "报出 cho: command not found 和 syntax error near unexpected token done，然后退出。12:51 监测发现结果包不存在，发出失败邮件并关机。"
        "恢复时没有重新训练，沿用这十八份只按验证集选定的权重。测试集在恢复之后才第一次计算。恢复评估使用 RTX 4090，推理不反传，批大小仍是 4。"
    )
    p(
        "模型选择只看验证集。规则是 select-pa：在 16 类平均交并比不低于 60.80% 且平均像素准确率不低于 72.00% 的轮次中，"
        "保留 16 类像素准确率最高的一轮。若没有任何一轮同时达到两道地板，则改用验证集 16 类平均交并比最高的一轮，并在选择表中标明。"
        "这个后备规则在测试之前写定，仍然不看测试集。测试集只在全部权重选定之后评估一次。"
    )
    p(rule_text())
    p(
        f"场景阈值来自验证集，写在 {protocol['coverage_median']:.6f} 的深度有效比例中位数、"
        f"亮度 20 分位 {protocol['brightness_p20']:.6f}、亮度 50 分位 {protocol['brightness_p50']:.6f}、"
        f"以及有效深度 75 分位 {protocol['far_mm_p75']} 毫米。"
        f"验证集落入高、低有效深度的图像数是 {protocol['val_group_image_counts']['depth_high']} 和 "
        f"{protocol['val_group_image_counts']['depth_low']}；低照度和正常照度是 "
        f"{protocol['val_group_image_counts']['light_low']} 和 {protocol['val_group_image_counts']['light_normal']}。"
        "低照度与正常照度之间的图像不进入这两个组。远距离、小目标、深度空洞邻域和语义边界是像素组，阈值分别是"
        f" {protocol['far_mm_p75']} 毫米、连通域面积 {protocol['small_area']}、膨胀半径 {protocol['occlusion_radius']} 和 {protocol['boundary_radius']}。"
        "几何结构类是管路、轨道区域、支护设备、门、金属构件和锚固设备。外观纹理类是人员、线缆、指示装置、电气设备、"
        "电子设备、采掘设备、工具材料、救援设备和容器。这两组按类别定义事先划分，不按测试分数挑选。"
    )
    rows = [[item["title"], item["change"]] for item in VARIANTS]
    table("表 1  一次只改一个因素。", ["变体", "相对完整模型的改动"], rows)

    h(2, "2  验证集选择")
    p("下表是每个种子实际保存的轮次。地板未达到时，使用事先规定的验证集平均交并比后备，不改用测试集。")
    select_rows = []
    fallback_notes = []
    for item in VARIANTS:
        for seed in SEEDS:
            record = found[(item["id"], int(seed))]
            chosen = record["selection"]
            flag = "是" if chosen["floor_met"] else "否，后备"
            select_rows.append([
                item["title"],
                str(seed),
                str(chosen["epoch"]),
                flag,
                pct(chosen["val_pa_16"]),
                pct(chosen["val_mpa_16"]),
                pct(chosen["val_miou_16"]),
                f"{record['val_check']['abs_gap']['miou_16']:.2e}",
            ])
            if not chosen["floor_met"]:
                fallback_notes.append(f"{item['title']}种子{seed}")
    table(
        "表 2  验证集选择。PA、MPA、mIoU 是该轮验证集的 16 类分数，单位是百分比。最后一列是重评与保存分数的 mIoU 绝对差。",
        ["变体", "种子", "轮次", "达到地板", "PA", "MPA", "mIoU", "重评差"],
        select_rows,
    )
    if fallback_notes:
        p("使用后备规则的运行：" + "，".join(fallback_notes) + "。")
    else:
        p("十八次运行都有至少一轮同时达到两道验证集地板，因此没有使用后备规则。")

    h(2, "3  测试集主结果")
    p("差值等于该变体减去同一种子的完整模型，再对三种子取均值和样本标准差。负值表示该变体低于完整模型。")
    main_rows = []
    judgements = {}
    for item in VARIANTS:
        pa = metric_pair(found, item["id"], "pa_16")
        mpa = metric_pair(found, item["id"], "mpa_16")
        miou = metric_pair(found, item["id"], "miou_16")
        if item["id"] == "full":
            main_rows.append([item["title"], pm(*pa), pm(*mpa), pm(*miou), "—", "—", "—"])
        else:
            _, pa_d = delta_pair(found, item["id"], "pa_16")
            _, mpa_d = delta_pair(found, item["id"], "mpa_16")
            raw, miou_d = delta_pair(found, item["id"], "miou_16")
            judgements[item["id"]] = judge(raw)
            main_rows.append([item["title"], pm(*pa), pm(*mpa), pm(*miou), spm(*pa_d), spm(*mpa_d), spm(*miou_d)])
    table(
        "表 3  测试集 16 类。单位是百分比，± 后面是三种子样本标准差。",
        ["变体", "PA", "MPA", "mIoU", "PA差", "MPA差", "mIoU差"],
        main_rows,
    )
    fifteen = []
    for item in VARIANTS:
        miou = metric_pair(found, item["id"], "miou_15")
        fifteen.append([item["title"], pm(*miou)])
    table("表 4  测试集 15 类平均交并比，单位是百分比。它不参与判读。", ["变体", "15类mIoU"], fifteen)
    seed_rows = []
    for item in VARIANTS:
        for seed in SEEDS:
            record = found[(item["id"], int(seed))]
            test = record["test"]
            seed_rows.append([item["title"], str(seed), pct(test["pa_16"]), pct(test["mpa_16"]), pct(test["miou_16"])])
    table("表 5  每个种子的测试集 16 类分数，单位是百分比。", ["变体", "种子", "PA", "MPA", "mIoU"], seed_rows)

    h(2, "4  机制")
    p(
        "去掉 Relay 并重新训练，回答的是 Relay 这一整块有没有可复现的增益。"
        "有效性掩码为常数，只改 Relay 内部的掩码，用来看增益是否依赖对无效深度的筛选。在当前实现里，覆盖度变成 1 之后，写进空洞的推断支路会乘上 0。"
        "关闭邻域传递保留全部 Relay 卷积和参数，但只从中心点取几何特征，用来区分传递机制和额外参数。"
        "原始深度和去掉几何输入只改进入 Relay 与几何头的张量。"
    )
    for item in VARIANTS:
        if item["id"] == "full":
            continue
        kind, mu, sd = judgements[item["id"]]
        p(judgement_sentence(item["id"], kind, mu, sd))

    h(2, "5  场景")
    p("分组规则和阈值在训练开始前写入协议文件。下表只比较完整模型和去掉 Relay 的重新训练。差仍然是去掉 Relay 减去完整模型。")
    group_rows = []
    stable_groups = []
    for name in GROUP_ORDER:
        full_values, no_values = [], []
        pixels = images = None
        for seed in SEEDS:
            full_group = found[("full", int(seed))]["groups"][name]
            no_group = found[("no_relay", int(seed))]["groups"][name]
            if full_group["pixels"] != no_group["pixels"] or full_group["images"] != no_group["images"]:
                raise SystemExit(f"group occupancy drifted for {name}")
            pixels = full_group["pixels"]
            images = full_group["images"]
            full_values.append(full_group["miou_16"])
            no_values.append(no_group["miou_16"])
        if any(v is None for v in full_values + no_values):
            group_rows.append([GROUP_TITLE[name], str(images), str(pixels), "无", "无", "无"])
            continue
        full_mu = mean_std(full_values)
        no_mu = mean_std(no_values)
        deltas = [b - a for a, b in zip(full_values, no_values)]
        kind, mu, sd = judge(deltas)
        delta_mu = mean_std(deltas)
        group_rows.append([
            GROUP_TITLE[name], str(images), str(pixels), pm(*full_mu), pm(*no_mu), spm(*delta_mu),
        ])
        if kind == "stable_negative":
            stable_groups.append(GROUP_TITLE[name])
    table(
        "表 6  测试集分组的 16 类平均交并比。单位是百分比。图像数和像素数不随种子改变。",
        ["分组", "图像", "像素", "完整模型", "去掉Relay", "mIoU差"],
        group_rows,
    )
    overall_kind = judgements["no_relay"][0]
    if overall_kind != "stable_negative":
        p("去掉 Relay 的总体 mIoU 差不是稳定为负，因此本文不把上表里的个别分组解释成 Relay 的适用场景。数字仍按事先分组报告。")
    elif stable_groups:
        p("总体差稳定为负，并且下列分组也满足同一条稳定标准、方向为去掉 Relay 后更低：" + "、".join(stable_groups) + "。")
    else:
        p("总体差稳定为负，但没有任何一个事先分组同时满足同一条稳定标准。增益没有集中到单一场景。")

    h(2, "6  类别")
    geo_full = class_mean(found, "full", GEOMETRY_IDS)
    geo_no = class_mean(found, "no_relay", GEOMETRY_IDS)
    app_full = class_mean(found, "full", APPEARANCE_IDS)
    app_no = class_mean(found, "no_relay", APPEARANCE_IDS)
    geo_delta = mean_std([b - a for a, b in zip(geo_full[0], geo_no[0])])
    app_delta = mean_std([b - a for a, b in zip(app_full[0], app_no[0])])
    p(
        f"几何结构类的非加权平均交并比，完整模型是 {pm(*geo_full[1])}%，去掉 Relay 是 {pm(*geo_no[1])}%，"
        f"差是 {spm(*geo_delta)} 个百分点。外观纹理类分别是 {pm(*app_full[1])}% 和 {pm(*app_no[1])}%，"
        f"差是 {spm(*app_delta)} 个百分点。差仍是去掉 Relay 减去完整模型。"
    )
    class_rows = []
    for index, name in enumerate(CLASS_ZH):
        full_values = [found[("full", int(seed))]["test"]["per_class_iou_16"][index] for seed in SEEDS]
        no_values = [found[("no_relay", int(seed))]["test"]["per_class_iou_16"][index] for seed in SEEDS]
        delta = mean_std([b - a for a, b in zip(full_values, no_values)])
        class_rows.append([name, pm(*mean_std(full_values)), pm(*mean_std(no_values)), spm(*delta)])
    table(
        "表 7  测试集各类 16 类交并比。单位是百分比。差是去掉 Relay 减去完整模型。",
        ["类别", "完整模型", "去掉Relay", "差"],
        class_rows,
    )

    h(2, "7  效率和可视化")
    p("耗时在验证集上测量，不产生测试分数。每种结构用种子 2026 的已选权重，批大小 1，先预热 5 张，再计 30 张。显存是这 30 张期间的峰值分配。")
    eff_rows = []
    efficiencies = {}
    for item in VARIANTS:
        record = found[(item["id"], int(SEEDS[0]))]
        eff = record["efficiency"]
        if not eff:
            raise SystemExit(f"missing efficiency for {item['id']}")
        efficiencies[item["id"]] = eff
        eff_rows.append([
            item["title"],
            f"{eff['active'] / 1e6:.2f}",
            f"{eff['trainable'] / 1e6:.2f}",
            f"{eff['relay'] / 1e6:.2f}",
            f"{eff['milliseconds_per_image']:.1f}",
            f"{eff['peak_memory_mib']:.0f}",
        ])
    table(
        "表 8  参数量、单张耗时和峰值显存。活跃参数在去掉 Relay 时不含 Relay 模块。",
        ["变体", "活跃参数(百万)", "可训练(百万)", "Relay参数(百万)", "毫秒/张", "显存(MiB)"],
        eff_rows,
    )
    extra = efficiencies["full"]["active"] - efficiencies["no_relay"]["active"]
    extra_ms = efficiencies["full"]["milliseconds_per_image"] - efficiencies["no_relay"]["milliseconds_per_image"]
    extra_mem = efficiencies["full"]["peak_memory_mib"] - efficiencies["no_relay"]["peak_memory_mib"]
    p(
        f"完整模型比去掉 Relay 的模型多 {extra / 1e6:.2f} 百万个活跃参数，单张耗时差 {extra_ms:+.1f} 毫秒，"
        f"峰值显存差 {extra_mem:+.0f} MiB。Relay 模块本身有 {efficiencies['full']['relay'] / 1e6:.2f} 百万参数。"
        "若表 3 里去掉 Relay 的 mIoU 差很小或不稳定，这些开销没有换成稳定的测试集增益。"
    )
    p(
        "可视化使用验证集上深度有效比例大于 0 且最低的 4 张，选择时只看有效比例和图像 id，不看预测误差。"
        "图中从左到右是 RGB、无效深度、归一化后还原的 HHA、Relay 第一层离开中心点的权重质量、完整模型预测、去掉 Relay 的预测，以及二者在标签不是 255 的像素上的差异。"
        "权重图来自种子 2026 的完整模型。预测差异来自同一种子的两个已选权重。"
    )
    contact = PAPER / "figures" / "contact.png"
    chart = PAPER / "figures" / "delta-miou.png"
    draw_chart(found, chart)
    if contact.is_file():
        fig("figures/contact.png", "图 1  验证集低有效深度图像上的几何输入、Relay 权重和预测差异。")
    else:
        p("接触图没有生成，论文不引用缺失图片。")
    fig("figures/delta-miou.png", "图 2  测试集 16 类 mIoU 差，单位是百分点。误差线是三种子样本标准差。")

    h(2, "8  结论")
    kind, mu, sd = judgements["no_relay"]
    p(judgement_sentence("no_relay", kind, mu, sd))
    if kind == "stable_negative":
        p("在这条事先写定的标准下，重新训练后的 Relay 带来了方向稳定的测试集平均交并比增益。")
    elif kind == "stable_positive":
        p("在这条标准下，去掉 Relay 并重新训练反而更高。本文不主张 Relay 提供了增益。")
    elif kind == "small":
        p("在这条标准下，Relay 的独立贡献很小。更简单的结构可以去掉这块模块。")
    else:
        p("在这条标准下，Relay 的独立贡献不稳定。本文不把它写成已经证实的收益，可以考虑简化 Relay。")
    p("机制变体的句子已在第 4 节按同一条规则写出。本文不把未达到稳定标准的单项差异解释成设计原因。")
    p(
        "文件：结果 JSON 是 /root/autodl-tmp/MineSeg/v2/runs/relay_retrain/results.json。"
        "分组阈值是同目录的 protocol.json。每个运行的验证历史、选择记录和 best.pth 在该目录的变体子文件夹里。"
        "论文的 Markdown、LaTeX、Word 和 PDF 在 /root/autodl-tmp/MineSeg/v2/paper_relay_retrain。"
    )
    return blocks


def draw_chart(found, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, means, stds = [], [], []
    for item in VARIANTS:
        if item["id"] == "full":
            continue
        _raw, (mu, sd) = delta_pair(found, item["id"], "miou_16")
        labels.append(item["title"])
        means.append(mu * 100)
        stds.append(sd * 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    if font.is_file():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
        plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(8, 4.2))
    positions = range(len(labels))
    axis.barh(list(positions), means, xerr=stds, color="#3c6e9f", ecolor="#222222", capsize=3)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels)
    axis.set_xlabel("test mIoU difference (percentage points)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_markdown(blocks):
    lines = []
    for block in blocks:
        if block[0] == "h":
            lines.append("#" * block[1] + " " + block[2])
            lines.append("")
        elif block[0] == "p":
            lines.append(block[1])
            lines.append("")
        elif block[0] == "fig":
            _kind, path, caption = block
            lines.append(f"![{caption}]({path})")
            lines.append("")
            lines.append(caption)
            lines.append("")
        else:
            _, caption, headers, rows = block
            lines.append(caption)
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(item) for item in row) + " |")
            lines.append("")
    (PAPER / "paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(blocks):
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1.6cm]{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        r"\setCJKmainfont{WenQuanYi Micro Hei}[AutoFakeBold=2.5]",
        r"\setmainfont{TeX Gyre Termes}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{caption}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{setspace}",
        r"\setstretch{1.12}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\begin{document}",
    ]
    for block in blocks:
        if block[0] == "h":
            level, text = block[1], block[2]
            if level == 1:
                parts.append(r"{\centering\Large " + tex_escape(text) + r"\par}")
                parts.append(r"\vspace{0.6em}")
            else:
                parts.append(r"\section*{" + tex_escape(text) + "}")
        elif block[0] == "p":
            parts.append(tex_escape(block[1]))
            parts.append("")
        elif block[0] == "fig":
            _kind, path, caption = block
            parts.append(r"\begin{center}")
            parts.append(r"\includegraphics[width=\textwidth]{" + path + "}")
            parts.append(r"\par\small " + tex_escape(caption))
            parts.append(r"\end{center}")
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
            parts.append(r"\end{longtable}}")
    parts.append(r"\end{document}")
    (PAPER / "paper.tex").write_text("\n".join(parts) + "\n", encoding="utf-8")


def set_run_font(run, size=11, bold=False):
    run.font.size = Pt(size)
    run.bold = bold
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "WenQuanYi Micro Hei")


def write_docx(blocks):
    document = Document()
    section = document.sections[0]
    for margin in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, margin, Cm(1.6))
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "WenQuanYi Micro Hei")
    for block in blocks:
        if block[0] == "h":
            level, text = block[1], block[2]
            paragraph = document.add_paragraph()
            if level == 1:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(text)
            set_run_font(run, 16 if level == 1 else 13, True)
            if level > 1:
                run.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
        elif block[0] == "p":
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.first_line_indent = Cm(0.74)
            run = paragraph.add_run(block[1])
            set_run_font(run, 10.5, False)
        elif block[0] == "fig":
            _kind, path, caption = block
            document.add_picture(str(PAPER / path), width=Inches(6.2))
            paragraph = document.add_paragraph()
            run = paragraph.add_run(caption)
            set_run_font(run, 9, True)
        else:
            _, caption, headers, rows = block
            cap = document.add_paragraph()
            run = cap.add_run(caption)
            set_run_font(run, 9, True)
            table = document.add_table(rows=1 + len(rows), cols=len(headers))
            table.style = "Table Grid"
            for index, header in enumerate(headers):
                cell = table.rows[0].cells[index]
                cell.text = ""
                run = cell.paragraphs[0].add_run(header)
                set_run_font(run, 7, True)
            for r_index, row in enumerate(rows):
                for c_index, value in enumerate(row):
                    cell = table.rows[r_index + 1].cells[c_index]
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(str(value))
                    set_run_font(run, 7, False)
            document.add_paragraph()
    document.save(PAPER / "paper.docx")


def main():
    if not RESULT.is_file():
        raise SystemExit(f"missing {RESULT}")
    result = load(RESULT)
    protocol = load(PROTOCOL)
    found = records_by(result)
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(exist_ok=True)
    blocks = build_blocks(result, protocol, found)
    write_markdown(blocks)
    write_latex(blocks)
    write_docx(blocks)
    shutil.copy(RESULT, PAPER / "results.json")
    print(f"wrote {PAPER / 'paper.md'}", flush=True)


if __name__ == "__main__":
    main()
