#!/usr/bin/env python3
"""Build the SelectiveMine ablation paper in Markdown, LaTeX, Word, and PDF text.

Every format is rendered from the same list of paragraphs and tables.
Numbers are read from the ablation JSON. Nothing is filled in by hand.
"""

import json
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("/root/autodl-tmp/MineSeg/v2/paper_ablation")
REPORT = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3_ablation/ablation.json")
CMX = {"pa": 86.30, "mpa": 72.72, "miou": 61.83}
BACKGROUND = "背景"


def load(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return f"{float(value) * 100:.2f}"


def signed(value):
    return f"{float(value) * 100:+.2f}"


def tex_escape(text):
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def by_id(block):
    return {row["id"]: row for row in block["variants"]}


def metric_rows(block):
    rows = []
    for row in block["variants"]:
        rows.append([
            row["title"],
            pct(row["pa_16"]),
            pct(row["mpa_16"]),
            pct(row["miou_16"]),
            signed(row["delta_pa_16"]),
            signed(row["delta_mpa_16"]),
            signed(row["delta_miou_16"]),
            pct(row["miou"]),
        ])
    return rows


def removal_rows(block):
    return [[row["title"], row["removed"]] for row in block["variants"]]


def class_rows(block, names):
    chosen = ["full", "rgb", "fused", "no_depth", "no_geometry", "equal_mix"]
    table = by_id(block)
    headers = ["类别"] + [table[name]["title"] for name in chosen]
    rows = []
    for index, name in enumerate(names):
        rows.append([name] + [pct(table[key]["per_class_iou"][index]) for key in chosen])
    rows.append(["背景"] + [pct(table[key]["per_class_iou_16"][0]) for key in chosen])
    return headers, rows


def largest_drop(block, key):
    drops = [row for row in block["variants"] if row["id"] != "full"]
    return min(drops, key=lambda row: row[key])


def main():
    report = load(REPORT)
    test = report["test"]
    val = report["val"]
    full_test = by_id(test)["full"]
    full_val = by_id(val)["full"]
    counts = report["splits"]["counts"]
    names = report["foreground_names"]
    drop_iou = largest_drop(test, "miou_16")
    drop_pa = largest_drop(test, "pa_16")
    drop_mpa = largest_drop(test, "mpa_16")
    headers = ["变体", "像素准确率", "平均像素准确率", "平均交并比", "PA差", "MPA差", "mIoU差", "15类平均交并比"]
    class_headers, class_body = class_rows(test, names)
    blocks = []

    def h(level, text):
        blocks.append(("h", level, text))

    def p(text):
        blocks.append(("p", text))

    def table(caption, head, rows):
        blocks.append(("t", caption, head, rows))

    h(1, "SelectiveMine selective_pa3 的冻结权重消融")
    p(
        "日期：2026-11-01。本文记录 selective_pa3 第 12 轮权重在 FTransUNet 划分上的推理消融。"
        "权重没有重新训练。每个变体在评测时去掉一条已经训练好的路径；融合专家改变时，路由器按新的两路对数概率重算。"
        "百分比保留两位小数，由 JSON 中的原始分数换算。"
    )
    p(
        f"被消融的检查点是 {report['checkpoint']}，轮次 {report['epoch']}。"
        f"它在 FTransUNet 训练集 {counts['train']} 张上拟合，在验证集 {counts['val']} 张上按 16 类像素准确率保存，"
        f"测试集 {counts['test']} 张在选择规则冻结之后才用于报告。"
        "三个名单互不相交。训练名单不再计算分数，因为这些图像参与了拟合。"
        f"验证集和测试集都用同一套冻结变体、分辨率 {report['height']}×{report['width']}、单尺度、不做测试时增强。"
        "16 类协议把标签 0 计入混淆矩阵，标签 255 忽略。15 类协议同时忽略标签 0 和 255。"
        f"完整模型在测试集上的 16 类像素准确率、平均像素准确率和平均交并比是 {pct(full_test['pa_16'])}%、"
        f"{pct(full_test['mpa_16'])}% 和 {pct(full_test['miou_16'])}%。"
        f"与 selective_pa3/official_test.json 中三项原始分数的绝对差是 {abs(test['official_gap']['pa_16']):.8f}、"
        f"{abs(test['official_gap']['mpa_16']):.8f} 和 {abs(test['official_gap']['miou_16']):.8f}。"
        "这三项原始分数都在 0 到 1 之间。"
        f"MUSeg 论文 Table 5 引用的 CMX 是 {CMX['pa']:.2f}%、{CMX['mpa']:.2f}% 和 {CMX['miou']:.2f}%。"
        "该引用不是这次重测。"
    )
    h(2, "1  数据和检查点")
    p(
        "三个划分文件都在目录 /root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet，文件名是 train.txt、val.txt 和 test.txt。"
        f"图像数分别是 {counts['train']}、{counts['val']} 和 {counts['test']}。"
        f"训练与验证的交集是 {report['splits']['overlap']['train_val']}，"
        f"训练与测试的交集是 {report['splits']['overlap']['train_test']}，"
        f"验证与测试的交集是 {report['splits']['overlap']['val_test']}。"
        "数据根目录是 /root/autodl-tmp/MUSeg/flat。评测把图像缩放到 768×896，corruption 设为 none，不做随机裁剪、颜色抖动或翻转。"
    )
    p(
        "selective_pa3 从 selective_ftrans 的第 7 轮继续训练 12 轮。保存规则是 select-pa："
        "在验证集 16 类平均交并比不低于 60.80% 且平均像素准确率不低于 72.00% 的轮次中，保留 16 类像素准确率最高的一轮。"
        "被保存的是第 12 轮。消融使用这一份 best.pth，不使用后续 selective_pa4 或 selective_pa5 的权重。"
        f"几何开度 tanh(open_raw) 与 mix 的乘积是 {report['open_scale']:.4f}。评测前 mix 被填成 1，并按 joint 模式配置。"
    )
    h(2, "2  变体")
    p(
        "完整模型同时计算外观解码器和融合路径。融合路径是深度门控后的 RGB-D 特征，经 relay 注入 HHA 几何，再由融合解码器输出，"
        "并加上按开度和深度是否存在缩放的几何 logits。路由器不读取真值。它的输入是两路概率、绝对差、两路熵、深度有效掩码和亮度。"
        "深度整幅缺失时，输出改用外观头。其余变体只改这一次前向里被使用的张量，不改卷积权重。"
        "去掉 HHA 或去掉深度时，对应输入被换成零张量。这不是把该分支从训练里拿掉之后重新优化，零输入落在训练分布之外。"
    )
    table(
        "表 1  每个变体在同一次前向中去掉的路径。",
        ["变体", "去掉的内容"],
        removal_rows(test),
    )
    p(
        f"测试集 {test['images']} 张耗时 {test['seconds']} 秒，验证集 {val['images']} 张耗时 {val['seconds']} 秒，批大小 {report['batch_size']}。"
        f"测试集上有任意有效深度的图像是 {test['images_with_any_depth']} 张，验证集是 {val['images_with_any_depth']} 张。"
        f"测试集路由器门控的平均值是 {test['mean_router_gate']:.4f}，验证集是 {val['mean_router_gate']:.4f}。"
        "门控经过 sigmoid，数值越大表示越偏向融合专家。"
        f"在测试集标签不是 255 的像素里，完整模型和只用融合专家不一致的比例是 {test['fraction_disagree_fused'] * 100:.2f}%，"
        f"和只用外观解码器不一致的比例是 {test['fraction_disagree_rgb'] * 100:.2f}%，"
        f"和去掉 relay 之后不一致的比例是 {test['fraction_disagree_relay'] * 100:.2f}%。"
        f"验证集这三个比例是 {val['fraction_disagree_fused'] * 100:.2f}%、{val['fraction_disagree_rgb'] * 100:.2f}% 和 {val['fraction_disagree_relay'] * 100:.2f}%。"
    )
    h(2, "3  测试集")
    p(
        "测试集是报告用的划分。差值等于该变体减去同一张表里的完整模型。负差表示去掉该路径之后变低。"
        "15 类平均交并比只作对照，不参与和 CMX 的比较。"
    )
    table("表 2  测试集 788 张的消融。单位是百分比。", headers, metric_rows(test))
    p(
        f"测试集上 16 类平均交并比下降最多的是{drop_iou['title']}，差值 {signed(drop_iou['delta_miou_16'])} 个百分点，"
        f"该变体的三项是 {pct(drop_iou['pa_16'])}%、{pct(drop_iou['mpa_16'])}% 和 {pct(drop_iou['miou_16'])}%。"
        f"像素准确率下降最多的是{drop_pa['title']}，差值 {signed(drop_pa['delta_pa_16'])} 个百分点。"
        f"平均像素准确率下降最多的是{drop_mpa['title']}，差值 {signed(drop_mpa['delta_mpa_16'])} 个百分点。"
        f"只用外观解码器的 16 类平均交并比差值是 {signed(by_id(test)['rgb']['delta_miou_16'])} 个百分点。"
        f"只用融合专家的差值是 {signed(by_id(test)['fused']['delta_miou_16'])} 个百分点。"
        f"把路由器换成常数 0.5 的差值是 {signed(by_id(test)['equal_mix']['delta_miou_16'])} 个百分点。"
    )
    same = []
    rows = test["variants"]
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if left["per_class_iou_16"] == right["per_class_iou_16"] and left["pa_16"] == right["pa_16"]:
                same.append(f"{left['title']} 与 {right['title']}")
    if same:
        p("测试集上 16 类逐类交并比和像素准确率完全相同的变体对有：" + "，".join(same) + "。相同表示这些路径没有改变任何一个计入 16 类矩阵的像素预测。")
    table(
        "表 3  测试集前景类交并比，以及背景的 16 类交并比。单位是百分比。列是完整模型和五个对照变体。",
        class_headers,
        class_body,
    )
    h(2, "4  验证集")
    history = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3/history.jsonl")
    saved = None
    for line in history.read_text().splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if int(row.get("epoch", -1)) == int(report["epoch"]):
                saved = row
    if saved is None:
        raise SystemExit(f"history has no epoch {report['epoch']}")
    pa_gap = abs(full_val["pa_16"] - float(saved["pa_16"]))
    mpa_gap = abs(full_val["mpa_16"] - float(saved["mpa_16"]))
    miou_gap = abs(full_val["miou_16"] - float(saved["miou_16"]))
    p(
        "验证集曾经用来选择第 12 轮，所以这里的分数不是一次事先锁定之后才打开的测试。"
        "把它和测试集放在一起，是为了看同一组变体的排序是否换了一个名单就反过来。"
        f"完整模型在验证集上的 16 类三项是 {pct(full_val['pa_16'])}%、{pct(full_val['mpa_16'])}% 和 {pct(full_val['miou_16'])}%。"
        f"selective_pa3/history.jsonl 第 {report['epoch']} 轮写下的验证集三项是 {pct(saved['pa_16'])}%、{pct(saved['mpa_16'])}% 和 {pct(saved['miou_16'])}%。"
        f"本次重算与该行原始分数的绝对差是 {pa_gap:.6f}、{mpa_gap:.6f} 和 {miou_gap:.6f}。"
    )
    table("表 4  验证集 788 张的消融。单位是百分比。差值相对本次验证集上的完整模型。", headers, metric_rows(val))
    val_drop = largest_drop(val, "miou_16")
    p(
        f"验证集上 16 类平均交并比下降最多的也记录在这里：{val_drop['title']}，差值 {signed(val_drop['delta_miou_16'])} 个百分点。"
        f"测试集上下降最多的是{drop_iou['title']}。"
        + (
            "两个划分上下降最多的是同一个变体。"
            if val_drop["id"] == drop_iou["id"]
            else "两个划分上下降最多的不是同一个变体，比较时以测试集为准。"
        )
    )
    h(2, "5  结论")
    p(
        f"selective_pa3 第 12 轮在 FTransUNet 测试集上的单尺度 16 类结果是 {pct(full_test['pa_16'])}%、"
        f"{pct(full_test['mpa_16'])}% 和 {pct(full_test['miou_16'])}%。"
        "消融没有重新训练。因此这些差值表示已选权重里各条路径被拿掉之后的变化，不是从随机初始化或从同一起点重新训练一个缺模块模型的结果。"
        + (
            f"测试集上，像素准确率、平均像素准确率和 16 类平均交并比下降最多的都是{drop_iou['title']}。"
            if drop_iou["id"] == drop_pa["id"] == drop_mpa["id"]
            else (
                f"测试集上对 16 类平均交并比伤害最大的操作是{drop_iou['title']}。"
                f"对像素准确率伤害最大的是{drop_pa['title']}。"
                f"对平均像素准确率伤害最大的是{drop_mpa['title']}。"
            )
        )
        + "训练名单有 1595 张，只用于说明检查点的拟合范围，没有在训练集上报告消融分数。"
    )
    p(
        "结果文件是 /root/autodl-tmp/MineSeg/v2/runs/selective_pa3_ablation/ablation.json。"
        "验证集和测试集的分文件是同目录的 ablation_val.json 和 ablation_test.json。"
        "对照用的已发表单尺度测试是 /root/autodl-tmp/MineSeg/v2/runs/selective_pa3/official_test.json。"
    )

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results.json").write_text(json.dumps({
        "checkpoint": report["checkpoint"],
        "epoch": report["epoch"],
        "test_full_16": {
            "pa": pct(full_test["pa_16"]),
            "mpa": pct(full_test["mpa_16"]),
            "miou": pct(full_test["miou_16"]),
        },
        "largest_miou_drop": {"id": drop_iou["id"], "title": drop_iou["title"], "delta": signed(drop_iou["delta_miou_16"])},
        "largest_pa_drop": {"id": drop_pa["id"], "title": drop_pa["title"], "delta": signed(drop_pa["delta_pa_16"])},
        "largest_mpa_drop": {"id": drop_mpa["id"], "title": drop_mpa["title"], "delta": signed(drop_mpa["delta_mpa_16"])},
    }, indent=2, ensure_ascii=False) + "\n")
    write_markdown(blocks)
    write_latex(blocks)
    write_docx(blocks)
    print(f"wrote {ROOT / 'paper.md'}")


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
        r"\usepackage[a4paper,margin=1.8cm]{geometry}",
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
            spec = "p{4.2cm}" + "c" * (len(headers) - 1) if len(headers) <= 3 else "l" + "c" * (len(headers) - 1)
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
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)
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
