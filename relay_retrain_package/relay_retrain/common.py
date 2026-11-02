"""Shared protocol for the Geometry Relay retrain study.

Numbers and rules live here so the trainer, the evaluator, the paper, and the
mail report cannot drift apart. This module does not import torch.
"""

from pathlib import Path

CODE = Path("/root/autodl-tmp/MineSeg/v2/relay_retrain")
RUN = Path("/root/autodl-tmp/MineSeg/v2/runs/relay_retrain")
PAPER = Path("/root/autodl-tmp/MineSeg/v2/paper_relay_retrain")
ZIP_PATH = Path("/root/autodl-tmp/MineSeg/v2/relay_retrain_package.zip")
INIT = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_ftrans/best.pth")
FROZEN_ABLATION = Path("/root/autodl-tmp/MineSeg/v2/runs/selective_pa3_ablation/ablation.json")
DATA_ROOT = Path("/root/autodl-tmp/MUSeg/flat")
SPLIT_DIR = Path("/root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet")
SOURCE = Path("/root/autodl-tmp/SelectiveMine_best_20261024/source")

SEEDS = (2026, 2027, 2028)
EPOCHS = 12
BATCH_SIZE = 4
HEIGHT = 768
WIDTH = 896
LR = 6e-6
WORKERS = 4
RARE_CROP = 0.4
LOVASZ = 0.25
PIXEL_CE = 0.12
BG_BOOST = 1.15
PA_FLOOR_MIOU = 0.608
PA_FLOOR_MPA = 0.72
SMALL_AREA = 2048
OCCLUSION_RADIUS = 8
BOUNDARY_RADIUS = 3
# Pre-registered before any test score is computed.
SMALL_MEAN = 0.002
SMALL_STD = 0.005

CLASS_ZH = [
    "背景", "人员", "线缆", "管路", "指示装置", "电气设备", "电子设备", "采掘设备",
    "轨道区域", "支护设备", "门", "工具材料", "救援设备", "容器", "金属构件", "锚固设备",
]
# Fixed from the class definition, not from a test score.
# Geometry: the label follows a surface or a structural layout.
# Appearance: the label is carried more by color, texture, or a compact object.
GEOMETRY_IDS = (3, 8, 9, 10, 14, 15)
APPEARANCE_IDS = (1, 2, 4, 5, 6, 7, 11, 12, 13)

VARIANTS = (
    {
        "id": "full",
        "title": "完整模型",
        "change": "保留 Geometry Relay，几何输入为 HHA。",
    },
    {
        "id": "no_relay",
        "title": "去掉 Relay 并重训",
        "change": "前向不调用 Relay，Relay 参数不加入优化器。深度门控、HHA 几何头和路由器保留。",
    },
    {
        "id": "const_valid",
        "title": "有效性掩码为常数",
        "change": "只把 Relay 内部的深度有效掩码换成全 1。覆盖度变为 1 后，推断支路乘上的 (1-覆盖度) 为 0，空洞不再写入传递特征。网络其余部分仍使用真实掩码。",
    },
    {
        "id": "no_neighbor",
        "title": "关闭邻域传递",
        "change": "Relay 的卷积和参数量保持不变，偏移只保留中心点。查询、键、值和后续卷积仍执行，但邻域像素不再进入传递结果。",
    },
    {
        "id": "raw_depth",
        "title": "原始深度代替 HHA",
        "change": "几何输入改为对数深度复制成三通道，按与 HHA 相同的均值和方差缩放，再拼上有效掩码，凑成茎网络要求的四通道。不使用高度和法向角。",
    },
    {
        "id": "no_geometry",
        "title": "去掉几何输入",
        "change": "HHA 位置换成全 0。Relay、几何编码器和几何头仍在，只是几何输入为空。",
    },
)

GROUP_ORDER = (
    "depth_high",
    "depth_low",
    "light_low",
    "light_normal",
    "far",
    "small",
    "occlusion",
    "boundary",
)
GROUP_TITLE = {
    "depth_high": "深度有效比例高",
    "depth_low": "深度有效比例低",
    "light_low": "低照度",
    "light_normal": "正常照度",
    "far": "远距离像素",
    "small": "小目标像素",
    "occlusion": "深度空洞邻域",
    "boundary": "语义边界带",
}


def variant_ids():
    return [item["id"] for item in VARIANTS]


def variant_title(name):
    for item in VARIANTS:
        if item["id"] == name:
            return item["title"]
    raise KeyError(name)


def run_dir(variant, seed):
    return RUN / variant / f"seed_{seed}"


def mean_std(values):
    values = [float(v) for v in values]
    count = len(values)
    if count == 0:
        raise ValueError("empty")
    mu = sum(values) / count
    if count < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (count - 1)
    return mu, var ** 0.5


def judge(deltas):
    """Classify variant-minus-full deltas. The thresholds are module constants."""
    mu, sd = mean_std(deltas)
    if abs(mu) < SMALL_MEAN and sd < SMALL_STD:
        return "small", mu, sd
    same = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    if same and abs(mu) > sd:
        label = "stable_positive" if mu > 0 else "stable_negative"
        return label, mu, sd
    return "unstable", mu, sd


def rule_text():
    return (
        "判读规则在查看测试分数之前写定。"
        f"某种子差定义为该变体减去完整模型。"
        f"若 mIoU 差的均值绝对值小于 {SMALL_MEAN * 100:.2f} 个百分点，"
        f"且样本标准差小于 {SMALL_STD * 100:.2f} 个百分点，记为差异很小。"
        "若三种子同号，且均值绝对值大于样本标准差，记为方向稳定。"
        "其余记为不稳定。"
        "只有去掉 Relay 的 mIoU 差方向稳定时，才根据场景表讨论 Relay 在哪些条件下更高。"
        "稳定为负表示去掉 Relay 后更低，也就是完整模型更高。"
    )


def apply_variant(model, name, export_weights=False):
    if name not in variant_ids():
        raise ValueError(name)
    model.geo_input = "hha"
    model.fusion = "relay"
    model.multimodal.mode = "relay"
    for block in model.multimodal.relay:
        block.control = "full"
        block.export_weights = bool(export_weights and name == "full")
    if name == "no_relay":
        model.fusion = "logits"
        model.multimodal.mode = "logits"
    elif name == "const_valid":
        for block in model.multimodal.relay:
            block.control = "const_valid"
    elif name == "no_neighbor":
        for block in model.multimodal.relay:
            block.control = "no_neighbor"
    elif name == "raw_depth":
        model.geo_input = "raw_depth"
    elif name == "no_geometry":
        model.geo_input = "none"
    model.multimodal.configure_training(joint=True)
    model.multimodal.mix.fill_(1.0)
    return model


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    relay = sum(p.numel() for p in model.multimodal.relay.parameters())
    active = total - relay if model.fusion != "relay" else total
    return {
        "total": int(total),
        "trainable": int(trainable),
        "relay": int(relay),
        "active": int(active),
    }
