
# SegFormer-Dual: RGB-D 矿井语义分割项目

本项目基于 **SegFormer-B2** 架构，针对矿井复杂环境下的 16 类目标实现了 **RGB + Depth 双模态** 语义分割。通过融合深度信息，模型在光照极差、粉尘遮挡及语义高度相似的场景下展现了卓越的鲁棒性。


---

## 一、 实验结果 (Validation Set)

在经过 150 Epochs 训练后，模型各项核心指标表现优异：

### 1. 核心性能指标

| 指标 | 数值 |
| --- | --- |
| **mIoU** | **71.45%** |
| **mPA** | **81.31%** |
| **Accuracy** | **89.50%** |
<img width="640" height="480" alt="mIoU" src="https://github.com/user-attachments/assets/f0d73517-c2b2-4796-b0ad-70a286a877c4" />

<img width="640" height="480" alt="mPA" src="https://github.com/user-attachments/assets/8ca1ab14-7e10-4a47-b714-1b83f87389de" />

### 2. 类别表现分析

* **优势类别 (>80%)**:
* `Background` (89.14%)、`Anchor` (84.17%)、`Door` (84.47%)、`Support Equipment` (82.12%)。
* *结论：模型对大型固定资产和背景噪声的抑制能力极强。*


* **中等精度 (60%-80%)**:
* `Person` (69.13%)、`Tube` (68.54%)、`Rail` (72.89%)。


* **待优化类别 (<60%)**:
* `Metal Fixture` (55.61%)、`Electronic Equipment` (56.31%)。
* *原因：目标尺寸微小、几何特征在深度图中不够明显。*



---

## 二、 深度分析：混淆矩阵 (Confusion Matrix)

基于 `miou_out/confusion_matrix.csv` 的定性研究：

* **边缘混淆（Edge Ambiguity）**: `metal fixture` 与 `background` 存在混淆，主要受限于细长形状在低分辨率特征图下的像素损失。
* **语义相似性挑战**: `cable`（电缆）与 `tube`（管路）在纹理模糊时较难区分。
* **背景鲁棒性**: `background` Recall 达到 **95.07%**，有效减少了假阳性误报。

---

## 三、 目录结构

```text
SegFormer/
├── datasets/                # 数据集根目录
│   ├── DepthImages/         # 深度图数据
│   ├── JPEGImages/          # RGB 原图
│   ├── SegmentationClass/   # 语义分割标签 (Mask)
│   └── ImageSets/           # 训练/验证集划分文件
├── logs/                    # 权重文件 (.pth) 与训练日志
├── miou_out/                # 评估结果 (混淆矩阵、mIoU 柱状图等)
├── nets/                    # SegFormer 双路网络架构实现
├── utils/                   # 工具函数 (数据处理、指标计算)
├── dual_train.py            # 双模态训练主程序
├── dual_get_miou.py         # mIoU 评估脚本
├── dual_segformer.py        # 模型推理封装
└── Installation.txt         # 环境配置清单

```

---

## 四、 快速上手

### 1. 环境准备

参考 `Installation.txt`。建议环境：

* GPU: NVIDIA RTX 4090 (24GB)
* Python 3.8+ & PyTorch 1.10+

### 2. 训练模型

```bash
python dual_train.py

```

### 3. 模型评估

1. 修改 `dual_segformer.py` 中的 `model_path` 为 `logs/` 下的最佳权重文件。
2. 运行评估脚本：

```bash
python dual_get_miou.py

```

---

## 五、 未来改进方向

1. **数据增强**: 针对 `metal fixture` 等小样本类别引入缩放、旋转及 CutMix 增强。
2. **损失函数优化**: 引入 **Dice Loss** 或 **Focal Loss** 缓解类别不平衡问题。
3. **融合机制**: 探索在 Decoder 阶段加入 **Cross-Modal Attention** 以更好地平衡 RGB 和深度特征的权重。

---

**想了解更多细节？** 您可以查看 `miou_out/mIoU.png` 获取完整的可视化分析报告。

> 您是否需要我为您编写一个用于自动绘制 Loss 和 mIoU 曲线的 Python 绘图脚本？
