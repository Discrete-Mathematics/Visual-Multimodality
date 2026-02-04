SegFormer-Dual: RGB-D 矿井语义分割项目
本项目基于 SegFormer-B2 架构，针对复杂环境下的 16 类目标实现了双模态（RGB + Depth）语义分割。通过融合深度信息，模型在光照不均和遮挡场景下展现了出色的鲁棒性。
一、实验结果 
模型在 150 Epochs 训练后，在验证集上的表现如下：
核心指标
暂时无法在飞书文档外展示此内容
 各类别 IoU 表现
模型对不同目标的识别能力存在差异，具体表现如下：
- 高精度类别 (>80%)：background (89.14%)、anchor (84.17%)、door (84.47%)、support equipment (82.12%)。
- 中等精度类别 (60%-80%)：person (69.13%)、tube (68.54%)、rail (72.89%)。
- 需改进类别 (<60%)：metal fixture (55.61%)、electronic equipment (56.31%)。

---
  混淆矩阵深度分析
根据生成的 confusion_matrix.csv，我们对模型性能进行了更深入的定性分析：
- 边缘混淆（Edge Ambiguity）：
- metal fixture（金属固定装置）与 background 之间存在一定程度的混淆。这主要是因为此类目标通常尺寸较小且呈细长状，在深度图中特征不够显著。
- 语义相似性：
- cable（电缆）与 tube（管路）之间存在交叉误判。分析认为两者的几何形状相似，当纹理模糊时，模型较难仅凭外观完全区分。
- 背景鲁棒性：
- background 类别的 Recall 高达 95.07%，说明模型对复杂背景的抑制效果极佳，有效地减少了假阳性（False Positive）预测。
类别性能分析
模型对大型、固定形状目标的识别极其精准，部分类别 IoU 超过 80%：
- Anchor (锚杆): 84.17%
- Door (门): 84.47%
- Support Equipment (支撑设备): 82.12%
注：详细的类别指标见 miou_out/mIoU.png。

---
二、 目录结构说明
SegFormer/
├── datasets/                
│   ├── DepthImages/     
│   ├── ImageSets/
|   |   └── Segmentation/
│   └── JPEGImages/
|   └── SegmentationClass/        
├── logs/               # 存放训练过程中的权重文件 (.pth) 和训练日志
├── miou_out/           # 存放评估结果，包括混淆矩阵、mIoU 柱状图等可视化图表
├── model_data/
├── nets/               #存放 SegFormer 双路网络架构 
├── utils/
├── dual_get_miou.py    #计算验证集 mIoU 的评估脚本
├── dual_segformer.py
├── dual_train.py       # 双模态训练主程序
├── Installation.txt
└── README.md           # 项目说明文档

---
三、使用说明
1. 环境准备
详见  `Installation.txt` 中的 内容，本实验在一块 RTX 4090（未阉割版）上进行实验。 
2. 模型训练
运行以下命令开始训练：
Bash
python dual_train.py
3. 模型评估
训练完成后，修改 dual_segformer.py 中的 model_path 指向最佳权重，然后运行：
Bash
python dual_get_miou.py

---
六、训练可视化
类别 IoU 分布
[图片]
混淆矩阵
详细的像素级混淆情况请查看 miou_out/confusion_matrix.csv。通过该矩阵可以分析出模型在 metal fixture 等细长物体上的误判原因。

---
五、 结论与后续
1. 数据增强：针对 IoU 较低的 metal fixture 和 electronic equipment 增加旋转和缩放增强。
2. 损失函数：引入 Dice Loss 或 Focal Loss 来应对类别不平衡问题（如细长电缆与大面积背景的比例失衡）。
3. 多模态融合优化：尝试在 Decoder 阶段引入更深层的跨模态注意力机制。
