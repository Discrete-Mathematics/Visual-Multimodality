# 井下 RGB-D 语义分割中的深度边缘双向融合

**英文题目：** Depth-Edge Bidirectional Fusion for Underground RGB-D Semantic Segmentation

## 摘要

井下图像光照差、粉尘重，线缆、管路和轨道在彩色图里往往只占很细的一条，但在深度图上有清楚的几何边界。本文在 MUSeg 的既定划分上研究 RGB-D 语义分割。划分来自 `FTransUNet/train.txt`、`val.txt` 和 `test.txt`，分别有 1595、788 和 788 张图，覆盖 6 个矿井、15 个前景类。原始标签 0 视为未标注，不参与损失和指标；标签 1 到 15 对应 15 个类别。

本文的模型 DEBFNet 使用 ImageNet 预训练的 ResNet-50 提取彩色特征，另用一个轻量编码器读取对数深度和深度边缘。两侧特征在四个尺度上做双向门控：深度边缘用来加强彩色特征里的几何边界，彩色特征用来抑制不可靠的深度响应，融合结果以残差形式加回彩色特征。解码器是轻量 FPN，并带一个边界辅助头。

在完全相同的训练设置下，验证集 mIoU 为：只使用彩色 32.04%，早期拼接融合 36.61%，只做深度边缘空间门控 34.02%，完整 DEBFNet 39.86%。深度本身带来 4.57 个百分点，双向融合比早期拼接再高 3.25 个百分点，提升主要来自轨道区域、锚固设备、管路和线缆。去掉边界头后，40 个 epoch 的验证 mIoU 从 38.04% 降到 37.36%。把稀有类贴到别的训练图上没有提高总体精度（38.17%）。模型只根据验证集选择。选定 DEBFNet 后，在验证集上确认水平翻转测试时增强有益（40.61%），再在测试集上评估一次，mIoU 为 40.14%，像素准确率为 63.74%。

这个结果说明深度和双向融合在该划分上是有效的，但工具材料、容器等长尾类仍然很差，也没有在同一划分上复现 CMX、DFormer 等已发表模型。因此本文记录的是一次受控实验，而不是已经达到 CCF-A 接收标准的完整投稿。

**关键词：** RGB-D 语义分割；井下场景；深度边缘；双向门控；MUSeg

## Abstract

Underground frames are dark and dusty. Cables, tubes and rails are thin in RGB but geometrically sharp in depth. We study RGB-D semantic segmentation on the fixed MUSeg lists `train.txt`, `val.txt` and `test.txt` (1595 / 788 / 788 images, 15 foreground classes, label 0 ignored). DEBFNet pairs an ImageNet ResNet-50 with a light depth-edge encoder and fuses the two streams by bidirectional gates at four scales. On the validation set the mean IoU is 32.04% for RGB only, 36.61% for early fusion, 34.02% for a depth-edge spatial gate without a depth encoder, and 39.86% for DEBFNet. Flip test-time augmentation was selected on validation (40.61%) and then applied once to the test set, giving 40.14% mIoU and 63.74% pixel accuracy. The gain is real on this split, especially for rails, anchoring equipment, tubes and cables. Long-tail classes remain weak, and published RGB-D models were not reimplemented, so the study is not a complete CCF-A submission.

## 1 引言

矿井视觉需要同时看清设备和细长结构。彩色图能区分设备外观，但灯光不均、反光和粉尘会把线缆、管路吞进背景。深度图没有颜色，却能给出遮挡边界和轨道、锚杆这类几何结构。两者互补，所以井下分割适合做成 RGB-D，而不是只靠彩色。

MUSeg 把图像按矿井放在 `01-Mine` 到 `06-Mine`，每个样本有彩色图、16 位深度图和语义标签。`Visual-Multimodality/SSRS-main/FTransUNet` 里的三个名单已经规定了训练、验证和测试，不能再随机重划。本文严格读取这三个文件。名单中的 3171 个样本全部能在磁盘上找到，没有缺失。

已有 RGB-D 分割常用两种做法。一种把深度当成第四个通道，从第一层就和彩色拼在一起。另一种用两个编码器，再用注意力或门控交换信息，例如 SA-Gate 和 CMX。井下数据还有两个具体困难：细结构依赖深度不连续，而不是依赖深度的绝对语义；人员、工具、容器、救援设备的像素很少。本文针对第一点做双向融合，并用实验检查第二点能不能靠简单的粘贴增强解决。

贡献有三点：

1. 给出 MUSeg 这三个名单上的可复现训练、验证和测试流程，包括标签 0 的忽略方式和深度归一化。
2. 提出 DEBFNet：深度边缘编码器加双向门控，并在同一骨干、同一损失、同一数据增强下和彩色基线、早期融合、纯边缘门控对比。
3. 只在验证集上选择模型和测试时增强，测试集只评估一次。报告类别和矿井差异，也报告没有生效的尝试。

## 2 相关工作

彩色语义分割从全卷积网络开始，后来普遍使用 ImageNet 预训练骨干和特征金字塔解码。本文的骨干是 ResNet-50，解码器是简化的 FPN，这部分不是新的。

RGB-D 分割的差别主要在融合位置。早期融合改第一层卷积，让深度从底层进入网络，实现简单，也是本文最重要的对照。SA-Gate 用分离—聚合门在两个方向传递特征。CMX 用跨模态校正和融合模块处理深度、热红外等多种辅模态。DFormer 则在 ImageNet 的图像—深度对上预训练，减轻“彩色骨干读深度”的错配。FTransUNet 面向遥感，在浅层卷积和深层 Transformer 上做多级融合；MUSeg 的名单放在该代码目录中，但官方 FTransUNet 实验的是 Vaihingen 和 Potsdam，不是这 15 类矿井数据。

本文没有把上述模型重新训练到 MUSeg 上，因此不能声称超过它们。DEBFNet 借鉴的是双向校正这一思路，但针对井下深度：辅分支的输入是对数深度和 Sobel 边缘，而不是把深度复制成三通道后送进另一个彩色骨干。

## 3 方法

### 3.1 数据

样本编号的前两位是矿井号。编号 `04-01-01-0056-230808080243-12-99` 对应

- `04-Mine/Image/<编号>.jpg`
- `04-Mine/Depth/<编号>.png`
- `04-Mine/Label/<编号>_label.png`

抽查样本的分辨率是 1082×932。深度是 16 位单通道，数值大约从 0 到一万多。训练时先做 `log(1+d) / log(1+10000)` 并截断到 `[0,1]`。多边形标注里的类别名是英文。像素值 1 到 15 依次对应：人员、线缆、管路、指示标识、电气设备、电子设备、采矿设备、轨道区域、支护设备、门、工具与材料、救援设备、容器、金属构件、锚固设备。像素值 0 在训练集中约占一半，语义上是未标注背景。映射后类别下标为 0 到 14，未标注位置记为 255，交叉熵忽略 255。

六个矿井的图像数并不均衡。训练集中 1 到 6 号矿井分别是 341、222、309、207、50、466 张；验证集是 190、135、118、116、34、195 张；测试集是 223、124、122、94、31、194 张。5 号矿井样本很少，按矿井统计时方差会大。

### 3.2 网络

彩色分支是 ImageNet-1K 预训练的 ResNet-50。四个阶段的通道数是 256、512、1024、2048，先用 1×1 卷积压到 128、128、256、256，减轻后续融合的参数量。若直接在 2048 通道上做 3×3 双向门控，参数会超过三亿，1595 张训练图撑不住，因此压缩是必要的。

深度分支的输入是两通道：归一化深度，以及深度的 Sobel 梯度幅值。梯度在每张图内部按最大值归一化。编码器只有卷积、批归一化和 ReLU，输出与压缩后的彩色特征对齐。

在每个尺度上，双向门控的计算是：把彩色特征 `Fr` 和深度特征 `Fd` 拼接，分别预测两个空间—通道门。彩色侧乘上 `(1 + 门)`，用来按深度边缘加强边界；深度侧乘上自己的门，用来压掉不可靠响应。两路再拼起来投影回原通道，并以残差加回 `Fr`。残差让网络可以退回到纯彩色特征，避免深度噪声一定被写进主干。

解码器把四个尺度加到 128 通道的 FPN 上。边界头在最高分辨率特征上预测一张边界图，特征乘上 `(1 + sigmoid(边界))` 后再分类。logits 双线性上采样到输入大小。

三个对照使用同一个解码器和同一种损失：

- RGB：不读深度。
- 早期融合：第一层卷积改为 4 通道，新增通道用原来三个彩色通道的均值初始化。
- 边缘门控：没有深度编码器，只把深度和边缘下采样后去门控彩色特征。

DEBFNet 约 35.5M 参数，RGB 和早期融合约 26.7M，边缘门控约 28.3M。

### 3.3 损失和训练

损失是加权交叉熵、Dice 和边界二元交叉熵：

`L = L_ce + 0.5 L_dice + 0.4 L_bd`

类别权重用训练集前景像素的中位频率平衡，再截断到 `[0.25, 6]` 并归一化，避免人员类权重过大。边界由标签的 3×3 膨胀和腐蚀之差得到，未标注像素不产生边界监督。边界像素上的交叉熵额外乘 3。

输入缩放到 640×640。训练时随机缩放 0.75 到 1.25、随机裁剪、水平翻转，并对彩色做亮度和对比度抖动。使用深度的模型还会随机挖掉一块深度、加上少量深度噪声。优化器是 AdamW，头部分支学习率 `3×10^-4`，骨干是其 0.1 倍，权重衰减 `1×10^-4`。前 3 个 epoch 线性预热，之后余弦退火，共 60 个 epoch，批大小 16，随机种子 42。每个 epoch 约 20 秒。按验证集 mIoU 保存最优权重。测试集不参与选择。

稀有类粘贴是一次额外尝试。从训练集裁出人员、门、工具与材料、救援设备和容器共 322 块，训练时以 0.8 的概率贴 1 到 2 块到原图上，彩色、深度和标签一起贴。无边界模型只训练 40 个 epoch，用来和 DEBFNet 第 40 个 epoch 的成绩对比。

## 4 实验

### 4.1 验证集对照

下表是验证集、无测试时增强的 mIoU 和像素准确率。除无边界模型外都训练了 60 个 epoch。

| 模型 | 参数量 | 验证 mIoU | 像素准确率 |
|---|---:|---:|---:|
| 只使用彩色 | 26.7M | 32.04% | 56.62% |
| 早期融合 | 26.7M | 36.61% | 62.33% |
| 深度边缘空间门控 | 28.3M | 34.02% | 59.13% |
| DEBFNet | 35.5M | 39.86% | 66.05% |
| DEBFNet + 稀有类粘贴 | 35.5M | 38.17% | 64.55% |
| DEBFNet 去掉边界头（40 epoch） | 35.5M | 37.36% | 63.14% |

DEBFNet 在第 40 个 epoch 的验证 mIoU 是 38.04%。因此在相同的 40 epoch 预算下，边界头大约贡献 0.7 个百分点，不是主要增益。主要增益来自融合方式：早期融合比彩色高 4.57 个百分点，DEBFNet 比早期融合再高 3.25 个百分点。只做空间门控反而不如早期融合，说明辅分支需要自己的特征，而不能只拿边缘图去乘彩色特征。

稀有类粘贴没有超过不加粘贴的 DEBFNet。轨道区域的 IoU 从 66.7% 降到 61.7%，总体 mIoU 降到 38.17%。把物体贴到几何上不匹配的位置，会破坏深度和外观的对应关系。

### 4.2 类别

下表是验证集各类 IoU（%），无测试时增强。

| 类别 | 彩色 | 早期融合 | 边缘门控 | DEBFNet |
|---|---:|---:|---:|---:|
| 人员 | 10.2 | 17.4 | 13.4 | 19.3 |
| 线缆 | 38.9 | 45.2 | 40.7 | 48.3 |
| 管路 | 35.6 | 42.7 | 37.8 | 48.8 |
| 指示标识 | 65.8 | 66.2 | 66.9 | 68.3 |
| 电气设备 | 27.2 | 33.0 | 26.6 | 32.0 |
| 电子设备 | 30.0 | 32.9 | 29.8 | 35.3 |
| 采矿设备 | 34.5 | 42.1 | 38.2 | 46.2 |
| 轨道区域 | 50.2 | 56.8 | 55.4 | 66.7 |
| 支护设备 | 37.4 | 40.8 | 38.0 | 45.6 |
| 门 | 28.0 | 29.8 | 30.7 | 36.7 |
| 工具与材料 | 7.9 | 14.6 | 10.9 | 11.9 |
| 救援设备 | 20.2 | 20.7 | 21.5 | 24.0 |
| 容器 | 11.7 | 9.7 | 11.3 | 11.2 |
| 金属构件 | 27.1 | 33.1 | 29.5 | 33.2 |
| 锚固设备 | 56.0 | 64.2 | 59.8 | 70.2 |

几何类的提升和动机一致。相对早期融合，DEBFNet 的轨道区域高 9.9 个百分点，锚固设备高 6.0 个百分点，管路高 6.1 个百分点，线缆高 3.1 个百分点。指示标识本身已经较高，融合带来的增幅很小。工具与材料上早期融合（14.6%）反而高于 DEBFNet（11.9%），容器三类模型都在 10% 上下。长尾外观类不是这个融合模块能解决的。

### 4.3 测试集

水平翻转再平均 logits，在验证集上把 DEBFNet 从 39.86% 提高到 40.61%。因此测试时采用这一增强，并且测试集只跑这一次。

测试集 mIoU 为 40.14%，像素准确率为 63.74%。与验证集加增强后的 40.61% 接近，没有出现明显的选择过拟合。各类 IoU（%）为：人员 32.7，线缆 46.8，管路 44.9，指示标识 67.1，电气设备 34.4，电子设备 32.4，采矿设备 41.1，轨道区域 65.6，支护设备 44.2，门 39.6，工具与材料 10.0，救援设备 23.8，容器 16.1，金属构件 34.6，锚固设备 68.8。

人员类在测试集上明显高于验证集（32.7% 对 19.3%），说明这一类在两个子集里的出现方式不一样，单次划分下的类别 IoU 不稳定。工具与材料在测试集只有 10.0%。

按矿井统计的测试 mIoU / 像素准确率：1 号矿 34.34% / 60.05%，2 号矿 43.83% / 64.40%，3 号矿 32.36% / 56.74%，4 号矿 48.74% / 77.69%，5 号矿 34.30% / 77.26%，6 号矿 39.33% / 77.09%。3 号矿最差，4 号矿最好。5 号矿测试图只有 31 张，像素准确率高但 mIoU 低，说明少数大类占了大部分像素。

定性图见 `runs/debf/eval_test_tta/vis/`。图中从左到右是彩色、深度、标注和预测。标注里的黑色是未标注，不计入 mIoU；预测是稠密的，所以在未标注区域也会涂上颜色。在有标注的位置，大块支护和线缆大致能对上，但预测边界更碎，细类会粘到邻近结构上。

### 4.4 局限

第一，没有在同一名单上训练 CMX、DFormer、SA-Gate 或 FTransUNet。3.25 个百分点是相对早期融合的增益，不是相对这些已发表模型的增益。第二，只使用了 MUSeg 这一个数据集和这一套划分，没有 NYU Depth v2 或 SUN RGB-D。第三，工具、容器和人员仍然差，粘贴增强没有修好。第四，40% 左右的 mIoU 距离可直接部署还有明显差距。基于这四点，本稿不能当作已经满足 CCF-A 接收标准的论文。

## 5 结论

在 MUSeg 的固定三分名单上，深度比只看彩色更有用，双向的深度边缘融合又比把深度拼进第一层更好。验证集上，彩色、早期融合和 DEBFNet 的 mIoU 分别是 32.04%、36.61% 和 39.86%。按验证集选定翻转增强后，测试集 mIoU 为 40.14%。增益集中在轨道、锚固设备、管路和线缆。长尾类、跨矿井差异，以及和已发表 RGB-D 模型的同协议对比，仍然没有完成。

## 参考文献

1. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CVPR, 2016.
2. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. CVPR, 2015.
3. X. Chen, K.-Y. Lin, J. Wang, W. Wu, C. Qian, H. Li, and G. Zeng. Bi-directional cross-modality feature propagation with separation-and-aggregation gate for RGB-D semantic segmentation. ECCV, 2020.
4. J. Zhang, H. Liu, K. Yang, X. Hu, R. Liu, and R. Stiefelhagen. CMX: Cross-modal fusion for RGB-X semantic segmentation with transformers. IEEE Transactions on Intelligent Transportation Systems, 2023. doi:10.1109/TITS.2023.3300537.
5. B. Yin, X. Zhang, Z. Li, L. Liu, M.-M. Cheng, and Q. Hou. DFormer: Rethinking RGBD representation learning for semantic segmentation. ICLR, 2024.
6. X. Ma, X. Zhang, M.-O. Pun, and M. Liu. A multilevel multimodal fusion transformer for remote sensing semantic segmentation. IEEE Transactions on Geoscience and Remote Sensing, 62:1–15, 2024. doi:10.1109/TGRS.2024.3373033.

## 复现

代码和权重在 `/root/autodl-tmp/MineSeg`。最优权重是 `runs/debf/best.pth`，对应第 59 个 epoch。测试指标在 `runs/debf/eval_test_tta/metrics.json`。训练命令：

```
python3 train.py --mode debf --epochs 60 --batch-size 16 --size 640 --workers 8
python3 test.py --checkpoint runs/debf/best.pth --split test --tta
```
