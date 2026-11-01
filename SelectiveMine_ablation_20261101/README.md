# selective_pa3 冻结权重消融（2026-11-01）

权重没有重新训练。评测时从已经训好的 selective_pa3 第 12 轮权重里去掉一条路径；融合专家改变时，路由器按新的两路对数概率重算。论文在 `MineSeg/v2/paper_ablation/`，分数在 `MineSeg/v2/runs/selective_pa3_ablation/`。目录清单见 `SelectiveMine_ablation_20261101_README.txt`。

划分是本包 `Visual-Multimodality/SSRS-main/FTransUNet/` 的 `train.txt`、`val.txt`、`test.txt`，1595 / 788 / 788 张，两两交集为 0。图像不在包里。验证集和测试集都用 768×896、单尺度、无测试时增强。16 类协议把标签 0 计入，255 忽略。训练集只用来说明拟合范围，没有在训练集上算消融分数。

加载的检查点是 `MineSeg/v2/runs/selective_pa3/best.pth`。它与 `selectivemine1030/MineSeg/v2/runs/selective_pa3/best.pth` 字节相同。完整模型在测试集上的 16 类 PA、MPA、mIoU 是 86.13%、74.52%、62.79%，与 `official_test.json` 的差在 1e-5 量级。CMX Table 5 的引用是 86.30%、72.72%、61.83%，不是这次重测。

这些差值表示已选权重里某条路径被拿掉之后的变化，不是缺模块之后重新训练的结果。去掉 HHA 或深度时，对应输入换成零张量，落在训练分布之外。

## 测试集 788 张

差值相对同一张表里的完整模型，单位是百分点。

| 变体 | PA | MPA | mIoU | mIoU 差 |
| --- | --- | --- | --- | --- |
| 完整模型 | 86.13 | 74.52 | 62.79 | 0.00 |
| 只用融合专家 | 86.09 | 74.41 | 62.95 | +0.16 |
| 去掉 relay | 86.13 | 74.52 | 62.79 | 0.00 |
| 去掉深度门控 | 85.94 | 73.83 | 62.65 | -0.14 |
| 常数 0.5 混合 | 84.86 | 72.17 | 58.94 | -3.85 |
| 只用外观解码器 | 75.76 | 60.17 | 44.07 | -18.72 |
| 去掉 HHA | 58.24 | 25.25 | 21.51 | -41.28 |
| 去掉深度和 HHA | 57.94 | 24.39 | 20.72 | -42.07 |

测试集上 PA、MPA 和 16 类 mIoU 下降最多的都是去掉深度和 HHA。去掉 relay 与完整模型在计入 16 类矩阵的像素上预测完全相同。路由器换成常数 0.5 时，mIoU 下降 3.85 个百分点。验证集 788 张上，下降最多的也是去掉深度和 HHA（mIoU 差 -41.60）。验证集曾经用来选择第 12 轮，所以验证分数不是事先锁定后才打开的测试。

## 权重不在 Git 里

`best.pth` 约 227 MB。Release [weights-20261101](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261101) 的附件 `selective_pa3_best.pth` 同时对应：

- `SelectiveMine_ablation_20261101/MineSeg/v2/runs/selective_pa3/best.pth`
- `selectivemine1030/MineSeg/v2/runs/selective_pa3/best.pth`

SHA256：`2e3cfbf3c2fb64919db703874cfe24251bfa8da7d1cc6aab7a356f5fd2f9ec91`。
