# MUSeg 实验材料

比较对象是 MUSeg 论文 Table 5 的 CMX（MiT-B2，RGB-D HHA）：PA 86.30，MPA 72.72，mIoU 61.83。这些数字是引用，不是本目录重新训练的结果。与 Table 5 对齐的是 16 类协议：标签 0 计入，只忽略 255。划分是 FTransUNet 的 `train.txt` / `val.txt` / `test.txt`，1595 / 788 / 788 张。

## 目录

| 目录 | 内容 |
| --- | --- |
| `MUSeg_exceed_CMX_20261025/` | 2026-10-25 打包的 SelectiveMine-F 与 SelectiveMine-C。测试记录、配置和论文在 Git 里。 |
| `SelectiveMineF_architecture_paper/` | SelectiveMine-F 的结构说明，对应 selective_ftrans 第 7 轮。 |
| `report_slides/` | 2026-10-29 的实验汇报幻灯片源文件。 |

## 已报告的测试集结果

| 模型 | 输入 | 测试时增强 | PA | MPA | mIoU |
| --- | --- | --- | --- | --- | --- |
| SelectiveMine-F | 768×896 | 六视图 | 86.04 | 75.73 | 62.61 |
| SelectiveMine-F | 768×896 | 关闭 | 85.93 | 75.46 | 62.42 |
| SelectiveMine-C | 640×768 | 六视图 | 86.10 | 76.64 | 62.41 |

F 和 C 的 MPA、mIoU 高于所引用的 CMX，PA 分别低 0.26 和 0.20 个百分点。两项没有同时超过 CMX。C 与 F 的分辨率不同，0.20 个 mIoU 差不能单独解释成补全模块的效果。DEBF-B5 的 15 类六视图测试 mIoU 是 50.35，没有附带权重。

两个约 227 MB 的 `best.pth` 超过 GitHub 单文件 100 MB 限制，没有进入 Git。下载和放回路径见 Release [weights-20261025](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261025)。

2026-10-30 的继续优化和 2026-11-01 的冻结权重消融在仓库根目录的 `selectivemine1030/` 与 `SelectiveMine_ablation_20261101/`。
