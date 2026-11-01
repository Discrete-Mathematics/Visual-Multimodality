# SelectiveMine-F 结构说明

这四份文件是同一篇说明的 Markdown、TeX、Word 和 PDF：

- `SelectiveMineF_architecture.md`
- `SelectiveMineF_architecture.tex`
- `SelectiveMineF_architecture.docx`
- `SelectiveMineF_architecture.pdf`

整理日期是 2026-10-29。对象是 `selection=utility`、`fusion=logits` 的 SelectiveMine，测试权重是 selective_ftrans 第 7 轮。几何只加在融合解码器的对数几率上；特征级 relay 不参与这条 F 导出的前向。

## 测试集

FTransUNet 官方测试列表 788 张，16 类协议（标签 0 计入，255 忽略），输入 768×896。

| 设置 | PA | MPA | mIoU |
| --- | --- | --- | --- |
| SelectiveMine-F，六视图测试时增强 | 86.04 | 75.73 | 62.61 |
| CMX，Table 5 引用，不是本目录重跑 | 86.30 | 72.72 | 61.83 |

MPA 与 mIoU 高于该引用，PA 低 0.26 个百分点。正文按数据流说明每个方块的来源和计算，并给出与源码一致的训练设置。权重文件不在本目录；selective_ftrans 的 `best.pth` 在 Release [weights-20261025](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261025)。
