# 2026-10-29 实验汇报

幻灯片源文件是 `museg_report.tex`（ctexbeamer）。本目录没有编译好的幻灯片 PDF。`selectivemine_architecture.pdf` 和 `selectivemine_architecture.png` 是结构图，`draw_architecture.py` 用来绘制，`_crop_*.png` 是切图中间文件。

比较对象仍是 MUSeg Table 5 引用的 CMX：PA 86.30，mPA 72.72，mIoU 61.83。对齐用的是 16 类协议。15 类像素准确率大约在 80 附近，不能和 86.30 放在一起比较。

## 幻灯片里的测试表

单位是百分比。C 的分辨率是 640×768，F 是 768×896。

| 方法 | 协议 | 测试时增强 | PA | mPA | mIoU |
| --- | --- | --- | --- | --- | --- |
| SelectiveMine-F | 16 类 | 六视图 | 86.04 | 75.73 | 62.61 |
| SelectiveMine-F | 16 类 | 关闭 | 85.93 | 75.46 | 62.42 |
| SelectiveMine-C | 16 类 | 六视图 | 86.10 | 76.64 | 62.41 |
| DEBF-B5，第 14/30 轮 | 15 类 | 六视图 | 71.71 | 66.70 | 50.35 |

六视图把 F 的 16 类 mIoU 从 62.42 提到 62.61，没有补上 PA 相对 CMX 的缺口。本目录已测试结果里，16 类 mIoU 最高的是 F 的 62.61。

selective_ftrans 的 `best.pth` 按 15 类验证 mIoU 保存，是第 7 轮。幻灯片记录该轮验证集为：15 类 mIoU 63.42%；16 类 PA 86.84，mPA 73.49，mIoU 61.22。更晚轮次的 16 类更高，但 15 类没有超过第 7 轮，那些权重没有保存。测试数字对应该文件。

这次汇报早于 2026-10-30 的 selective_pa3，不包含后来的 86.13 / 74.52 / 62.79。
