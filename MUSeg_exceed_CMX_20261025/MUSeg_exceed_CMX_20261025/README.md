# 2026-10-25：超过 CMX 的两个测试检查点

本目录没有重新训练，也没有重新测试。数字来自已有 JSON。更完整的文件清单在 `README.txt`。

协议：16 类，标签 0 计入、255 忽略。划分在 `splits/`，来自 `SSRS-main/FTransUNet` 的 `train.txt`、`val.txt`、`test.txt`，行数是 1595、788、788。

CMX 论文 Table 5 的引用值是 PA 86.30，MPA 72.72，mIoU 61.83。

| 检查点 | 模型 | 设置 | PA | MPA | mIoU | 记录 |
| --- | --- | --- | --- | --- | --- | --- |
| `models/selective_ftrans/best.pth` | SelectiveMine-F | 768×896，六视图 | 86.04 | 75.73 | 62.61 | `records/v3/runs/selective_ftrans_test/test_tta.json` |
| 同一权重 | SelectiveMine-F | 768×896，单尺度 | 85.93 | 75.46 | 62.42 | `records/v2/runs/selective_ftrans_test_single/official_test.json` |
| `models/selective_complement/best.pth` | SelectiveMine-C | 640×768，六视图 | 86.10 | 76.64 | 62.41 | `records/v3/runs/selective_complement/test_tta.json` |

这两项的 MPA 和 mIoU 高于 CMX，也高于该包记录中的 DFormer（85.63 / 70.37 / 59.74）、SA-Gate（84.77 / 71.34 / 59.38）和 CMNeXt（84.05 / 66.77 / 56.06）。PA 分别低 0.26 和 0.20 个百分点。DEBF 的测试 mIoU 为 50.35，没有放权重。

## 权重不在 Git 里

两个 `best.pth` 各约 227 MB，超过 GitHub 单文件 100 MB 限制。请从 Release [weights-20261025](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261025) 下载：

| Release 附件 | 放回路径 |
| --- | --- |
| `MUSeg_selective_ftrans_best.pth` | `models/selective_ftrans/best.pth` |
| `MUSeg_selective_complement_best.pth` | `models/selective_complement/best.pth` |

同目录的 `config.json` 已在仓库中。`parameters/` 里的 `config.json` 与 `args.json` 远小于 100 MB，也已提交。
