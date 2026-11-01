# SelectiveMine 2026-10-30 继续优化

从 selective_ftrans 第 7 轮继续训练。划分是 FTransUNet 的 `train.txt` / `val.txt` / `test.txt`（1595 / 788 / 788），图像不在本包里。检查点只按验证集保存，测试集在选择规则冻结之后才读取。16 类协议把标签 0 计入，255 忽略。论文全文在 `MineSeg/v2/paper_20261030/`。文件清单见 `SelectiveMine_opt_20261030_README.txt`。

CMX 的 Table 5 引用是 PA 86.30，MPA 72.72，mIoU 61.83，不是这次重测。

## 测试集

主运行 selective_pa3：12 轮，768×896，batch 4，初始学习率 6e-6，种子 2026。保存规则 select-pa：验证集 16 类 mIoU 不低于 60.80% 且 MPA 不低于 72.00% 的轮次中，保留 16 类 PA 最高的一轮。被保存的是第 12 轮。该轮验证集是 PA 87.17，MPA 72.49，mIoU 61.51。

| 设置 | PA | MPA | mIoU | 相对 CMX |
| --- | --- | --- | --- | --- |
| selective_ftrans 第 7 轮，单尺度 | 85.93 | 75.46 | 62.42 | -0.37 / +2.74 / +0.59 |
| selective_pa3 第 12 轮，单尺度 | 86.13 | 74.52 | 62.79 | -0.17 / +1.80 / +0.96 |
| selective_pa3 第 12 轮，六视图 | 86.34 | 74.87 | 63.08 | +0.04 / +2.15 / +1.25 |

单尺度记录是 `MineSeg/v2/runs/selective_pa3/official_test.json`。六视图尺度是事先指定的 0.75、1.0、1.25，每个尺度再水平翻转，记录是 `MineSeg/v2/runs/selective_pa3_tta/official_test.json`。六视图与单尺度不是同一协议。按 16 类计算，六视图三项都高于所引用的 CMX；单尺度的 PA 仍低 0.17 个百分点。

报告用的主权重是 `MineSeg/v2/runs/selective_pa3/best.pth`。它和同目录的 `epoch_12.pth`、`best_pa.pth`、`last.pth` 字节不同，不能互相替换。

## 没有超过这个单尺度 PA 的尝试

- selective_pa4 加大像素负对数似然和背景权重。第 1 轮验证集 87.03 / 71.06 / 61.19，低于起点 87.17 / 72.49 / 61.51，随即停止，没有读测试集。
- 验证集上扫描分辨率和背景偏置，选中 800×928、背景偏置 0.10。随后只测这一格：测试集 86.10 / 74.11 / 62.57，PA 低于 86.13。
- 第 7、9、10、11、12 轮的概率平均没有在验证集上超过第 12 轮单模型的 PA，没有为此再读测试集。
- selective_pa5 把学习率降到 6e-7。第 1 轮验证集 87.11 / 71.70 / 61.35，PA 没有高于起点，没有读测试集。

## 权重不在 Git 里

下面这些文件都超过 100 MB，在 Release [weights-20261101](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261101)。`config.json`、`history.jsonl` 和测试 JSON 已随代码提交。

| 本目录中的路径 | Release 附件 |
| --- | --- |
| `MineSeg/v2/runs/selective_pa3/best.pth` | `selective_pa3_best.pth` |
| `MineSeg/v2/runs/selective_pa3/best_pa.pth` | `selective_pa3_best_pa.pth` |
| `MineSeg/v2/runs/selective_pa3/last.pth` | `selective_pa3_last.pth` |
| `MineSeg/v2/runs/selective_pa3/epoch_01.pth` … `epoch_12.pth` | `selective_pa3_epoch_01.pth` … `selective_pa3_epoch_12.pth` |
| `MineSeg/v2/runs/selective_pa4/best_pa.pth` | `selective_pa4_best_pa.pth` |
| `MineSeg/v2/runs/selective_pa4/last.pth` | `selective_pa4_last.pth` |
| `MineSeg/v2/runs/selective_pa5/best_pa.pth` | `selective_pa5_best_pa.pth` |
| `MineSeg/v2/runs/selective_pa5/last.pth` | `selective_pa5_last.pth` |

`MineSeg/v2/runs/selective_ftrans/best.pth` 与 Release [weights-20261025](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261025) 的 `MUSeg_selective_ftrans_best.pth` 是同一个文件（SHA256 `9159d3d4187365d9153c2c6cd9bfa27f8559d5fa06216d30d9f6ea015ef7a885`），这次不再重复上传。
