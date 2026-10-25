MUSeg 实验打包
日期：2026-10-25
未重新训练，未重新测试。数值来自已有 JSON。

超过 CMX（论文 Table 5：PA 86.30，MPA 72.72，mIoU 61.83）的测试集模型
协议：16 类，标签 0 计入、255 忽略。划分：本包 splits/。

1. models/selective_ftrans/best.pth
   SelectiveMine-F，768x896，六视图测试时增强
   PA 86.04，MPA 75.73，mIoU 62.61
   记录：records/v3/runs/selective_ftrans_test/test_tta.json
   单尺度对照 PA 85.93，MPA 75.46，mIoU 62.42
   记录：records/v2/runs/selective_ftrans_test_single/official_test.json

2. models/selective_complement/best.pth
   SelectiveMine-C，640x768，六视图测试时增强
   PA 86.10，MPA 76.64，mIoU 62.41
   记录：records/v3/runs/selective_complement/test_tta.json

这两项的 MPA 和 mIoU 高于 CMX，也高于 DFormer、SA-Gate、CMNeXt。PA 分别低 0.26 和 0.20 个百分点。DEBF 的测试 mIoU 为 50.35，没有放权重。

划分行数：train 1595，val 788，test 788。
来源：Visual-Multimodality/SSRS-main/FTransUNet 的 train.txt、val.txt、test.txt。

目录
- models/ 上述两个检查点及对应 config.json
- parameters/ 各次运行的 config.json 与 args.json
- records/ 训练曲线、测试 JSON、结果汇总、PROGRESS.txt
- scripts/ 训练与测试入口
- utils/ 实验中编写的辅助模块。不含 PyTorch 等第三方库
- code/ 上述代码的原始目录，便于按原相对路径导入
- paper/ paper.md、paper.tex、paper.docx、paper.pdf、results.json
- splits/ train.txt、val.txt、test.txt

权重不在 Git 里
models/selective_ftrans/best.pth 与 models/selective_complement/best.pth 各约 227 MB，超过 GitHub 单文件 100 MB 限制，没有提交。
请从本仓库 Release「weights-20261025」下载对应附件，按 Release 说明放回上述路径。同目录的 config.json 已在仓库中。
parameters/ 里的 config.json 与 args.json 都远小于 100 MB，已随代码提交。
