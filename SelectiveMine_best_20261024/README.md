# SelectiveMine 2026-10-24 开发模型

交接包在 `SelectiveMine_best_20261024/`。实验结果、加载方式和限制见该目录的 `README.md`。

主指标是开发验证集上的 16 类 mIoU 59.47%（第 99 / 120 轮）。这 314 张图用于选模，不是官方测试集。推理权重约 227 MB，在 Release [weights-20261025](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261025)，附件名是 `SelectiveMine_best_inference.pth`。
