# SelectiveMine 最佳开发模型（2026-10-24）

模型是 SelectiveMine，`selection=utility`，`fusion=logits`，RGB 编码器为 MiT-B2。任务是 MUSeg 矿井 RGB-D 的 16 类语义分割。完整交接说明、扰动定义和复现步骤在 `README.txt`。

权重是 120 轮训练结束后，按普通开发集总体 mIoU 选中的第 99 轮推理导出。不含优化器和随机状态，不能续训。开发集从官方训练集内部按地点分组：1281 张开发训练图、314 张开发验证图，地点组隔离，种子 2026。这 314 张用于选模，不是独立官方测试集。

评估：480×640，batch 8，单次缩放，无测试时增强，CUDA bf16。同一个第 99 轮权重没有按不同条件重新选模。

| 条件 | mIoU |
| --- | --- |
| 普通输入 | 59.47% |
| 人工深度孔洞 | 58.29% |
| RGB 人工暗化 | 58.04% |
| 整幅深度完全缺失 | 43.86% |
| 深度高斯噪声 | 43.87% |

原始 JSON 在 `results/eval_*.json`。这些合成扰动不等于真实传感器故障，也不能写成 SOTA，或直接拿去和不同划分、训练预算、评估协议下的论文数字比较。

## 权重不在 Git 里

`weights/best_inference.pth` 是 238,273,211 字节，约 227 MB，超过 GitHub 单文件 100 MB 限制。请从 Release [weights-20261025](https://github.com/Discrete-Mathematics/Visual-Multimodality/releases/tag/weights-20261025) 下载 `SelectiveMine_best_inference.pth`，保存为 `weights/best_inference.pth`。`weights/export.json` 已在仓库中，必须与权重放在同一目录。

SHA256：`f7bd8de044ec8fd6e747a670f08de4dd2955d7f0c831b7639244de42d3a78cef`。

加载使用 `source/tools/selective_inference_bundle_20261023.py` 的 `load_for_inference`。不要只做 `torch.load` 后忽略预处理或 `mix=1`。
