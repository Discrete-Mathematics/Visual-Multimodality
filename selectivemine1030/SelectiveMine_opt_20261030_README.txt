SelectiveMine 2026-10-30 这一轮优化的代码、论文、配置和参数。

划分
  train/val/test 使用
  /root/autodl-tmp/Visual-Multimodality/SSRS-main/FTransUNet/train.txt
  val.txt、test.txt
  数据集本身不在这个压缩包里，路径是 /root/autodl-tmp/MUSeg/flat。

代码
  MineSeg/v2/finetune_selective.py    训练和测试入口
  MineSeg/v2/sweep_pa3_prior.py       验证集上的分辨率和背景偏置扫描
  MineSeg/v2/ensemble_pa3.py          验证集上的检查点概率平均
  MineSeg/v2/engine.py                Lovász 损失
  SelectiveMine_best_20261024/source  模型、数据读取和 selective_loss

论文（同一份内容的四个版本）
  MineSeg/v2/paper_20261030/paper.md
  MineSeg/v2/paper_20261030/paper.tex
  MineSeg/v2/paper_20261030/paper.docx
  MineSeg/v2/paper_20261030/paper.pdf

参数
  selective_ftrans/best.pth           本轮起点，日志里的第 7 轮
  selective_pa3/epoch_01.pth … epoch_12.pth
  selective_pa3/best.pth              按保存规则留下的主权重，测试用的是这一份
  selective_pa3/best_pa.pth           验证集 16 类像素准确率最高的一轮
  selective_pa3/last.pth              第 12 轮结束时的最后一份
  selective_pa4/best_pa.pth 和 last.pth
  selective_pa5/best_pa.pth 和 last.pth
  以上文件内容互不相同，都保留在压缩包里。

配置和记录在对应 runs 目录的 config.json、history.jsonl、train.log 和测试 JSON。
