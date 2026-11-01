SelectiveMine 2026-11-01 selective_pa3 冻结权重消融的代码、论文、配置和实验记录。

划分
  train/val/test 使用
  Visual-Multimodality/SSRS-main/FTransUNet/train.txt
  val.txt、test.txt
  三个名单在压缩包里，图像数是 1595、788、788，两两交集为 0。
  图像本身不在这个压缩包里，路径是 /root/autodl-tmp/MUSeg/flat。

代码
  MineSeg/v2/ablate_selective.py
      本次消融入口。权重不重新训练，评测时去掉一条已经训练好的路径。
  MineSeg/v2/run_ablation_then_shutdown.sh
      跑消融、写论文、发结果邮件，然后关机。
  MineSeg/v2/finetune_selective.py
      消融脚本从这里取 prepare_model、seed_everything 和 PAPER_CLASSES。
  MineSeg/v2/engine.py
      finetune_selective.py 导入的 Lovász 损失。
  MineSeg/v2/paper_ablation/build_paper.py
      从 ablation.json 生成四份内容相同的论文。
  SelectiveMine_best_20261024/source
      模型、数据读取和推理加载。不含 __pycache__。

论文（同一份内容的四个版本）
  MineSeg/v2/paper_ablation/paper.md
  MineSeg/v2/paper_ablation/paper.tex
  MineSeg/v2/paper_ablation/paper.docx
  MineSeg/v2/paper_ablation/paper.pdf
  同目录的 results.json 是生成论文时读到的摘要。

权重
  MineSeg/v2/runs/selective_pa3/best.pth
      消融实际加载的检查点，第 12 轮。
  同目录的 epoch_01.pth 到 epoch_12.pth、best_pa.pth、last.pth 这次没有加载，不在压缩包里。
  构造网络时 prepare_model 还会先读
  /root/autodl-tmp/SelectiveMine_best_20261024/weights/best_inference.pth
  和同目录的 export.json，读完后被 best.pth 的 state_dict 覆盖。这两份不在压缩包里。

配置和记录
  MineSeg/v2/runs/selective_pa3/config.json
  MineSeg/v2/runs/selective_pa3/history.jsonl
  MineSeg/v2/runs/selective_pa3/train.log
  MineSeg/v2/runs/selective_pa3/test.log
  MineSeg/v2/runs/selective_pa3/official_test.json
      消融用来核对完整模型的已发表单尺度测试。
  MineSeg/v2/runs/selective_pa3_ablation/ablation.json
  MineSeg/v2/runs/selective_pa3_ablation/ablation_val.json
  MineSeg/v2/runs/selective_pa3_ablation/ablation_test.json
  MineSeg/v2/runs/selective_pa3_ablation/ablation.log
      验证集和测试集的消融分数。训练名单只做划分核对，没有在训练集上算分。
