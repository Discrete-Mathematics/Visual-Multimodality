# lib/datasets/build_dataloader.py
from torch.utils.data import DataLoader
from pathlib import Path


def build_dataloader(args, mode='train'):
    """
    极简版本：只负责把「自定义 DATASET」包装成 DataLoader
    args 必须字段：
        args.dataset == 'custom'
        args.data_root   – 数据根目录
        args.train_list / val_list / test_list – 存放 ID 的 txt
        args.batch_size
        args.num_workers
        args.num_gpus    – 仅用于判断是否分布式（这里默认 False）
    """
    if args.dataset != 'custom':
        raise NotImplementedError(f"only 'custom' dataset is implemented here, got {args.dataset}")

    # 1. 读取对应 split 的 ID 列表
    split_file = {'train': args.train_list,
                  'val'  : args.val_list,
                  'test' : args.test_list}[mode]
    with open(Path(args.data_root) / split_file, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]

    # 2. 构造数据集（直接用你写好的 DATASET 类）
    from .dataset import DATASET  # 假设本文件同级目录有 dataset.py
    dataset = DATASET(
        data_root=args.data_root,
        mode=mode,
        # 如果你想把超参暴露到 args，继续往下加
        # window_size=(args.out_height, args.out_width),
        # scales=args.scales,
    )

    # 3. 返回标准 DataLoader
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(mode == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )