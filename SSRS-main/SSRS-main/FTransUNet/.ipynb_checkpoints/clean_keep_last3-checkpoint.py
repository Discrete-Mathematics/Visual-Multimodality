#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动清理训练结果，仅保留最近 3 个检查点与对应日志 / 配置 / 结果文件
用法:
    python clean_keep_last3.py  [--root ./train-experiment9] [--dry-run]
"""
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import time

# ---------- 工具 ----------
def extract_datetime(fname: str) -> datetime:
    """
    从文件名中提取时间戳，支持两种格式：
      experiment_20260108_143022.log
      checkpoint_epoch_95_20260108_143022.pth
    返回 datetime 对象；若无法解析则返回 datetime.min
    """
    # 1) experiment_YYYYMMDD_HHMMSS*
    m = re.search(r'_(\d{8}_\d{6})', fname)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    # 2) checkpoint_epoch_XX_YYYYMMDD_HHMMSS.pth
    m = re.search(r'_(\d{8}_\d{6})\.pth$', fname)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return datetime.min


def group_files_by_timestamp(files: List[Path]) -> List[Tuple[datetime, List[Path]]]:
    """
    将文件按时间戳分组，返回 [(dt, [file1, file2, ...]), ...] 且按 dt 降序
    """
    buckets = {}
    for f in files:
        dt = extract_datetime(f.name)
        buckets.setdefault(dt, []).append(f)
    return sorted(buckets.items(), key=lambda x: x[0], reverse=True)


def scan_target_files(root: Path):
    """
    扫描需要清理的文件：
      - checkpoints 目录下 *.pth
      - 根目录下 experiment_*.{log,json}
    返回列表[Path]
    """
    ckpt_dir = root / "checkpoints"
    files = []
    if ckpt_dir.exists():
        files.extend(ckpt_dir.glob("*.pth"))
    files.extend(root.glob("experiment_*.log"))
    files.extend(root.glob("experiment_*_config.json"))
    files.extend(root.glob("experiment_*_results.json"))
    return files


def delete_files(file_list: List[Path], dry: bool = True):
    for f in file_list:
        if dry:
            print(f"[dry-run] 删除: {f}")
        else:
            try:
                f.unlink()
                print(f"已删除: {f}")
            except Exception as e:
                print(f"删除失败: {f} -> {e}")


# ---------- 主逻辑 ----------
def main():
    parser = argparse.ArgumentParser(description="保留最近 3 组训练文件，其余删除")
    parser.add_argument("--root", default="./checkpoints", help="训练输出目录")
    parser.add_argument("--keep", type=int, default=3, help="保留最近 N 组")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不真正删除")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"目录不存在: {root}")
        return

    files = scan_target_files(root)
    if not files:
        print("未找到任何可清理的文件")
        return

    grouped = group_files_by_timestamp(files)  # 已按时间降序
    keep_n = args.keep
    to_delete = []
    for idx, (_, group) in enumerate(grouped):
        if idx >= keep_n:
            to_delete.extend(group)

    if not to_delete:
        print("没有需要删除的文件")
        return

    print(f"将删除以下 {len(to_delete)} 个文件（保留最近 {keep_n} 组）:")
    delete_files(to_delete, dry=args.dry_run)

    if args.dry_run:
        print("\n如确认无误，请去掉 --dry-run 再次运行以真正删除")


if __name__ == "__main__":
    while True:
        main()
        # 倒计时等待
        time.sleep(1)