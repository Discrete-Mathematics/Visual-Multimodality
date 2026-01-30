#!/usr/bin/env python3
"""
把模型 2 的 lib/ 目录整包复制到当前目录，
用于与模型 1 训练脚本协同工作。
"""
import shutil
import sys
from pathlib import Path
from datetime import datetime

def main():
    # 0. 模型 2 源码根目录 —— 改成你的实际路径
    model2_repo = Path("/root/autodl-tmp/RDFC-GAN-main/RDFC-GAN")      # <<< 只改这一行
    src_lib     = model2_repo / "lib"
    if not src_lib.exists():
        print(f"[Error] 未找到 {src_lib} ，请检查 model2_repo 路径！")
        sys.exit(1)

    # 1. 当前目录
    dst_root = Path(__file__).absolute().parent
    dst_lib  = dst_root / "lib"

    # 2. 若已存在 lib，先备份
    if dst_lib.exists():
        backup = dst_root / f"lib_backup_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(str(dst_lib), str(backup))
        print(f"[Info] 原 lib 已备份到 {backup}")

    # 3. 拷贝
    shutil.copytree(str(src_lib), str(dst_lib))
    print(f"[Success] 模型 2 的 lib 已完整拷贝到：{dst_lib}")

if __name__ == "__main__":
    main()