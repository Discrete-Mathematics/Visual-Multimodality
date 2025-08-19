import os
from typing import List

TARGET_DIR = "/root/autodl-tmp/DFormer/datasets/MUSeg"  

FILES_TO_PROCESS: List[str] = [
    os.path.join(TARGET_DIR, "train.txt"),
    os.path.join(TARGET_DIR, "test.txt"),
]


def transform_line(name: str) -> str:
    name = name.strip()
    if not name:
        return ""  # 保持空行
    return f"RGB/{name}.jpg Label/{name}.png"


def process_file(file_path: str) -> None:
    if not os.path.isfile(file_path):
        print(f"[WARN] 文件不存在，跳过: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    transformed = [transform_line(line) for line in lines]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transformed) + ("\n" if transformed else ""))

    print(f"[OK] 已处理: {file_path}")


if __name__ == "__main__":
    for path in FILES_TO_PROCESS:
        process_file(path) 