"""Compare saved evaluation archives on AutoDL without loading a model."""

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation_stats import compare_archives, read_archive


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_archives(read_archive(args.baseline), read_archive(args.candidate), args.draws, args.seed)
    result["archive_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in (("baseline", args.baseline), ("candidate", args.candidate))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result["overall_paired_location_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
