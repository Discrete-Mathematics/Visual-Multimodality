"""AutoDL evaluation; official test evaluation is explicit and recorded."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .data import MineDataset, read_ids, group_id
from .model import MineRelay
from .engine import require_cuda, seed_everything, fingerprint, source_fingerprints, evaluate, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--corruption", choices=["none", "holes", "noise", "dropout", "dark"], default="none")
    p.add_argument("--corruption-seed", type=int, default=2026,
                   help="Fixed across training seeds so models see identical corruption")
    p.add_argument("--official-test", action="store_true", help="Explicitly evaluate the reserved official test set")
    args = p.parse_args()
    require_cuda()
    archive = args.output.with_name(args.output.stem + ".groups.npz")
    if args.output.suffix != ".json":
        raise ValueError("Evaluation output must end in .json")
    if args.output.exists() or archive.exists():
        raise FileExistsError("Use a new evaluation output path")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid evaluation batch size or worker count")
    test_ids = set(read_ids(args.data_root / "val.txt"))
    eval_ids = set(read_ids(args.split))
    if not eval_ids:
        raise ValueError("Empty evaluation split")
    if eval_ids & test_ids and not args.official_test:
        raise ValueError("Official test data require --official-test; use dev_val for model selection")
    if args.official_test and eval_ids != test_ids:
        raise ValueError("Official evaluation must use the entire official test split")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    code_hashes = source_fingerprints()
    if checkpoint["source_hashes"] != code_hashes:
        raise ValueError("Source differs from the training checkpoint; evaluate with its original source package")
    if config["max_steps"]:
        raise ValueError("A smoke checkpoint is not a research evaluation model")
    if checkpoint.get("partial_epoch_replayed_on_resume"):
        raise ValueError("Use a completed-epoch checkpoint for research evaluation")
    if config["protocol"] == "development" and eval_ids != set(checkpoint["val_ids"]):
        raise ValueError("Development evaluation must use the complete fixed validation split")
    if args.official_test and config["protocol"] != "final":
        raise ValueError("Use a fixed-epoch final retraining checkpoint for official testing; historical runs are exploratory")
    if args.official_test and (checkpoint["next_epoch"] != config["epochs"] or checkpoint.get("partial_epoch_replayed_on_resume")):
        raise ValueError("Final retraining has not completed its predeclared epochs")
    train_ids = set(checkpoint["train_ids"])
    if eval_ids & train_ids or set(map(group_id, eval_ids)) & set(map(group_id, train_ids)):
        raise ValueError("Evaluation sample appeared in training")
    seed_everything(config["seed"])
    model = MineRelay(config["mode"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.mix.fill_(1.0)
    model.cuda().eval()
    loader = DataLoader(MineDataset(args.data_root, args.split, (config["height"], config["width"]),
                                    corruption=args.corruption, seed=args.corruption_seed),
                        batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    training = {key: config[key] for key in ("mode", "seed", "epochs", "batch_size", "lr", "aux",
                                            "paired_weight", "recovery_weight", "quality_weight", "rare_crop")}
    training.update(completed_epochs=checkpoint["next_epoch"], initial_sha256=checkpoint["initial_hash"],
                    train_split_sha256=checkpoint["split_hashes"]["train"],
                    val_split_sha256=checkpoint["split_hashes"]["val"],
                    elapsed_seconds=checkpoint.get("elapsed_seconds"))
    metadata = {"checkpoint_sha256": fingerprint(args.checkpoint), "split_sha256": fingerprint(args.split),
                "corruption": args.corruption, "corruption_seed": args.corruption_seed,
                "protocol": config["protocol"], "official_test": args.official_test,
                "input_size": [config["height"], config["width"]],
                "input_fingerprint_contract": "RGB float32 HWC [0,1] and uint16 depth after resize/corruption",
                "inference": "single resize, no TTA, CUDA bf16 autocast",
                "source_hashes": code_hashes, "training": training}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(metadata, metrics=evaluate(model, loader, archive_path=archive, metadata=metadata),
                   group_archive=archive.name, group_archive_sha256=fingerprint(archive))
    write_json(args.output, results)
    print(results["metrics"]["all"], flush=True)


if __name__ == "__main__":
    main()
