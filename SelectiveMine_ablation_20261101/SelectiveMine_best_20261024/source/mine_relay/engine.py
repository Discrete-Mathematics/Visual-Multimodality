"""CUDA-only execution helpers. No local CPU fallback is provided."""

import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from .evaluation_stats import EvaluationAccumulator, summary_scores
from .data import read_ids, group_id


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this package on the rented AutoDL GPU, not the local computer.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): fingerprint(p) for p in sorted(root.rglob("*.py"))}


def validate_protocol(root, train_file, val_file, protocol, pretrained_hha=False):
    official_train = set(read_ids(Path(root) / "train.txt"))
    official_test = set(read_ids(Path(root) / "val.txt"))
    train = set(read_ids(train_file))
    val = set(read_ids(val_file)) if val_file else set()
    if train & val or set(map(group_id, train)) & set(map(group_id, val)):
        raise ValueError("Training and validation must be separated by location group")
    if protocol == "development":
        if pretrained_hha or not val or not train <= official_train or not val <= official_train:
            raise ValueError("Development requires a clean ImageNet start and disjoint subsets of official TRAIN")
    elif protocol == "continuation":
        if train != official_train or val != official_test:
            raise ValueError("Continuation uses the historical official split and is exploratory only")
    elif protocol == "final":
        if train != official_train or val or pretrained_hha:
            raise ValueError("Final retraining uses all official TRAIN, ImageNet initialization, and no validation/test selection")
    else:
        raise ValueError(protocol)


def class_weights(root, ids):
    counts = np.zeros(16, dtype=np.int64)
    for name in ids:
        label = np.array(Image.open(Path(root) / "Label" / (name + "_label.png")))
        if not np.isin(label, list(range(16)) + [255]).all():
            raise ValueError(f"Unexpected class in {name}")
        counts += np.bincount(label[label < 16], minlength=16)
    frequency = counts / max(int(counts.sum()), 1)
    weights = 1.0 / np.sqrt(np.clip(frequency, 1e-8, None))
    weights /= weights.mean()
    return torch.tensor(np.clip(weights, 0.25, 3.0), dtype=torch.float32, device="cuda"), counts


def forward_batch(model, batch, damaged=False):
    keys = ("rgb", "depth", "valid", "luma", "hha")
    tensors = []
    for key in keys:
        name = "damaged_" + key if damaged and key in ("depth", "valid", "hha") else key
        tensors.append(batch[name].cuda(non_blocking=True))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(*tensors)


def serial_scores(confusion):
    return summary_scores(confusion)


@torch.no_grad()
def evaluate(model, loader, max_batches=None, archive_path=None, metadata=None):
    model.eval()
    accumulator = EvaluationAccumulator(capture_groups=archive_path is not None)
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        pred = forward_batch(model, batch)["logits"].argmax(1).cpu().numpy()
        gt = batch["label"].numpy()
        for i, name in enumerate(batch["id"]):
            input_hash = None
            if archive_path is not None:
                input_hash = batch["sensor_fingerprint"][i]
            accumulator.update(name=name, group=group_id(name), prediction=pred[i], target=gt[i],
                               original_valid=batch["original_valid"][i, 0].numpy(),
                               current_valid=batch["valid"][i, 0].numpy() > 0,
                               coverage=float(batch["coverage"][i]), brightness=float(batch["brightness"][i]),
                               input_fingerprint=input_hash)
    result = accumulator.results()
    if archive_path is not None:
        accumulator.save(archive_path, metadata or {})
    return result


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
