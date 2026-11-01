"""NumPy-only evaluation records and paired location bootstrap.

Rows are ground truth; columns are predictions. The legacy primary mIoU
averages over classes present in ground truth, including background. Empty
regions are reported as null, never as a measured zero accuracy.
"""

import hashlib
import json
from pathlib import Path
import numpy as np

from .legacy.labels import CLASS_NAMES, NUM_CLASSES
from .legacy.metrics import scores_from_confusion

SCHEMA = "mine-relay-evaluation-v2"
RARE = [1, 11, 12, 13]


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def confusion_matrix(prediction, target, mask=None):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    if not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("Class ids must be integers")
    if np.any((target != 255) & ((target < 0) | (target >= NUM_CLASSES))):
        raise ValueError("Unexpected ground-truth label")
    if np.any((prediction < 0) | (prediction >= NUM_CLASSES)):
        raise ValueError("Unexpected prediction label")
    selected = target != 255
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError("Region mask and label shapes differ")
        selected &= mask
    encoded = NUM_CLASSES * target[selected].astype(np.int64) + prediction[selected]
    return np.bincount(encoded, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def summary_scores(matrix):
    pixels = int(matrix.sum())
    present = matrix.sum(1) > 0
    raw = scores_from_confusion(matrix)
    return {
        "miou": raw["miou"] if pixels else None,
        "mpa": raw["mpa"] if pixels else None,
        "pa": raw["pa"] if pixels else None,
        # Kept fixed at four classes for compatibility; absence counts as zero.
        "rare_miou": float(raw["iou"][RARE].mean()) if pixels else None,
        "class_iou": dict(zip(CLASS_NAMES, raw["iou"].tolist())) if pixels else dict.fromkeys(CLASS_NAMES),
        "ground_truth_pixels": dict(zip(CLASS_NAMES, matrix.sum(1).tolist())),
        "present_classes": [name for name, yes in zip(CLASS_NAMES, present) if yes],
        "pixels": pixels,
    }


class EvaluationAccumulator:
    def __init__(self, capture_groups=False):
        self.capture_groups = capture_groups
        self.matrices, self.image_counts = {}, {}
        self.group_matrices, self.samples = {}, {}

    def update(self, *, name, group, prediction, target, original_valid,
               current_valid, coverage, brightness, input_fingerprint=None):
        if name in self.samples:
            raise ValueError(f"Duplicate evaluated sample: {name}")
        original_valid = np.asarray(original_valid, dtype=bool)
        current_valid = np.asarray(current_valid, dtype=bool)
        if original_valid.shape != target.shape or current_valid.shape != target.shape:
            raise ValueError("Depth masks and labels must align")
        if np.any(current_valid & ~original_valid):
            raise ValueError("Evaluation corruption cannot invent valid depth")
        if not (0 <= coverage <= 1 and 0 <= brightness <= 1):
            raise ValueError("Invalid original image statistics")
        if self.capture_groups and not input_fingerprint:
            raise ValueError("Archived comparisons require model-input fingerprints")
        self.samples[name] = (group, input_fingerprint or "", array_fingerprint(target, original_valid, current_valid))
        dark = brightness < 40 / 255
        regions = {
            "all": None,
            "mine_" + name.split("-")[0]: None,
            "depth_low" if coverage < 0.4 else "depth_mid" if coverage < 0.8 else "depth_high": None,
            "dark" if dark else "other_light": None,
            "natural_depth_valid": original_valid,
            "natural_depth_hole": ~original_valid,
            "synthetic_depth_hole": original_valid & ~current_valid,
            "dark_natural_depth_hole": (~original_valid) if dark else np.zeros_like(original_valid),
        }
        whole = confusion_matrix(prediction, target)
        for region, mask in regions.items():
            matrix = whole if mask is None else confusion_matrix(prediction, target, mask)
            self.matrices.setdefault(region, np.zeros_like(matrix))[:] += matrix
            self.image_counts[region] = self.image_counts.get(region, 0) + int(matrix.sum() > 0)
            if self.capture_groups:
                key = (group, region)
                self.group_matrices.setdefault(key, np.zeros_like(matrix))[:] += matrix

    def results(self):
        if not self.samples:
            raise ValueError("No samples were evaluated")
        return {key: dict(summary_scores(matrix), images=self.image_counts[key])
                for key, matrix in self.matrices.items()}

    def save(self, path, metadata):
        path = Path(path)
        if not self.capture_groups or not self.samples:
            raise ValueError("No location records captured")
        if path.exists():
            raise FileExistsError(path)
        names = sorted(self.samples)
        groups = sorted({self.samples[name][0] for name in names})
        regions = sorted(self.matrices)
        matrices = np.zeros((len(groups), len(regions), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for i, group in enumerate(groups):
            for j, region in enumerate(regions):
                if (group, region) in self.group_matrices:
                    matrices[i, j] = self.group_matrices[group, region]
        metadata = dict(metadata, schema=SCHEMA, class_names=CLASS_NAMES,
                        primary_miou="mean IoU over GT-present classes, including background",
                        rare_miou="fixed mean over ids 1,11,12,13; absent classes score zero")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, sample_ids=np.array(names),
                                sample_groups=np.array([self.samples[n][0] for n in names]),
                                input_fingerprints=np.array([self.samples[n][1] for n in names]),
                                target_fingerprints=np.array([self.samples[n][2] for n in names]),
                                group_ids=np.array(groups), regions=np.array(regions), matrices=matrices,
                                metadata=np.array(json.dumps(metadata, ensure_ascii=False)))


def read_archive(path):
    with np.load(path, allow_pickle=False) as packed:
        result = {key: packed[key] for key in packed.files}
    metadata = json.loads(result.pop("metadata").item())
    if metadata.get("schema") != SCHEMA or metadata.get("class_names") != CLASS_NAMES:
        raise ValueError("Unsupported evaluation schema or class mapping")
    for key in ("protocol", "official_test", "corruption", "corruption_seed", "input_size", "inference",
                "input_fingerprint_contract"):
        if key not in metadata:
            raise ValueError(f"Missing evaluation protocol field: {key}")
    matrix = result["matrices"]
    expected = (len(result["group_ids"]), len(result["regions"]), NUM_CLASSES, NUM_CLASSES)
    if matrix.shape != expected or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError("Invalid location confusion matrices")
    for key in ("sample_ids", "group_ids", "regions"):
        if len(result[key]) != len(set(result[key].tolist())) or not len(result[key]):
            raise ValueError(f"Empty or duplicated {key}")
    if set(result["sample_groups"].tolist()) != set(result["group_ids"].tolist()):
        raise ValueError("Inconsistent sample groups")
    for key in ("sample_groups", "input_fingerprints", "target_fingerprints"):
        if len(result[key]) != len(result["sample_ids"]):
            raise ValueError(f"Wrong record count: {key}")
    if "all" not in result["regions"]:
        raise ValueError("Overall confusion matrix is missing")
    result["metadata"] = metadata
    return result


def compare_archives(baseline, candidate, draws=2000, seed=2026):
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    for key in ("sample_ids", "sample_groups", "group_ids", "regions", "input_fingerprints", "target_fingerprints"):
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"Unmatched evaluation data: {key}")
    for key in ("protocol", "official_test", "corruption", "corruption_seed", "input_size", "inference",
                "input_fingerprint_contract"):
        if baseline["metadata"].get(key) != candidate["metadata"].get(key):
            raise ValueError(f"Unmatched evaluation protocol: {key}")
    left, right = baseline["matrices"], candidate["matrices"]
    if not np.array_equal(left.sum(-1), right.sum(-1)):
        raise ValueError("Ground-truth pixel counts differ")
    regions = baseline["regions"].tolist()
    point = {}
    for j, region in enumerate(regions):
        a, b = summary_scores(left[:, j].sum(0)), summary_scores(right[:, j].sum(0))
        delta = {key: 100 * (b[key] - a[key]) if a[key] is not None else None
                 for key in ("miou", "pa", "rare_miou")}
        point[region] = {"baseline": a, "candidate": b, "difference_percentage_points": delta}
    # Pair locations, keeping their correlated views together. Preserve the
    # observed mine composition; this does not estimate generalization to new mines.
    groups = baseline["group_ids"].tolist()
    if len(groups) < 2:
        raise ValueError("At least two location groups are required")
    strata = {}
    for index, group in enumerate(groups):
        strata.setdefault(group.split("-")[0], []).append(index)
    j = regions.index("all")
    if left[:, j].sum() == 0:
        raise ValueError("No labeled evaluation pixels")
    a_flat, b_flat = left[:, j].reshape(len(groups), -1), right[:, j].reshape(len(groups), -1)
    rng = np.random.default_rng(seed)
    deltas = np.empty((draws, 3), dtype=np.float64)
    class_counts = []
    for i in range(draws):
        counts = np.zeros(len(groups), dtype=np.int64)
        for indices in strata.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            counts += np.bincount(sampled, minlength=len(groups))
        a_matrix = (counts @ a_flat).reshape(NUM_CLASSES, NUM_CLASSES)
        b_matrix = (counts @ b_flat).reshape(NUM_CLASSES, NUM_CLASSES)
        a, b = scores_from_confusion(a_matrix), scores_from_confusion(b_matrix)
        deltas[i] = (100 * (b["miou"] - a["miou"]), 100 * (b["pa"] - a["pa"]),
                     100 * (b["iou"][RARE].mean() - a["iou"][RARE].mean()))
        class_counts.append(int((a_matrix.sum(1) > 0).sum()))
    intervals = {key: {"low": float(np.quantile(deltas[:, k], .025)),
                       "high": float(np.quantile(deltas[:, k], .975))}
                 for k, key in enumerate(("miou", "pa", "rare_miou"))}
    training_a = baseline["metadata"].get("training", {})
    training_b = candidate["metadata"].get("training", {})
    differences = {key: {"baseline": training_a.get(key), "candidate": training_b.get(key)}
                   for key in sorted(set(training_a) | set(training_b)) if training_a.get(key) != training_b.get(key)}
    return {"schema": SCHEMA, "point_results": point,
            "overall_paired_location_bootstrap": {
                "draws": draws, "seed": seed, "groups": len(groups),
                "mine_strata": {key: len(value) for key, value in strata.items()},
                "percentile_95_interval_percentage_points": intervals,
                "resampled_present_class_count_range": [min(class_counts), max(class_counts)],
            }, "training_differences": differences,
            "interpretation": [
                "Positive differences favor the candidate; subgroup results are descriptive.",
                "Intervals condition on fitted models and the observed mines; they do not measure training-seed variation.",
                "Primary mIoU averages GT-present classes separately in each draw; inspect the reported class-count range.",
                "Matched evaluation inputs alone do not prove a fair training budget or model-selection procedure.",
                "Repeated development comparisons are exploratory; an interval excluding zero is not evidence of novelty.",
            ]}
