"""Small correctness checks, to be executed on AutoDL only; no model training."""

import argparse
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .evaluation_stats import (EvaluationAccumulator, array_fingerprint, compare_archives,
                               confusion_matrix, read_archive, summary_scores)

DATA_ROOT = None
SPLIT = None


def fixture(perfect=True):
    accumulator = EvaluationAccumulator(capture_groups=True)
    label = np.arange(16, dtype=np.int64).reshape(4, 4)
    mask = np.ones_like(label, dtype=bool)
    for i in range(4):
        accumulator.update(name=f"mine-{i}-x-y-a-b-c", group=f"mine-{i}-x-y",
                           prediction=label if perfect else (label + 1) % 16, target=label,
                           original_valid=mask, current_valid=mask,
                           coverage=1.0, brightness=0.5, input_fingerprint=array_fingerprint(label))
    return accumulator


META = {"protocol": "development", "official_test": False, "corruption": "none",
        "corruption_seed": 2026, "input_size": [4, 4], "inference": "fixture only",
        "input_fingerprint_contract": "fixture only"}


class StatisticalChecks(unittest.TestCase):
    def test_known_confusion_and_absent_regions(self):
        label = np.array([[0, 1], [1, 255]], dtype=np.int64)
        pred = np.array([[0, 0], [1, 15]], dtype=np.int64)
        scores = summary_scores(confusion_matrix(pred, label))
        self.assertEqual(scores["pixels"], 3)
        self.assertAlmostEqual(scores["miou"], 0.5)
        self.assertAlmostEqual(scores["pa"], 2 / 3)
        empty = summary_scores(confusion_matrix(pred, label, np.zeros_like(label, dtype=bool)))
        self.assertIsNone(empty["miou"])
        self.assertEqual(empty["present_classes"], [])
        with self.assertRaises(ValueError):
            confusion_matrix(pred, np.array([[0, 16], [1, 255]]))
        with self.assertRaises(ValueError):
            confusion_matrix(pred + 16, label)

    def test_region_partition_and_group_preservation(self):
        accumulator = EvaluationAccumulator(capture_groups=True)
        target = np.array([[0, 1], [2, 255]], dtype=np.int64)
        original = np.array([[1, 0], [1, 1]], dtype=bool)
        current = np.array([[1, 0], [0, 1]], dtype=bool)
        for suffix in ("a", "b"):
            accumulator.update(name="mine-1-x-y-" + suffix, group="mine-1-x-y",
                               prediction=target % 16, target=target, original_valid=original,
                               current_valid=current, coverage=0.75, brightness=0.1,
                               input_fingerprint=array_fingerprint(target))
        results = accumulator.results()
        self.assertEqual(results["all"]["pixels"], 6)
        self.assertEqual(results["natural_depth_hole"]["pixels"], 2)
        self.assertEqual(results["natural_depth_valid"]["pixels"], 4)
        self.assertEqual(results["synthetic_depth_hole"]["pixels"], 2)
        self.assertEqual(results["dark_natural_depth_hole"]["pixels"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.npz"
            accumulator.save(path, META)
            archive = read_archive(path)
            self.assertEqual(len(archive["sample_ids"]), 2)
            self.assertEqual(len(archive["group_ids"]), 1)
            with self.assertRaises(FileExistsError):
                accumulator.save(path, META)
            with self.assertRaises(ValueError):
                compare_archives(archive, archive, draws=100)

    def test_bootstrap_identity_known_gain_and_data_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.npz", Path(directory) / "right.npz"
            fixture(perfect=False).save(left, META)
            fixture(perfect=True).save(right, META)
            baseline, candidate = read_archive(left), read_archive(right)
            result = compare_archives(baseline, candidate, draws=100)
            interval = result["overall_paired_location_bootstrap"]["percentile_95_interval_percentage_points"]
            for key in ("miou", "pa", "rare_miou"):
                self.assertEqual(interval[key], {"low": 100.0, "high": 100.0})
            identity = compare_archives(candidate, candidate, draws=100)
            self.assertEqual(identity["overall_paired_location_bootstrap"]["percentile_95_interval_percentage_points"]["miou"],
                             {"low": 0.0, "high": 0.0})
            for key in ("input_fingerprints", "target_fingerprints", "sample_groups"):
                broken = copy.deepcopy(candidate)
                broken[key][0] = "different"
                with self.assertRaisesRegex(ValueError, "Unmatched evaluation data"):
                    compare_archives(baseline, broken, draws=100)
            broken = copy.deepcopy(candidate)
            broken["metadata"]["corruption_seed"] += 1
            with self.assertRaisesRegex(ValueError, "Unmatched evaluation protocol"):
                compare_archives(baseline, broken, draws=100)
            broken = copy.deepcopy(candidate)
            broken["matrices"][0, 0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "Ground-truth pixel counts"):
                compare_archives(baseline, broken, draws=100)

    def test_partial_gain_and_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            fixture(perfect=False).save(path, META)
            baseline = read_archive(path)
            candidate = copy.deepcopy(baseline)
            # One of four locations changes from wholly wrong to perfect.
            for j in range(len(candidate["regions"])):
                rows = baseline["matrices"][0, j].sum(1)
                candidate["matrices"][0, j] = np.diag(rows)
            result = compare_archives(baseline, candidate, draws=200, seed=42)
            repeat = compare_archives(baseline, candidate, draws=200, seed=42)
            self.assertEqual(result, repeat)
            point = result["point_results"]["all"]["difference_percentage_points"]
            self.assertAlmostEqual(point["pa"], 25.0)
            self.assertAlmostEqual(point["miou"], 100 / 7)
            interval = result["overall_paired_location_bootstrap"]["percentile_95_interval_percentage_points"]["pa"]
            self.assertLessEqual(interval["low"], 25.0)
            self.assertGreaterEqual(interval["high"], 25.0)

    def test_real_data_corruptions_keep_natural_holes(self):
        # Data loading only: no model import, forward pass, or accuracy experiment.
        from .data import MineDataset
        from PIL import Image
        clean = MineDataset(DATA_ROOT, SPLIT, size=(96, 128), seed=2026)[0]
        name = clean["id"]
        raw = np.array(Image.open(DATA_ROOT / "Depth" / (name + ".png")), dtype=np.uint16)
        expected = np.array(Image.fromarray(raw).resize((128, 96), Image.Resampling.NEAREST)) > 0
        np.testing.assert_array_equal(clean["original_valid"][0].numpy(), expected)
        np.testing.assert_array_equal(clean["original_valid"].numpy(), clean["valid"].numpy() > 0)
        for kind in ("holes", "noise", "dropout", "dark"):
            data = MineDataset(DATA_ROOT, SPLIT, size=(96, 128), seed=2026, corruption=kind)[0]
            repeat = MineDataset(DATA_ROOT, SPLIT, size=(96, 128), seed=2026, corruption=kind)[0]
            self.assertEqual(data["sensor_fingerprint"], repeat["sensor_fingerprint"])
            np.testing.assert_array_equal(data["original_valid"].numpy(), clean["original_valid"].numpy())
            for key in ("rgb", "depth", "valid", "hha", "label"):
                np.testing.assert_array_equal(data[key].numpy(), repeat[key].numpy())
            if kind == "dropout":
                self.assertEqual(int(data["valid"].count_nonzero()), 0)
            if kind in ("noise", "dark"):
                np.testing.assert_array_equal(data["valid"].numpy(), clean["valid"].numpy())


if __name__ == "__main__":
    if not Path("/root/autodl-tmp").is_dir():
        raise SystemExit("Run these checks on AutoDL, not on the local computer")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/MUSeg"))
    parser.add_argument("--split", type=Path, default=Path("/root/autodl-tmp/relay_splits_v1/dev_val.txt"))
    args = parser.parse_args()
    DATA_ROOT, SPLIT = args.data_root, args.split
    unittest.main(argv=["cloud_checks"], verbosity=2)
