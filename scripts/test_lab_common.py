#!/usr/bin/env python3
"""Unit tests for the perf_threshold helpers in content/labs/lab_common.py.

Usage:
    python3 scripts/test_lab_common.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "content" / "labs"))
from lab_common import enforce_perf_thresholds, validate_perf_threshold  # noqa: E402


def run(metrics, threshold, tol=None, listed=None):
    meta = {"metrics": listed or list(threshold), "perf_threshold": threshold}
    if tol is not None:
        meta["perf_tolerance"] = tol
    result = {"passed": True, "checks": [{"name": "c", "passed": True, "detail": ""}],
              "metrics": metrics}
    return enforce_perf_thresholds(result, meta)


class PerfThreshold(unittest.TestCase):
    def test_min_pass(self):
        r = run({"x": 1.6}, {"x": {"min": 1.5}})
        self.assertTrue(r["passed"])

    def test_min_fail(self):
        r = run({"x": 1.0}, {"x": {"min": 1.5}})
        self.assertFalse(r["passed"])
        self.assertIn("measured 1", r["checks"][-1]["detail"])

    def test_min_tolerance_direction(self):
        # 1.45 >= 1.5 * 0.95 = 1.425 passes. A wrong 1.5 * 1.05 = 1.575 bound fails it.
        self.assertTrue(run({"x": 1.45}, {"x": {"min": 1.5}})["passed"])
        self.assertFalse(run({"x": 1.42}, {"x": {"min": 1.5}})["passed"])

    def test_max_within_tolerance(self):
        # 10.4 <= 10 * 1.05 = 10.5 passes. 10.6 fails.
        self.assertTrue(run({"x": 10.4}, {"x": {"max": 10}})["passed"])
        self.assertFalse(run({"x": 10.6}, {"x": {"max": 10}})["passed"])

    def test_custom_tolerance(self):
        self.assertFalse(run({"x": 1.45}, {"x": {"min": 1.5}}, tol=0.01)["passed"])

    def test_absent_or_nonnumeric_skipped(self):
        for metrics in ({}, {"x": "skipped"}, {"x": None}, {"x": True}):
            r = run(metrics, {"x": {"min": 1.5}})
            self.assertTrue(r["passed"], metrics)
            self.assertTrue(r["checks"][-1]["skipped"])

    def test_failed_correctness_stays_failed(self):
        r = {"passed": False, "checks": [{"name": "c", "passed": False}], "metrics": {"x": 9}}
        meta = {"metrics": ["x"], "perf_threshold": {"x": {"min": 1}}}
        self.assertFalse(enforce_perf_thresholds(r, meta)["passed"])

    def test_no_block_is_noop(self):
        r = {"passed": True, "checks": [], "metrics": {}}
        self.assertEqual(enforce_perf_thresholds(r, {"metrics": []}), r)

    def test_bad_shapes_rejected(self):
        bad = [
            {"x": {"min": 1, "max": 2}},
            {"x": {"minimum": 1}},
            {"x": {"min": "fast"}},
            {"x": {"min": -1}},
            {"x": 1.5},
            {"y": {"min": 1}},  # not in metrics
            {},
        ]
        for block in bad:
            meta = {"metrics": ["x"], "perf_threshold": block}
            self.assertTrue(validate_perf_threshold(meta), block)
        self.assertTrue(validate_perf_threshold(
            {"metrics": ["x"], "perf_threshold": {"x": {"min": 1}}, "perf_tolerance": 2}))
        self.assertTrue(validate_perf_threshold({"metrics": ["x"], "perf_tolerance": 0.1}))
        self.assertEqual(validate_perf_threshold({"metrics": ["x"]}), [])

    def test_bad_shape_fails_result(self):
        r = run({"x": 5}, {"x": {"min": "a"}})
        self.assertFalse(r["passed"])


if __name__ == "__main__":
    unittest.main()
