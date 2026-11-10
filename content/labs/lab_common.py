"""Helpers shared by every lab's test harness."""

from __future__ import annotations

import math
import traceback
from typing import Any, Callable


class Checks:
    """Collects pass/fail results and the metrics a lab reports."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"name": name, "passed": bool(passed), "detail": detail})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return bool(passed)

    def check(self, name: str, fn: Callable[[], Any], detail: str = "") -> bool:
        """Run `fn`; a truthy return passes, an exception fails with its message."""
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.add(name, False, f"{type(exc).__name__}: {exc}")
        if isinstance(value, tuple) and len(value) == 2:
            passed, extra = value
            return self.add(name, passed, extra or detail)
        return self.add(name, bool(value), detail)

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value
        print(f"      {key} = {value}", flush=True)

    def require(self, submission: Any, *names: str) -> bool:
        """Fail early when the submission is missing a required function."""
        missing = [n for n in names if not hasattr(submission, n)]
        if missing:
            self.add(
                "required functions",
                False,
                f"Your file must define: {', '.join(missing)}",
            )
            return False
        return True

    def finish(self) -> dict[str, Any]:
        passed = bool(self.results) and all(r["passed"] for r in self.results)
        total = len(self.results)
        won = sum(1 for r in self.results if r["passed"])
        print(f"\n{won}/{total} checks passed", flush=True)
        return {"passed": passed, "checks": self.results, "metrics": self.metrics}


def close(actual, expected, tol: float = 1e-5) -> tuple[bool, str]:
    """Compare two tensors and report the largest absolute difference."""
    import torch

    actual = torch.as_tensor(actual).float()
    expected = torch.as_tensor(expected).float()
    if actual.shape != expected.shape:
        return False, f"shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    err = (actual - expected).abs().max().item()
    return err <= tol, f"max abs error {err:.3e} (tolerance {tol:.1e})"


def require_cuda() -> "torch.device":  # type: ignore[name-defined]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This lab needs a GPU but none is visible.")
    return torch.device("cuda")


def pick_device():
    """Use the GPU when there is one, otherwise fall back to the CPU.

    Labs where the GPU is the subject call `require_cuda` instead. Labs that
    only check mathematics use this, so the same tests run anywhere.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_PERF_TOLERANCE = 0.05


def validate_perf_threshold(meta: dict[str, Any]) -> list[str]:
    """Return the problems with a lab's `perf_threshold` and `perf_tolerance`.

    An empty list means the block is absent or well formed. Each threshold must
    be `{"min": number}` or `{"max": number}` and name a metric in `metrics`.
    """
    errors: list[str] = []

    def is_number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    if "perf_tolerance" in meta:
        tol = meta["perf_tolerance"]
        if not is_number(tol) or not 0 <= tol < 1:
            errors.append(f"perf_tolerance must be a number in [0, 1), got {tol!r}")
    block = meta.get("perf_threshold")
    if block is None:
        if "perf_tolerance" in meta:
            errors.append("perf_tolerance is set but perf_threshold is not")
        return errors
    if not isinstance(block, dict) or not block:
        return errors + ["perf_threshold must be a non-empty object"]
    known = set(meta.get("metrics", []))
    for metric, spec in block.items():
        if metric not in known:
            errors.append(f"perf_threshold names '{metric}', which is not in metrics")
        if not isinstance(spec, dict) or len(spec) != 1:
            errors.append(f"perf_threshold['{metric}'] must have exactly one key, min or max")
            continue
        (kind, value), = spec.items()
        if kind not in ("min", "max"):
            errors.append(f"perf_threshold['{metric}'] has unknown key '{kind}' (use min or max)")
        elif not is_number(value) or value <= 0:
            errors.append(f"perf_threshold['{metric}'].{kind} must be a positive number, got {value!r}")
    return errors


def enforce_perf_thresholds(result: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Append one check per `perf_threshold` entry to a lab result.

    A `min` threshold passes when metric >= min * (1 - tol), and a `max`
    threshold passes when metric <= max * (1 + tol), where tol is
    `perf_tolerance` (default 0.05). A metric that is absent or not a number,
    such as one skipped on a CPU, is recorded as a passing "skipped" line so
    that CPU runs and CI stay green. A malformed block adds one failing check.
    Updates `result["passed"]` and returns `result`.
    """
    block = meta.get("perf_threshold")
    if block is None:
        return result
    checks = result.setdefault("checks", [])
    first_new = len(checks)
    problems = validate_perf_threshold(meta)
    if problems:
        checks.append({"name": "perf_threshold config", "passed": False,
                       "detail": "; ".join(problems)})
    else:
        tol = meta.get("perf_tolerance", DEFAULT_PERF_TOLERANCE)
        metrics = result.get("metrics", {})
        for metric, spec in block.items():
            (kind, limit), = spec.items()
            bound = limit * (1 - tol) if kind == "min" else limit * (1 + tol)
            op = ">=" if kind == "min" else "<="
            name = f"perf: {metric} {op} {limit:g} (tolerance {tol:.0%}, bound {bound:g})"
            value = metrics.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                checks.append({"name": name, "passed": True, "skipped": True,
                               "detail": f"skipped, metric not measured (got {value!r})"})
                continue
            ok = value >= bound if kind == "min" else value <= bound
            checks.append({"name": name, "passed": bool(ok),
                           "detail": f"measured {value:g}, need {op} {bound:g}"})
    for c in checks[first_new:]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']} — {c['detail']}", flush=True)
    # Thresholds can only turn a pass into a failure, never the reverse: a
    # result that already failed (for example, a harness error) stays failed.
    prior = bool(result.get("passed", True))
    result["passed"] = prior and all(c["passed"] for c in checks[first_new:])
    return result
