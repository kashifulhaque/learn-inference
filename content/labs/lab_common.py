"""Helpers shared by every lab's test harness."""

from __future__ import annotations

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
