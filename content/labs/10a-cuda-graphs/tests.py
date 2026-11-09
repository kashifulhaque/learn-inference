"""Checks for batch buckets, capture rules, and CUDA graph replay.

The first check calls only `pick_bucket`. Later checks add padding waste,
the bucket set, the capture-rule names, and, when a CUDA device is visible,
replay.

A plausible wrong implementation rounds the batch up to the next power of
two and ignores the bucket list. `pick_bucket(5, [1, 2, 4, 6, 8])` is 6.
That implementation returns 8. On the hand trace `[1, 3, 5]` its waste is
4/13, the power-of-two waste, rather than 2/11.

The name check grades source text against the chapter's rules. It does not
run a capture. The replay checks skip when no CUDA device is visible.
"""

from __future__ import annotations

from lab_common import Checks

# The chapter's trace: 200 steps, 1070 real slots. Powers of two leave 402
# idle slots out of 1472. The tighter set leaves 150 out of 1220.
# scripts/figures/10a-cuda-graphs-and-launch-overhead.py prints both.
COUNTS = {
    1: 40,
    2: 20,
    3: 20,
    5: 30,
    6: 24,
    7: 12,
    8: 10,
    9: 16,
    10: 12,
    12: 8,
    13: 4,
    15: 4,
}
POW2 = [1, 2, 4, 8, 16, 32]
CHOSEN = [1, 2, 4, 6, 8, 12, 16, 24, 32]

SNIPPETS = {
    "static_copy": "def step(x, scale):\n    return x * scale\n",
    "host_item": "def step(x):\n    gate = x.sum().item()\n    return x * gate\n",
    "dynamic_rows": "def step(x, n):\n    return x[:n] * 2\n",
    "allocator_cache": "def step(x):\n    torch.cuda.empty_cache()\n    return x + 1\n",
}
FAILING = ["allocator_cache", "dynamic_rows", "host_item"]


def _trace() -> list[int]:
    steps: list[int] = []
    for batch, count in COUNTS.items():
        steps.extend([batch] * count)
    return steps


TRACE = _trace()


def _ref_pick(batch, buckets):
    fits = [size for size in buckets if size >= batch]
    if not fits:
        return None
    return min(fits)


def _ref_waste(trace, buckets):
    real = 0
    padded = 0
    for batch in trace:
        bucket = _ref_pick(batch, buckets)
        if bucket is None:
            raise ValueError(f"batch {batch} exceeds every bucket")
        real += batch
        padded += bucket
    return (padded - real) / padded


def _call(fn, expected):
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected!r}, got {type(exc).__name__}: {exc}"
    if actual != expected:
        return False, f"expected {expected!r}, got {actual!r}"
    return True, f"expected {expected!r}, got {actual!r}"


def _call_fraction(fn, expected, label):
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {label} ({expected!r}), got {type(exc).__name__}: {exc}"
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False, f"expected {label} ({expected!r}), got {actual!r}"
    if abs(actual - expected) > 1e-12:
        return False, f"expected {label} ({expected!r}), got {actual!r}"
    return True, f"expected {label} ({expected!r}), got {actual!r}"


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def run(submission):
    c = Checks()

    c.check(
        "an exact batch stays on its own bucket",
        lambda: _call(lambda: submission.pick_bucket(1, [1, 2, 4, 8]), 1),
    )

    # The power-of-two rounder returns 8 here.
    c.check(
        "a batch of 5 uses the bucket 6, not the next power of two",
        lambda: _call(lambda: submission.pick_bucket(5, [1, 2, 4, 6, 8]), 6),
    )

    def edges():
        first = _call(lambda: submission.pick_bucket(3, [8, 1, 4, 2]), 4)
        if not first[0]:
            return first
        second = _call(lambda: submission.pick_bucket(9, [1, 2, 4, 8]), None)
        if not second[0]:
            return second
        return True, f"{first[1]}; {second[1]}"

    c.check("unsorted buckets resolve to the smallest fit, and overflow is None", edges)

    def hand():
        wide = _call_fraction(
            lambda: submission.padding_waste([1, 3, 5], [1, 2, 4, 8]),
            4 / 13,
            "4/13",
        )
        if not wide[0]:
            return wide
        tight = _call_fraction(
            lambda: submission.padding_waste([1, 3, 5], [1, 2, 4, 6, 8]),
            2 / 11,
            "2/11",
        )
        if not tight[0]:
            return tight
        return True, f"{wide[1]}; {tight[1]}"

    c.check("padding waste on [1, 3, 5] is 4/13 and 2/11", hand)

    def full_trace():
        expected_pow = _ref_waste(TRACE, POW2)
        expected_chosen = _ref_waste(TRACE, CHOSEN)
        got_pow = _call_fraction(
            lambda: submission.padding_waste(TRACE, POW2),
            expected_pow,
            "201/736",
        )
        if not got_pow[0]:
            return got_pow
        got_chosen = _call_fraction(
            lambda: submission.padding_waste(TRACE, CHOSEN),
            expected_chosen,
            "15/122",
        )
        if not got_chosen[0]:
            return got_chosen
        return True, f"powers of two {got_pow[1]}; captured sizes {got_chosen[1]}"

    c.check("padding waste on the 200-step trace matches the slot totals", full_trace)

    def beats():
        buckets = getattr(submission, "BUCKETS", None)
        if not isinstance(buckets, (list, tuple)) or not buckets:
            return False, f"expected a non-empty BUCKETS list, got {buckets!r}"
        try:
            theirs = _ref_waste(TRACE, list(buckets))
            wide = _ref_waste(TRACE, POW2)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected waste below 201/736, got {type(exc).__name__}: {exc}"
        if not theirs < wide:
            return (
                False,
                f"expected waste below 201/736 ({wide!r}), got {theirs!r} for BUCKETS {list(buckets)!r}",
            )
        return True, f"expected waste below 201/736 ({wide!r}), got {theirs!r}"

    c.check("the chosen bucket set wastes less than powers of two", beats)
    buckets = getattr(submission, "BUCKETS", None)
    if isinstance(buckets, (list, tuple)) and buckets:
        try:
            c.metric("padding_waste", round(_ref_waste(TRACE, list(buckets)), 6))
        except Exception:
            pass

    c.check(
        "three of the four snippets fail the capture rules",
        lambda: _call(lambda: submission.failing_captures(SNIPPETS), FAILING),
    )

    if not _cuda_available():
        print(
            "      No CUDA device is visible, so the capture and replay checks are skipped.",
            flush=True,
        )
        c.metric("capture", "skipped")
        return c.finish()

    import torch

    def model_step(x, scale):
        return x * scale + x

    def _capture():
        torch.manual_seed(0)
        static_x = torch.randn(4, 8, device="cuda")
        static_scale = torch.randn(4, 8, device="cuda")
        captured = submission.capture_decode(model_step, (static_x, static_scale))
        return static_x, static_scale, captured

    def replay_matches():
        try:
            static_x, static_scale, captured = _capture()
        except Exception as exc:  # noqa: BLE001
            return False, f"expected 5 exact matches, got {type(exc).__name__}: {exc}"
        mismatches = []
        for index in range(5):
            torch.manual_seed(10 + index)
            fresh_x = torch.randn_like(static_x)
            fresh_scale = torch.randn_like(static_scale)
            eager = model_step(fresh_x, fresh_scale)
            static_x.copy_(fresh_x)
            static_scale.copy_(fresh_scale)
            try:
                actual = captured.replay()
            except Exception as exc:  # noqa: BLE001
                return False, f"expected exact match on content {index}, got {type(exc).__name__}: {exc}"
            if actual is None or actual.shape != eager.shape or not torch.equal(actual, eager):
                if actual is None or actual.shape != eager.shape:
                    got = None if actual is None else tuple(actual.shape)
                    mismatches.append(f"content {index}: expected shape {tuple(eager.shape)}, got {got}")
                else:
                    err = (actual - eager).abs().max().item()
                    mismatches.append(f"content {index}: expected max abs error 0, got {err:.3e}")
        if mismatches:
            return False, f"expected 5 exact matches, got {mismatches[0]}"
        return True, "expected 5 exact matches, got 5"

    c.check("replay matches eager execution on 5 input contents", replay_matches)

    def no_new_allocations():
        try:
            static_x, static_scale, captured = _capture()
        except Exception as exc:  # noqa: BLE001
            return False, f"expected allocation deltas [0, 0, 0, 0, 0], got {type(exc).__name__}: {exc}"
        deltas = []
        for index in range(5):
            torch.manual_seed(20 + index)
            fresh_x = torch.randn_like(static_x)
            fresh_scale = torch.randn_like(static_scale)
            static_x.copy_(fresh_x)
            static_scale.copy_(fresh_scale)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            try:
                captured.replay()
            except Exception as exc:  # noqa: BLE001
                return False, (
                    f"expected allocation deltas [0, 0, 0, 0, 0], got {type(exc).__name__}: {exc}"
                )
            torch.cuda.synchronize()
            deltas.append(torch.cuda.memory_allocated() - before)
        expected = [0, 0, 0, 0, 0]
        if deltas != expected:
            return False, f"expected {expected}, got {deltas}"
        return True, f"expected {expected}, got {deltas}"

    c.check("replay does not allocate", no_new_allocations)

    def _milliseconds(fn, iters: int) -> float:
        starter = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(end) / iters

    try:
        static_x, static_scale, captured = _capture()
        eager_ms = _milliseconds(lambda: model_step(static_x, static_scale), 30)
        graph_ms = _milliseconds(captured.replay, 30)
        speedup = eager_ms / graph_ms if graph_ms else 0.0
        c.metric("eager_ms", round(eager_ms, 4))
        c.metric("graph_ms", round(graph_ms, 4))
        c.metric("speedup", round(speedup, 3))
    except Exception as exc:  # noqa: BLE001
        c.add(
            "replay timing",
            False,
            f"expected eager_ms, graph_ms, and speedup, got {type(exc).__name__}: {exc}",
        )
    return c.finish()
