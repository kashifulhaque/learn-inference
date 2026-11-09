"""Checks for the log-sum-exp merge and the split-count heuristic.

The first check calls only `merge_partials`. Later checks call
`choose_num_splits`. `split_kv_decode` runs only when CUDA is available.

A plausible wrong implementation adds the partial outputs and divides by the
split count, without rescaling each partial by exp(lse_i - lse). On the
scores [0, 2] and [1] with values 4, 6, and 10, that mean is about 7.881
and the full softmax is about 6.799. The unequal-chunk check fails it.
Random scores use seed 14.
"""

from __future__ import annotations

import torch

from lab_common import Checks

# scripts/figures/14a-split-kv-decode-attention.py prints these.
# (batch, heads, kv_len, num_sms) -> splits
_SPLITS = {
    (1, 24, 1, 108): 1,
    (1, 24, 17, 108): 1,
    (1, 24, 64, 108): 1,
    (1, 24, 65, 108): 2,
    (1, 24, 128, 108): 2,
    (1, 24, 192, 108): 3,
    (1, 24, 256, 108): 4,
    (1, 24, 4096, 108): 4,
    (1, 24, 32768, 108): 4,
    (1, 24, 65536, 108): 4,
    (1, 108, 65536, 108): 1,
    (2, 54, 100, 108): 1,
    (4, 40, 10, 108): 1,
    (8, 24, 65536, 108): 1,
    (1, 87, 65536, 108): 1,
    (1, 86, 65536, 108): 5,
    (1, 1, 64, 108): 1,
    (1, 1, 65536, 108): 92,
    (2, 8, 8192, 108): 6,
}


def _softmax(scores, values):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    peak = scores.max()
    weights = torch.exp(scores - peak)
    denom = weights.sum()
    out = (weights[:, None] * values).sum(dim=0) / denom
    return out, peak + torch.log(denom)


def _partials(chunks):
    outs = []
    lses = []
    for scores, values in chunks:
        if len(scores) == 0:
            outs.append(torch.zeros(len(values[0]) if values else 1, dtype=torch.float32))
            lses.append(torch.tensor(float("-inf")))
            continue
        out, lse = _softmax(scores, values)
        outs.append(out)
        lses.append(lse)
    return torch.stack(outs), torch.stack(lses)


def _show(out, lse) -> str:
    out_v = [round(float(x), 6) for x in torch.as_tensor(out).flatten()]
    lse_v = round(float(torch.as_tensor(lse).flatten()[0]), 6) if torch.as_tensor(lse).numel() == 1 else [
        round(float(x), 6) for x in torch.as_tensor(lse).flatten()
    ]
    return f"out {out_v}, lse {lse_v}"


def _close(actual_out, actual_lse, exp_out, exp_lse, tol=1e-5):
    out_err = (torch.as_tensor(actual_out).float() - exp_out.float()).abs().max().item()
    lse_a = torch.as_tensor(actual_lse).float()
    lse_e = torch.as_tensor(exp_lse).float()
    if lse_e.numel() == 1 and not torch.isfinite(lse_e):
        lse_ok = not torch.isfinite(lse_a) and float(lse_a) < 0
        lse_err = 0.0 if lse_ok else float("inf")
    else:
        lse_err = (lse_a - lse_e).abs().max().item()
        lse_ok = lse_err <= tol
    ok = out_err <= tol and lse_ok
    detail = (
        f"expected {_show(exp_out, exp_lse)}, got {_show(actual_out, actual_lse)}"
        f" (out error {out_err:.3e}, lse error {lse_err:.3e})"
    )
    return ok, detail


def _call_merge(fn, outs, lses):
    try:
        return fn(outs, lses), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def run(submission):
    c = Checks()

    def two_equal_chunks():
        # Scores [0, 2] and [1], values 4, 6, and 10. The full softmax is the
        # expected pair. An unscaled mean of the two partial outputs is not.
        chunks = [([0.0, 2.0], [[4.0], [6.0]]), ([1.0], [[10.0]])]
        outs, lses = _partials(chunks)
        expected_out, expected_lse = _softmax([0.0, 2.0, 1.0], [[4.0], [6.0], [10.0]])
        wrong = outs.mean(dim=0)
        if torch.allclose(wrong, expected_out, atol=1e-4):
            return False, "expected the unscaled mean to miss the softmax, and it matched"
        got, err = _call_merge(submission.merge_partials, outs, lses)
        if err:
            return False, f"expected {_show(expected_out, expected_lse)}, got {err}"
        ok, detail = _close(got[0], got[1], expected_out, expected_lse)
        if ok:
            err_value = (got[0].float() - expected_out).abs().max().item()
            c.metric("merge_error", err_value)
        return ok, detail

    c.check("merge_partials matches a two-chunk softmax", two_equal_chunks)

    def unequal_and_empty():
        scores = [0.5, -1.0, 2.0, 0.25, 1.5, -0.5]
        values = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.25, -0.25], [2.0, -1.0]]
        chunks = [
            (scores[:1], values[:1]),
            (scores[1:4], values[1:4]),
            ([], [[0.0, 0.0]]),
            (scores[4:], values[4:]),
        ]
        outs, lses = _partials(chunks)
        if not (lses[2] == float("-inf")):
            return False, "expected the empty chunk's lse to be -inf"
        expected_out, expected_lse = _softmax(scores, values)
        got, err = _call_merge(submission.merge_partials, outs, lses)
        if err:
            return False, f"expected {_show(expected_out, expected_lse)}, got {err}"
        return _close(got[0], got[1], expected_out, expected_lse)

    c.check(
        "merge_partials matches unequal chunks and ignores an empty chunk",
        unequal_and_empty,
    )

    def only_empty():
        outs = torch.zeros(2, 3)
        lses = torch.tensor([float("-inf"), float("-inf")])
        expected_out = torch.zeros(3)
        expected_lse = torch.tensor(float("-inf"))
        got, err = _call_merge(submission.merge_partials, outs, lses)
        if err:
            return False, f"expected {_show(expected_out, expected_lse)}, got {err}"
        return _close(got[0], got[1], expected_out, expected_lse)

    c.check("merge_partials of only empty chunks returns zeros and -inf", only_empty)

    def _pair(left, right):
        out_l, lse_l = left
        out_r, lse_r = right
        return _call_merge(
            submission.merge_partials,
            torch.stack([torch.as_tensor(out_l).float(), torch.as_tensor(out_r).float()]),
            torch.stack([
                torch.as_tensor(lse_l).float().reshape(()),
                torch.as_tensor(lse_r).float().reshape(()),
            ]),
        )

    def associative():
        generator = torch.Generator().manual_seed(14)
        scores = torch.randn(12, generator=generator)
        values = torch.randn(12, 4, generator=generator)
        groups = [scores[:3], scores[3:4], scores[4:]]
        rows = [values[:3], values[3:4], values[4:]]
        parts = [_softmax(group, row) for group, row in zip(groups, rows)]
        full_out, full_lse = _softmax(scores, values)
        left_ab, err = _pair(parts[0], parts[1])
        if err:
            return False, f"expected {_show(full_out, full_lse)}, got {err}"
        left, err = _pair(left_ab, parts[2])
        if err:
            return False, f"expected {_show(full_out, full_lse)}, got {err}"
        right_bc, err = _pair(parts[1], parts[2])
        if err:
            return False, f"expected {_show(full_out, full_lse)}, got {err}"
        right, err = _pair(parts[0], right_bc)
        if err:
            return False, f"expected {_show(full_out, full_lse)}, got {err}"
        left_ok, left_detail = _close(left[0], left[1], full_out, full_lse, tol=1e-4)
        right_ok, right_detail = _close(right[0], right[1], full_out, full_lse, tol=1e-4)
        if not (left_ok and right_ok):
            return False, f"expected {_show(full_out, full_lse)}; left {left_detail}; right {right_detail}"
        return True, f"expected {_show(full_out, full_lse)}, got {_show(left[0], left[1])}"

    c.check("merge_partials is associative on three seeded chunks", associative)

    def fills_the_sms():
        cases = [(1, 108, 65536, 108), (2, 54, 100, 108), (4, 40, 10, 108), (8, 24, 65536, 108), (1, 87, 65536, 108)]
        try:
            got = [submission.choose_num_splits(*case) for case in cases]
        except Exception as exc:  # noqa: BLE001
            return False, f"expected 1 for each full grid, got {type(exc).__name__}: {exc}"
        expected = [1, 1, 1, 1, 1]
        return got == expected, f"expected {expected}, got {got}"

    c.check("choose_num_splits returns 1 when batch times heads covers the SMs", fills_the_sms)

    def split_table():
        try:
            got = {key: submission.choose_num_splits(*key) for key in _SPLITS}
        except Exception as exc:  # noqa: BLE001
            return False, f"expected {_SPLITS}, got {type(exc).__name__}: {exc}"
        mismatches = {key: (got[key], _SPLITS[key]) for key in _SPLITS if got[key] != _SPLITS[key]}
        if mismatches:
            return False, f"expected {_SPLITS}, got {got}"
        return True, f"expected {_SPLITS}, got {got}"

    c.check("choose_num_splits matches the occupancy table", split_table)

    def monotone():
        lengths = list(range(1, 4097))
        try:
            series = [submission.choose_num_splits(1, 24, length, 108) for length in lengths]
            other = [submission.choose_num_splits(1, 1, length, 108) for length in lengths]
        except Exception as exc:  # noqa: BLE001
            return False, f"expected a non-decreasing series, got {type(exc).__name__}: {exc}"
        for name, values in (("24 heads", series), ("1 head", other)):
            for index, (left, right) in enumerate(zip(values, values[1:])):
                if right < left:
                    return (
                        False,
                        f"expected non-decreasing splits for {name}, "
                        f"got {right} after {left} at kv_len {index + 2}",
                    )
        anchors = [(64, 1), (65, 2), (192, 3), (256, 4), (4096, 4)]
        got_anchors = [(length, series[length - 1]) for length, _ in anchors]
        if got_anchors != anchors:
            return False, f"expected {anchors}, got {got_anchors}"
        return True, f"expected {anchors} and a non-decreasing sweep, got {got_anchors}"

    c.check("choose_num_splits is monotone in KV length", monotone)

    def triton_kernel():
        if not torch.cuda.is_available():
            return True, "expected skip, got skip (CUDA is not available)"
        shapes = [
            (1, 1, 1, 64),
            (2, 4, 17, 64),
            (1, 2, 128, 64),
            (1, 1, 4096, 64),
            (1, 1, 65536, 64),
        ]
        try:
            return _triton_cases(submission, shapes, torch.device("cuda"))
        except Exception as exc:  # noqa: BLE001
            return False, f"expected max abs error <= 1.0e-2, got {type(exc).__name__}: {exc}"

    def _triton_cases(submission, shapes, device):
        torch.manual_seed(14)
        worst = 0.0
        where = ""
        for batch, heads, kv_len, dim in shapes:
            query = torch.randn(batch, heads, dim, device=device)
            key = torch.randn(batch, heads, kv_len, dim, device=device)
            value = torch.randn(batch, heads, kv_len, dim, device=device)
            reference = torch.nn.functional.scaled_dot_product_attention(
                query[:, :, None, :], key, value
            ).squeeze(2)
            split_choices = {1, 3, submission.choose_num_splits(batch, heads, kv_len, 108)}
            for splits in split_choices:
                actual = submission.split_kv_decode(query, key, value, splits)
                err = (actual.float() - reference.float()).abs().max().item()
                if err > worst:
                    worst = err
                    where = f"batch {batch}, heads {heads}, kv_len {kv_len}, splits {splits}"
        limit = 1e-2
        return worst <= limit, f"expected max abs error <= {limit:.1e}, got {worst:.3e} at {where}"

    c.check("split_kv_decode matches scaled dot-product attention", triton_kernel)

    return c.finish()
