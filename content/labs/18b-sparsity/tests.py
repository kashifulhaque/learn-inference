"""Checks for magnitude and Wanda pruning, 2:4 sparsity, and the KD loss.

Plausible wrong implementations, each run against these checks:

* `wanda_scores` that forgets the activations and returns the magnitude. The
  two methods then tie, so the "Wanda beats magnitude" checks fail. On the
  seeded layer both errors are 0.248337, and the check wants Wanda under
  half of the magnitude error.
* A 2:4 mask built by keeping the top half of each row, not of each group of
  four. The mask has the right density and the wrong structure, so
  `is_valid_2_4` must reject it and `prune_2_4` must not return one.
* `kd_loss` without the T^2 factor. At T = 2 its gradient is a quarter of
  the one in Hinton et al., and the analytic-gradient check fails.
"""

from __future__ import annotations

import math

import torch

from lab_common import Checks


def _call(submission, name, *args, **kwargs):
    try:
        return getattr(submission, name)(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _layer():
    """Seed 18: 32 outputs, 64 inputs, 256 tokens, four outlier channels."""
    g = torch.Generator().manual_seed(18)
    weight = torch.randn(32, 64, generator=g, dtype=torch.float64)
    inputs = torch.randn(256, 64, generator=g, dtype=torch.float64)
    inputs[:, [3, 17, 40, 58]] *= 12.0
    return weight, inputs


def _rel_error(weight, pruned, inputs) -> float:
    ref = inputs @ weight.T
    return float((ref - inputs @ pruned.T).norm() / ref.norm())


def _kd_reference(student, teacher, temperature):
    """Pure-Python reference on lists of lists."""

    def softmax(row):
        scaled = [v / temperature for v in row]
        peak = max(scaled)
        exps = [math.exp(v - peak) for v in scaled]
        total = sum(exps)
        return [e / total for e in exps]

    total = 0.0
    for s_row, t_row in zip(student, teacher):
        p_s = softmax(s_row)
        p_t = softmax(t_row)
        total += sum(pt * math.log(pt / ps) for pt, ps in zip(p_t, p_s) if pt > 0)
    return temperature**2 * total / len(student)


def run(submission):
    c = Checks()
    needed = [
        "magnitude_scores",
        "wanda_scores",
        "prune_rowwise",
        "prune_2_4",
        "is_valid_2_4",
        "compress_2_4",
        "decompress_2_4",
        "compressed_bytes",
        "output_error",
        "kd_loss",
        "sparse_semi_structured_check",
    ]
    if not c.require(submission, *needed):
        return c.finish()

    def magnitude_worked():
        weight = torch.tensor([[0.9, -0.1, 0.4, -0.7]], dtype=torch.float64)
        mag, err = _call(submission, "magnitude_scores", weight)
        if err:
            return False, f"expected [[0.9, 0.1, 0.4, 0.7]], got {err}"
        got = torch.as_tensor(mag).double()
        return torch.allclose(got, weight.abs()), f"expected [[0.9, 0.1, 0.4, 0.7]], got {got.tolist()}"

    c.check("magnitude_scores returns the absolute value", magnitude_worked)

    def wanda_scores_worked():
        weight = torch.tensor([[0.9, -0.1, 0.4, -0.7]], dtype=torch.float64)
        # Column norms 1, 1, 3, 1: one token holds those values, one token is zero.
        inputs = torch.tensor([[1.0, 1.0, 3.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        expected = [[0.9, 0.1, 1.2, 0.7]]
        got, err = _call(submission, "wanda_scores", weight, inputs)
        if err:
            return False, f"expected {expected}, got {err}"
        got = torch.as_tensor(got).double()
        if got.shape != weight.shape or not torch.allclose(got, torch.tensor(expected, dtype=torch.float64), atol=1e-9):
            return False, f"expected {expected}, got {got.tolist()}"
        # Norm over tokens, not a mean: two identical tokens scale the norm by sqrt(2).
        both = torch.cat([inputs[:1], inputs[:1]])
        got2, err = _call(submission, "wanda_scores", weight, both)
        want2 = torch.tensor(expected, dtype=torch.float64) * 2**0.5
        if err:
            return False, f"expected {want2.tolist()}, got {err}"
        if not torch.allclose(torch.as_tensor(got2).double(), want2, atol=1e-9):
            return False, f"expected {want2.tolist()}, got {torch.as_tensor(got2).tolist()}"
        return True, "expected [[0.9, 0.1, 1.2, 0.7]], got the same"

    c.check("wanda_scores multiplies the magnitude by the input-channel L2 norm", wanda_scores_worked)

    def rowwise_keeps_top_scores():
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float64)
        scores = torch.tensor([[0.5, 0.9, 0.1, 0.9], [3.0, 1.0, 2.0, 4.0]], dtype=torch.float64)
        # Row 0 drops the two lowest scores: 0.1 (index 2) and 0.5 (index 0).
        # Row 1 drops 1.0 (index 1) and 2.0 (index 2).
        expected_mask = [[False, True, False, True], [True, False, False, True]]
        got, err = _call(submission, "prune_rowwise", weight, scores, 0.5)
        if err:
            return False, f"expected mask {expected_mask}, got {err}"
        pruned, mask = got
        if torch.as_tensor(mask).bool().tolist() != expected_mask:
            return False, f"expected mask {expected_mask}, got {torch.as_tensor(mask).tolist()}"
        want = weight * torch.tensor(expected_mask, dtype=torch.float64)
        if not torch.allclose(torch.as_tensor(pruned).double(), want):
            return False, f"expected {want.tolist()}, got {torch.as_tensor(pruned).tolist()}"
        # A tie drops the lower index first.
        tied = torch.ones(1, 4, dtype=torch.float64)
        got, err = _call(submission, "prune_rowwise", tied, tied, 0.5)
        if err:
            return False, f"expected mask [[False, False, True, True]], got {err}"
        if torch.as_tensor(got[1]).bool().tolist() != [[False, False, True, True]]:
            return False, f"expected mask [[False, False, True, True]], got {torch.as_tensor(got[1]).tolist()}"
        weight, inputs = _layer()
        _, mask = submission.prune_rowwise(weight, submission.magnitude_scores(weight), 0.5)
        kept = torch.as_tensor(mask).bool().sum(dim=1).tolist()
        if kept != [32] * 32:
            return False, f"expected 32 kept per row, got {kept}"
        return True, "expected the lowest scores of each row dropped, got the same"

    c.check("prune_rowwise drops the lowest scores of every output row", rowwise_keeps_top_scores)

    def prune_24_valid():
        weight = torch.tensor([[0.9, -0.1, 0.4, -0.7]], dtype=torch.float64)
        got, err = _call(submission, "prune_2_4", weight)
        if err:
            return False, f"expected mask [[True, False, False, True]], got {err}"
        pruned, mask = got
        if torch.as_tensor(mask).bool().tolist() != [[True, False, False, True]]:
            return False, f"expected mask [[True, False, False, True]], got {torch.as_tensor(mask).tolist()}"
        scores = torch.tensor([[0.9, 0.1, 1.2, 0.7]], dtype=torch.float64)
        got, err = _call(submission, "prune_2_4", weight, scores)
        if err:
            return False, f"expected mask [[True, False, True, False]], got {err}"
        if torch.as_tensor(got[1]).bool().tolist() != [[True, False, True, False]]:
            return False, f"expected mask [[True, False, True, False]], got {torch.as_tensor(got[1]).tolist()}"
        weight, inputs = _layer()
        pruned, mask = submission.prune_2_4(weight, submission.wanda_scores(weight, inputs))
        mask = torch.as_tensor(mask).bool()
        if not submission.is_valid_2_4(mask):
            return False, "expected every group of 4 to keep exactly 2, got a mask is_valid_2_4 rejects"
        # The pruned weight is the original where kept and 0 elsewhere.
        want = weight * mask.double()
        if not torch.allclose(torch.as_tensor(pruned).double(), want):
            return False, "expected pruned == weight * mask, got a different tensor"
        return True, "expected 2 of every 4 kept, got the same"

    c.check("prune_2_4 keeps two of every four and the mask is valid", prune_24_valid)

    def validity_rejects_wrong_structure():
        weight, _ = _layer()
        # Same 50% density, built per row: the groups of four are not 2 each.
        _, row_mask = submission.prune_rowwise(weight, weight.abs(), 0.5)
        row_mask = torch.as_tensor(row_mask).bool()
        if submission.is_valid_2_4(row_mask):
            return False, "expected a per-row 50% mask to fail is_valid_2_4, got True"
        good = torch.tensor([[True, True, False, False, False, True, False, True]])
        if not submission.is_valid_2_4(good):
            return False, "expected [[1,1,0,0,0,1,0,1]] to be valid, got False"
        three = torch.tensor([[True, True, True, False]])
        if submission.is_valid_2_4(three):
            return False, "expected 3 kept in one group to be invalid, got True"
        if submission.is_valid_2_4(torch.ones(1, 6, dtype=torch.bool)):
            return False, "expected a width that is not a multiple of 4 to be invalid, got True"
        _, m24 = submission.prune_2_4(weight)
        density = float(torch.as_tensor(m24).float().mean())
        return density == 0.5, f"expected density 0.5, got {density}"

    c.check("is_valid_2_4 rejects a mask with the right density and the wrong structure", validity_rejects_wrong_structure)

    def compressed_format():
        pruned = torch.tensor([[0.0, 5.0, 0.0, -2.0, 7.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        mask = torch.tensor([[False, True, False, True, True, False, False, True]])
        got, err = _call(submission, "compress_2_4", pruned, mask)
        if err:
            return False, f"expected values [[5, -2, 7, 1]] and indices [[1, 3, 0, 3]], got {err}"
        values, indices = got
        if torch.as_tensor(values).double().tolist() != [[5.0, -2.0, 7.0, 1.0]]:
            return False, f"expected values [[5.0, -2.0, 7.0, 1.0]], got {torch.as_tensor(values).tolist()}"
        if torch.as_tensor(indices).long().tolist() != [[1, 3, 0, 3]]:
            return False, f"expected indices [[1, 3, 0, 3]], got {torch.as_tensor(indices).tolist()}"
        # A kept weight that is exactly zero still takes a slot.
        z_pruned = torch.tensor([[0.0, 0.0, 3.0, 0.0]], dtype=torch.float64)
        z_mask = torch.tensor([[True, False, True, False]])
        v, i = submission.compress_2_4(z_pruned, z_mask)
        if torch.as_tensor(i).long().tolist() != [[0, 2]]:
            return False, f"expected indices [[0, 2]] for a kept zero, got {torch.as_tensor(i).tolist()}"
        weight, inputs = _layer()
        pruned, mask = submission.prune_2_4(weight, submission.wanda_scores(weight, inputs))
        v, i = submission.compress_2_4(pruned, mask)
        if tuple(v.shape) != (32, 32) or tuple(i.shape) != (32, 32):
            return False, f"expected shapes (32, 32), got {tuple(v.shape)} and {tuple(i.shape)}"
        if int(torch.as_tensor(i).max()) > 3 or int(torch.as_tensor(i).min()) < 0:
            return False, "expected indices in 0..3, got a value outside that range"
        back = submission.decompress_2_4(v, i, 64)
        if not torch.equal(torch.as_tensor(back).double(), pruned):
            return False, "expected decompress(compress(W)) == W, got a different tensor"
        stored, meta = submission.compressed_bytes(4096, 4096, 2)
        want = (4096 * 4096 // 2 * 2, 4096 * 4096 // 2 * 2 // 8)
        if (stored, meta) != want:
            return False, f"expected {want} bytes for 4096x4096 bf16, got {(stored, meta)}"
        per_weight = (stored + meta) / (4096 * 4096)
        return per_weight == 1.125, f"expected 1.125 bytes per weight, got {per_weight}"

    c.check("compress_2_4 stores values plus 2-bit indices and round-trips", compressed_format)

    def wanda_beats_magnitude():
        weight, inputs = _layer()
        mag_w, _ = submission.prune_rowwise(weight, submission.magnitude_scores(weight), 0.5)
        wan_w, _ = submission.prune_rowwise(weight, submission.wanda_scores(weight, inputs), 0.5)
        e_mag = _rel_error(weight, torch.as_tensor(mag_w).double(), inputs)
        e_wan = _rel_error(weight, torch.as_tensor(wan_w).double(), inputs)
        got = submission.output_error(weight, wan_w, inputs)
        if abs(got - e_wan) > 1e-9:
            return False, f"expected output_error {e_wan:.6f}, got {got:.6f}"
        c.metric("wanda_over_magnitude", round(e_wan / e_mag, 4))
        # Expected 0.091117 against 0.248337, a ratio of 0.367.
        ok = e_wan < 0.5 * e_mag and e_mag > 0.2
        return ok, f"expected Wanda error under half of magnitude error {e_mag:.6f}, got {e_wan:.6f}"

    c.check("Wanda keeps the layer output closer than magnitude at 50% sparsity", wanda_beats_magnitude)

    def wanda_beats_magnitude_24():
        weight, inputs = _layer()
        mag_w, _ = submission.prune_2_4(weight, submission.magnitude_scores(weight))
        wan_w, wan_m = submission.prune_2_4(weight, submission.wanda_scores(weight, inputs))
        e_mag = _rel_error(weight, torch.as_tensor(mag_w).double(), inputs)
        e_wan = _rel_error(weight, torch.as_tensor(wan_w).double(), inputs)
        if not submission.is_valid_2_4(torch.as_tensor(wan_m).bool()):
            return False, "expected a valid 2:4 mask, got an invalid one"
        # Expected 0.123251 against 0.332604.
        return e_wan < 0.5 * e_mag, f"expected Wanda 2:4 error under half of {e_mag:.6f}, got {e_wan:.6f}"

    c.check("Wanda also wins under the 2:4 constraint", wanda_beats_magnitude_24)

    def kd_checks():
        student = [[0.0, 0.0], [1.0, -1.0]]
        teacher = [[2.0, 0.0], [0.5, 0.5]]
        for temperature in (1.0, 2.0):
            expected = _kd_reference(student, teacher, temperature)
            got, err = _call(
                submission,
                "kd_loss",
                torch.tensor(student, dtype=torch.float64),
                torch.tensor(teacher, dtype=torch.float64),
                temperature,
            )
            if err:
                return False, f"expected {expected:.6f} at T={temperature}, got {err}"
            if abs(float(got) - expected) > 1e-9:
                return False, f"expected {expected:.6f} at T={temperature}, got {float(got):.6f}"
        same = torch.tensor(teacher, dtype=torch.float64)
        zero = float(submission.kd_loss(same.clone(), same, 3.0))
        if abs(zero) > 1e-12:
            return False, f"expected 0 when student equals teacher, got {zero}"

        g = torch.Generator().manual_seed(5)
        s = torch.randn(3, 6, generator=g, dtype=torch.float64)
        t = torch.randn(3, 6, generator=g, dtype=torch.float64)
        temperature = 2.0
        s_var = s.clone().requires_grad_(True)
        submission.kd_loss(s_var, t, temperature).backward()
        auto = s_var.grad.clone()
        # Analytic gradient: T * (softmax(s/T) - softmax(t/T)) / batch.
        analytic = temperature * (torch.softmax(s / temperature, -1) - torch.softmax(t / temperature, -1)) / 3
        if not torch.allclose(auto, analytic, atol=1e-9):
            return False, f"expected gradient {analytic[0].tolist()}, got {auto[0].tolist()}"
        # Finite differences of the submission's own loss.
        eps = 1e-6
        numeric = torch.zeros_like(s)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                up = s.clone()
                dn = s.clone()
                up[i, j] += eps
                dn[i, j] -= eps
                numeric[i, j] = (
                    float(submission.kd_loss(up, t, temperature)) - float(submission.kd_loss(dn, t, temperature))
                ) / (2 * eps)
        worst = float((numeric - auto).abs().max())
        t_var = t.clone().requires_grad_(True)
        if submission.kd_loss(s.clone(), t_var, temperature).requires_grad:
            return False, "expected the teacher outside the autograd graph, got a loss that depends on it"
        return worst < 1e-7, f"expected finite-difference error under 1e-7, got {worst:.2e}"

    c.check("kd_loss matches the reference, and its gradient matches T*(p_s - p_t)/B", kd_checks)

    def semi_structured():
        import torch as _t

        if not _t.cuda.is_available():
            return True, "expected skip, got skip (CUDA is not available)"
        g = torch.Generator().manual_seed(7)
        weight = torch.randn(128, 128, generator=g)
        weight, _ = submission.prune_2_4(weight)
        inputs = torch.randn(128, 128, generator=g)
        diff, err = _call(submission, "sparse_semi_structured_check", weight, inputs)
        if err:
            return False, f"expected a max abs difference under 0.1, got {err}"
        return diff is not None and diff < 0.1, f"expected max abs difference under 0.1, got {diff}"

    c.check("to_sparse_semi_structured matches the dense product when CUDA is available", semi_structured)
    return c.finish()
