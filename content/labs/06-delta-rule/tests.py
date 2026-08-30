"""Checks lab 06.

The chunked form is intricate enough that a sign error produces plausible
output. These checks pin it down from four directions: agreement with the
sequential loop, invariance to chunk size, correct resumption from a carried
state, and a chunked prefill followed by single decode steps.
"""

import torch
import torch.nn.functional as F
from lab_common import Checks, close, pick_device

BATCH, HEADS, SEQ, K_DIM, V_DIM = 2, 4, 131, 16, 24


def make_inputs(device, seed=0, seq=SEQ):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = F.normalize(torch.randn(BATCH, HEADS, seq, K_DIM, generator=gen), dim=-1)
    k = F.normalize(torch.randn(BATCH, HEADS, seq, K_DIM, generator=gen), dim=-1)
    v = torch.randn(BATCH, HEADS, seq, V_DIM, generator=gen)
    alpha = torch.rand(BATCH, HEADS, seq, generator=gen) * 0.4 + 0.6
    beta = torch.rand(BATCH, HEADS, seq, generator=gen)
    return (t.to(device) for t in (q, k, v, alpha, beta))


def run(submission):
    c = Checks()
    if not c.require(submission, "delta_rule_step", "delta_rule_recurrent",
                     "delta_rule_chunked"):
        return c.finish()

    device = pick_device()
    q, k, v, alpha, beta = make_inputs(device)

    out_ref, state_ref = submission.delta_rule_recurrent(q, k, v, alpha, beta)
    c.check(
        "the sequential form returns the right shapes",
        lambda: tuple(out_ref.shape) == (BATCH, HEADS, SEQ, V_DIM)
        and tuple(state_ref.shape) == (BATCH, HEADS, K_DIM, V_DIM),
        f"out {tuple(out_ref.shape)}, state {tuple(state_ref.shape)}",
    )

    # With alpha = 1 and beta = 1 the delta rule reduces to a form we can write
    # in closed form, which catches an inverted update before anything else.
    ones = torch.ones(BATCH, HEADS, 3, device=device)
    q1 = F.normalize(torch.randn(BATCH, HEADS, 3, K_DIM, device=device), dim=-1)
    k1 = F.normalize(torch.randn(BATCH, HEADS, 3, K_DIM, device=device), dim=-1)
    v1 = torch.randn(BATCH, HEADS, 3, V_DIM, device=device)
    o1, _ = submission.delta_rule_recurrent(q1, k1, v1, ones, ones)
    # The first step writes v_1 against key k_1, so o_1 = (q_1 . k_1) v_1.
    expected_first = (q1[:, :, 0] * k1[:, :, 0]).sum(-1, keepdim=True) * v1[:, :, 0]
    c.check(
        "the first output is (q . k) v when the gates are open",
        lambda: close(o1[:, :, 0], expected_first, 1e-4),
    )

    worst_out = 0.0
    for chunk in (1, 7, 32, 64, 256):
        out_c, state_c = submission.delta_rule_chunked(
            q, k, v, alpha, beta, chunk_size=chunk)
        err_o = (out_c - out_ref).abs().max().item()
        err_s = (state_c - state_ref).abs().max().item()
        worst_out = max(worst_out, err_o)
        c.check(
            f"chunk_size={chunk} matches the sequential form",
            lambda e=err_o, s=err_s: (max(e, s) < 1e-4, f"out {e:.2e}, state {s:.2e}"),
        )

    # Resuming from a carried state: this is prefill followed by more prefill.
    split = 50
    o1, s1 = submission.delta_rule_chunked(
        q[:, :, :split], k[:, :, :split], v[:, :, :split],
        alpha[:, :, :split], beta[:, :, :split], chunk_size=16)
    o2, s2 = submission.delta_rule_chunked(
        q[:, :, split:], k[:, :, split:], v[:, :, split:],
        alpha[:, :, split:], beta[:, :, split:], state=s1, chunk_size=16)
    resume_err = (torch.cat([o1, o2], dim=2) - out_ref).abs().max().item()
    c.check(
        "a split sequence resumes from the carried state",
        lambda: (resume_err < 1e-4, f"max abs error {resume_err:.2e}"),
    )

    # Chunked prefill followed by single steps: this is the real decode path.
    prefill = 100
    o_pre, state = submission.delta_rule_chunked(
        q[:, :, :prefill], k[:, :, :prefill], v[:, :, :prefill],
        alpha[:, :, :prefill], beta[:, :, :prefill], chunk_size=32)
    steps = []
    for t in range(prefill, SEQ):
        state, out = submission.delta_rule_step(
            state, q[:, :, t], k[:, :, t], v[:, :, t], alpha[:, :, t], beta[:, :, t])
        steps.append(out)
    decode_err = (
        torch.cat([o_pre, torch.stack(steps, dim=2)], dim=2) - out_ref
    ).abs().max().item()
    c.check(
        "chunked prefill then single steps matches one long run",
        lambda: (decode_err < 1e-4, f"max abs error {decode_err:.2e}"),
    )

    # A closed gate must erase the state.
    zero_alpha = torch.full((BATCH, HEADS, SEQ), 1e-8, device=device)
    _, closed = submission.delta_rule_chunked(q, k, v, zero_alpha, beta, chunk_size=32)
    _, closed_ref = submission.delta_rule_recurrent(q, k, v, zero_alpha, beta)
    c.check(
        "a nearly closed forget gate agrees between both forms",
        lambda: close(closed, closed_ref, 1e-4),
    )

    speedup = None
    if device.type == "cuda":
        from engine.bench import benchmark

        ql, kl, vl, al, bl = make_inputs(device, seed=1, seq=2048)
        seq_t = benchmark(
            lambda: submission.delta_rule_recurrent(ql, kl, vl, al, bl),
            "recurrent", warmup=1, runs=3)
        chunk_t = benchmark(
            lambda: submission.delta_rule_chunked(ql, kl, vl, al, bl, chunk_size=64),
            "chunked", warmup=2, runs=5)
        speedup = round(seq_t.median_ms / chunk_t.median_ms, 1)
        c.metric("recurrent_ms", seq_t.median_ms)
        c.metric("chunked_ms", chunk_t.median_ms)
        c.check(
            "the chunked form is faster on a 2048-token sequence",
            lambda: speedup > 5.0,
            f"{speedup}x",
        )

    c.metric("chunk_error", float(f"{worst_out:.3e}"))
    c.metric("resume_error", float(f"{resume_err:.3e}"))
    c.metric("decode_error", float(f"{decode_err:.3e}"))
    if speedup is not None:
        c.metric("speedup", speedup)
    return c.finish()
