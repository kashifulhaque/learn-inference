"""Checks lab 14 against naive attention, then measures it against SDPA."""

import math

import torch
import torch.nn.functional as F
from lab_common import Checks, require_cuda

HEAD_DIM = 128
HEADS, KV_HEADS = 8, 2
GROUP = HEADS // KV_HEADS


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, repeats, s, d).reshape(b, h * repeats, s, d)


def naive(q, k, v, causal=True):
    """Ground truth, in float32, materializing the whole score matrix."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    k = repeat_kv(k, q.shape[1] // k.shape[1])
    v = repeat_kv(v, q.shape[1] // v.shape[1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        q_len, kv_len = scores.shape[-2], scores.shape[-1]
        offset = kv_len - q_len
        idx_q = torch.arange(q_len, device=scores.device).unsqueeze(-1)
        idx_k = torch.arange(kv_len, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(idx_k > idx_q + offset, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def run(submission):
    c = Checks()
    if not c.require(submission, "flash_attention"):
        return c.finish()

    device = require_cuda()
    torch.manual_seed(0)

    def tensors(q_len, kv_len, dtype=torch.float32):
        q = torch.randn(2, HEADS, q_len, HEAD_DIM, device=device, dtype=dtype)
        k = torch.randn(2, KV_HEADS, kv_len, HEAD_DIM, device=device, dtype=dtype)
        v = torch.randn(2, KV_HEADS, kv_len, HEAD_DIM, device=device, dtype=dtype)
        return q, k, v

    worst = 0.0
    cases = [
        ("non-causal, square", 256, 256, False),
        ("causal, square", 256, 256, True),
        ("causal, decode step behind a long prefix", 1, 1024, True),
        ("causal, chunk behind a cached prefix", 64, 512, True),
        ("causal, length not a multiple of the block", 130, 130, True),
        ("causal, long", 1024, 1024, True),
    ]
    for label, q_len, kv_len, causal in cases:
        q, k, v = tensors(q_len, kv_len)
        mine = submission.flash_attention(q, k, v, causal=causal)
        reference = naive(q, k, v, causal=causal)
        err = (mine.float() - reference).abs().max().item()
        worst = max(worst, err)
        c.check(f"{label}", lambda e=err: (e < 1e-3, f"max abs error {e:.2e}"))

    q, k, v = tensors(256, 256, torch.bfloat16)
    mine = submission.flash_attention(q, k, v, causal=True)
    ref = naive(q, k, v, causal=True)
    err_bf16 = (mine.float() - ref).abs().max().item()
    c.check("bfloat16 stays within tolerance",
            lambda: (err_bf16 < 5e-2, f"max abs error {err_bf16:.2e}"))

    # A single decode query must match the last row of a full prefill.
    q, k, v = tensors(128, 128)
    full = submission.flash_attention(q, k, v, causal=True)
    step = submission.flash_attention(q[:, :, -1:], k, v, causal=True)
    step_err = (step[:, :, 0].float() - full[:, :, -1].float()).abs().max().item()
    c.check("a decode step matches the prefill's last row",
            lambda: (step_err < 1e-3, f"max abs error {step_err:.2e}"))

    from engine.bench import benchmark

    results = {}
    for seq in (512, 2048, 8192):
        q = torch.randn(1, HEADS, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, KV_HEADS, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, KV_HEADS, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)

        flash_t = benchmark(
            lambda: submission.flash_attention(q, k, v, causal=True), f"flash{seq}",
            warmup=3, runs=10)
        sdpa_t = benchmark(
            lambda: F.scaled_dot_product_attention(
                q, repeat_kv(k, GROUP), repeat_kv(v, GROUP), is_causal=True),
            f"sdpa{seq}", warmup=3, runs=10)
        results[seq] = (flash_t.median_ms, sdpa_t.median_ms)
        c.metric(f"flash_ms_{seq}", flash_t.median_ms)
        c.metric(f"sdpa_ms_{seq}", sdpa_t.median_ms)

    # Peak memory: naive is quadratic in sequence length, flash is linear.
    seq = 4096
    q = torch.randn(1, HEADS, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, KV_HEADS, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, KV_HEADS, seq, HEAD_DIM, device=device, dtype=torch.bfloat16)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    submission.flash_attention(q, k, v, causal=True)
    torch.cuda.synchronize()
    flash_peak = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    naive(q, k, v, causal=True)
    torch.cuda.synchronize()
    naive_peak = torch.cuda.max_memory_allocated()

    ratio = naive_peak / max(flash_peak, 1)
    c.check(
        "flash uses far less memory than materializing the scores",
        lambda: ratio > 3.0,
        f"{naive_peak / 1024**2:.0f} MiB naive vs {flash_peak / 1024**2:.0f} MiB flash",
    )

    speedup_8k = results[8192][1] / results[8192][0]
    c.metric("max_error", float(f"{worst:.3e}"))
    c.metric("speedup_8k", round(speedup_8k, 2))
    c.metric("peak_memory_ratio", round(ratio, 1))
    return c.finish()
