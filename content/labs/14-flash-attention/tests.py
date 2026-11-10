"""Checks lab 14.

The launch-grid and rescale-count checks run on any machine. The kernel checks
compare against naive attention and then measure against SDPA, and they skip
when no CUDA device is visible.
"""

import math

import torch
import torch.nn.functional as F
from lab_common import Checks, require_cuda

HEAD_DIM = 128
HEADS, KV_HEADS = 8, 2
GROUP = HEADS // KV_HEADS

# The model from chapter 14a: 24 query heads, 4 KV heads, head dimension 256.
MODEL_HEADS, MODEL_KV_HEADS, MODEL_HEAD_DIM = 24, 4, 256
SMS = 108


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


def _cpu_checks(c, submission):
    """Launch-grid and rescale-count checks. No GPU, no Triton."""

    def eq(label, got, expected):
        return got == expected, f"{label}: expected {expected}, got {got}"

    # Wrong implementation caught here: counting blocks over KV tiles instead
    # of query tiles. At q_len = 1 and kv_len = 32768 that returns
    # 24 * ceil(32768 / 128) = 6144 programs, and the SM-fill check then says
    # the decode launch fills the GPU, which is the opposite of the truth.
    def prefill_grid():
        got = submission.count_programs(1, MODEL_HEADS, 32768, 32768, 128)
        return eq("programs, prefill 32768 tokens, 24 heads, block_m 128", got, 6144)

    def decode_grid():
        got = submission.count_programs(1, MODEL_HEADS, 1, 32768, 128)
        return eq("programs, decode q_len 1 at kv_len 32768, 24 heads", got, MODEL_HEADS)

    def grid_ignores_kv():
        got = [submission.count_programs(1, MODEL_HEADS, 1, n, 128) for n in (1, 1024, 32768, 65536)]
        return eq("programs for decode at kv_len 1, 1024, 32768, 65536", got, [MODEL_HEADS] * 4)

    def grid_rounds_up():
        got = [
            submission.count_programs(2, 8, 130, 130, 128),
            submission.count_programs(1, 8, 128, 128, 128),
            submission.count_programs(1, 8, 129, 129, 128),
        ]
        return eq("programs for (2, 8, 130), (1, 8, 128), (1, 8, 129)", got, [32, 8, 16])

    def fill():
        got = [submission.sm_fill(p) for p in (24, 32, 108, 6144)]
        expected = [24 / 108, 32 / 108, 1.0, 1.0]
        ok = all(abs(a - b) < 1e-9 for a, b in zip(got, expected))
        got_text = [round(g, 4) for g in got]
        exp_text = [round(e, 4) for e in expected]
        return ok, f"fill for 24, 32, 108, 6144 programs: expected {exp_text}, got {got_text}"

    def fill_custom_sms():
        got = submission.sm_fill(24, num_sms=48)
        return abs(got - 0.5) < 1e-9, f"fill for 24 programs on 48 SMs: expected 0.5, got {got}"

    def lab_shape_fill():
        # The lab's 512-token benchmark: 8 heads, block_m 128 launches 4 * 8 programs.
        programs = submission.count_programs(1, HEADS, 512, 512, 128)
        got = submission.sm_fill(programs)
        expected = 32 / 108
        return (
            programs == 32 and abs(got - expected) < 1e-9,
            f"512 tokens, 8 heads: expected 32 programs and fill {expected:.4f}, "
            f"got {programs} programs and fill {got:.4f}",
        )

    c.check("a long prefill launches one program per query tile and head", prefill_grid)
    c.check("a decode step launches batch * heads programs, 24 here", decode_grid)
    c.check("the grid does not depend on the KV length", grid_ignores_kv)
    c.check("a partial query tile still launches a program", grid_rounds_up)
    c.check("sm_fill is the occupied fraction, capped at 1", fill)
    c.check("sm_fill honours num_sms", fill_custom_sms)
    c.check("the 512-token benchmark fills 32 of 108 SMs", lab_shape_fill)

    # Wrong implementation caught here: normalising every tile (FlashAttention-1)
    # where FlashAttention-2 defers. Version 2 would then return 2 scalings per
    # tile, 136,314,880 elements at L = 8192, instead of 69,206,016.
    def fa2_square():
        got = submission.rescale_elements(8192, 8192, 128, 64, 128, version=2)
        return eq("FA-2 scaled elements, L 8192 causal, 128x64 tiles, d 128", got, 69_206_016)

    def fa1_square():
        got = submission.rescale_elements(8192, 8192, 128, 64, 128, version=1)
        return eq("FA-1 scaled elements, same shape", got, 136_314_880)

    def ragged():
        got = [
            submission.rescale_elements(130, 130, 128, 64, 8, version=v) for v in (1, 2)
        ]
        return eq("(FA-1, FA-2) elements, L 130 causal, d 8", got, [4192, 3136])

    def non_causal():
        got = [
            submission.rescale_elements(256, 256, 128, 64, 64, version=v, causal=False)
            for v in (1, 2)
        ]
        return eq("(FA-1, FA-2) elements, L 256 non-causal, d 64", got, [131072, 81920])

    def single_tile():
        got = [
            submission.rescale_elements(128, 64, 128, 64, 128, version=v, causal=False)
            for v in (1, 2)
        ]
        return eq("(FA-1, FA-2) with one KV tile, equal because of the final divide", got, [32768, 32768])

    c.check("FlashAttention-2 defers the divide: 69,206,016 elements at L 8192", fa2_square)
    c.check("FlashAttention-1 scales both terms: 136,314,880 elements at L 8192", fa1_square)
    c.check("rescale counts use the real row count of a partial tile", ragged)
    c.check("rescale counts without a causal mask", non_causal)
    c.check("with one KV tile the two versions cost the same", single_tile)


def run(submission):
    c = Checks()
    if not c.require(submission, "flash_attention", "count_programs", "sm_fill", "rescale_elements"):
        return c.finish()

    _cpu_checks(c, submission)

    if not torch.cuda.is_available():
        c.add(
            "kernel and benchmark checks",
            True,
            "expected skip, got skip (CUDA is not available)",
        )
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
        # tl.dot runs float32 inputs through the TF32 tensor cores, which have
        # 10 mantissa bits, so about 1e-3 of relative error is expected and is
        # not a bug. Anything much larger is.
        c.check(f"{label}", lambda e=err: (e < 5e-3, f"max abs error {e:.2e}"))

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
            lambda: (step_err < 5e-3, f"max abs error {step_err:.2e}"))

    # Head dimensions 64, 128, and 256 in bfloat16. Width 256 uses a 64-row
    # query tile so the accumulator and the staged K and V tiles fit.
    for hd in (64, 128, 256):
        qh = torch.randn(1, HEADS, 512, hd, device=device, dtype=torch.bfloat16)
        kh = torch.randn(1, KV_HEADS, 512, hd, device=device, dtype=torch.bfloat16)
        vh = torch.randn(1, KV_HEADS, 512, hd, device=device, dtype=torch.bfloat16)
        mine = submission.flash_attention(qh, kh, vh, causal=True, block_m=64 if hd == 256 else 128)
        err_hd = (mine.float() - naive(qh, kh, vh, causal=True)).abs().max().item()
        c.check(f"head_dim {hd} stays within tolerance",
                lambda e=err_hd: (e < 5e-2, f"max abs error {e:.2e}"))

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

    # Head-dimension sweep at 4096 tokens, then the decode shape. Neither has a
    # pass threshold: they exist to show where the 8k number stops holding.
    def sweep_ratio(q, k, v, label, **kwargs):
        flash_t = benchmark(
            lambda: submission.flash_attention(q, k, v, causal=True, **kwargs), f"flash_{label}",
            warmup=3, runs=10)
        sdpa_t = benchmark(
            lambda: F.scaled_dot_product_attention(
                q, repeat_kv(k, q.shape[1] // k.shape[1]), repeat_kv(v, q.shape[1] // v.shape[1]),
                is_causal=True),
            f"sdpa_{label}", warmup=3, runs=10)
        return sdpa_t.median_ms / flash_t.median_ms, flash_t.median_ms

    for hd in (64, 128, 256):
        qh = torch.randn(1, HEADS, 4096, hd, device=device, dtype=torch.bfloat16)
        kh = torch.randn(1, KV_HEADS, 4096, hd, device=device, dtype=torch.bfloat16)
        vh = torch.randn(1, KV_HEADS, 4096, hd, device=device, dtype=torch.bfloat16)
        ratio_hd, _ = sweep_ratio(qh, kh, vh, f"hd{hd}", block_m=64 if hd == 256 else 128)
        c.metric(f"speedup_hd{hd}_4k", round(ratio_hd, 2))

    # Decode: one query token behind 32,768 cached tokens, on the chapter 14a
    # model. The grid is 24 programs for 108 SMs.
    q = torch.randn(1, MODEL_HEADS, 1, MODEL_HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, MODEL_KV_HEADS, 32768, MODEL_HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, MODEL_KV_HEADS, 32768, MODEL_HEAD_DIM, device=device, dtype=torch.bfloat16)
    mine = submission.flash_attention(q, k, v, causal=True, block_m=64)
    err_decode = (mine.float() - naive(q, k, v, causal=True)).abs().max().item()
    c.check("decode at 32k on the 24-head model stays within tolerance",
            lambda: (err_decode < 5e-2, f"max abs error {err_decode:.2e}"))
    speedup_decode_32k, decode_ms = sweep_ratio(q, k, v, "decode32k", block_m=64)
    programs = submission.count_programs(1, MODEL_HEADS, 1, 32768, 64)
    c.metric("decode_programs", programs)
    c.metric("decode_sm_fill", round(submission.sm_fill(programs), 3))
    c.metric("flash_ms_decode_32k", decode_ms)

    speedup_8k = results[8192][1] / results[8192][0]
    # PyTorch dispatches scaled_dot_product_attention to a vendor kernel that
    # has had far more tuning than this one. Landing within about 2x of it is a
    # good result for a kernel you can read in one sitting; the memory
    # behaviour, not the clock, is what this lab is for.
    c.check(
        "flash lands within 3x of the vendor kernel at 8k",
        lambda: speedup_8k > 0.33,
        f"{speedup_8k:.2f}x of scaled_dot_product_attention",
    )
    c.metric("max_error", float(f"{worst:.3e}"))
    c.metric("speedup_8k", round(speedup_8k, 2))
    c.metric("peak_memory_ratio", round(ratio, 1))
    c.metric("speedup_decode_32k", round(speedup_decode_32k, 2))
    return c.finish()
