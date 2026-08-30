import torch
import torch.nn.functional as F
from lab_common import Checks, close, require_cuda

# An A100 80GB PCIe is rated at 1935 GB/s. A plain copy reaches about
# 1275, which is the ceiling a real kernel is measured against.
PEAK_BANDWIDTH_GBS = 1935.0
COPY_CEILING_GBS = 1275.0
HIDDEN = 5120


def reference_norm(x, w, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        x.dtype
    )


def run(submission):
    c = Checks()
    if not c.require(submission, "rms_norm", "rms_norm_residual", "swiglu"):
        return c.finish()

    device = require_cuda()
    torch.manual_seed(0)

    for rows, cols in [(64, HIDDEN), (1, HIDDEN), (4096, HIDDEN), (13, 4097)]:
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)
        w = torch.randn(cols, device=device, dtype=torch.float32)
        c.check(
            f"rms_norm matches at ({rows}, {cols})",
            lambda x=x, w=w: close(submission.rms_norm(x, w), reference_norm(x, w), 1e-4),
        )

    x = torch.randn(2, 8, HIDDEN, device=device, dtype=torch.float32)
    w = torch.randn(HIDDEN, device=device, dtype=torch.float32)
    c.check(
        "rms_norm accepts a 3-D input and keeps its shape",
        lambda: close(submission.rms_norm(x, w), reference_norm(x, w), 1e-4),
    )

    xb = torch.randn(128, HIDDEN, device=device, dtype=torch.bfloat16)
    wb = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    out = submission.rms_norm(xb, wb)
    c.check("the output keeps the input dtype", lambda: out.dtype == torch.bfloat16,
            f"got {out.dtype}")

    x = torch.randn(1024, HIDDEN, device=device, dtype=torch.float32)
    residual = torch.randn_like(x)
    normed, new_residual = submission.rms_norm_residual(x, residual, w)
    c.check("the fused kernel returns the updated residual",
            lambda: close(new_residual, x + residual, 1e-4))
    c.check("the fused kernel normalizes the sum",
            lambda: close(normed, reference_norm(x + residual, w), 1e-4))

    gate_c = torch.randn(1024, 17408, device=device, dtype=torch.bfloat16)
    up_c = torch.randn_like(gate_c)
    c.check(
        "swiglu matches silu(gate) * up",
        lambda: close(submission.swiglu(gate_c, up_c).float(),
                      (F.silu(gate_c.float()) * up_c.float()), 5e-2),
    )

    from engine.bench import benchmark

    def fusion_at(rows: int) -> tuple[float, float, float]:
        """Time the fused and unfused paths at one size.

        Output buffers are reused. Allocating two large tensors per call costs
        about as much as the kernels do, which would measure the allocator
        instead of the memory traffic.
        """
        x = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        w = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        new_res = torch.empty_like(x)
        total = torch.empty_like(x)

        def fused():
            return submission.rms_norm_residual(
                x, residual, w, out=out, new_residual=new_res)

        def unfused():
            torch.add(x, residual, out=total)
            return submission.rms_norm(total, w, out=out), total

        fused_t = benchmark(fused, f"fused{rows}", warmup=10, runs=30)
        unfused_t = benchmark(unfused, f"unfused{rows}", warmup=10, runs=30)
        del x, residual, w, out, new_res, total
        torch.cuda.empty_cache()
        return fused_t.median_ms, unfused_t.median_ms, unfused_t.median_ms / fused_t.median_ms

    # Below L2 the unfused pair's intermediate never reaches HBM, so the traffic
    # the fusion saves was already free. Past L2 the saving is real. The A100
    # has 40 MB of L2; a 4096 x 5120 bfloat16 tensor is 42 MB, right at the
    # boundary, and 16384 rows is four times over it.
    small_f, small_u, small_speedup = fusion_at(4096)
    big_f, big_u, big_speedup = fusion_at(16384)

    print(
        f"\n  4096 rows (42 MB, about L2): fused {small_f:.4f} ms, "
        f"unfused {small_u:.4f} ms -> {small_speedup:.2f}x"
        f"\n 16384 rows (168 MB, past L2): fused {big_f:.4f} ms, "
        f"unfused {big_u:.4f} ms -> {big_speedup:.2f}x\n",
        flush=True,
    )

    x = torch.randn(4096, HIDDEN, device=device, dtype=torch.bfloat16)
    w = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    norm_bytes = 2 * x.numel() * x.element_size()
    norm_t = benchmark(lambda: submission.rms_norm(x, w, out=out), "rms_norm",
                       warmup=10, runs=30)
    achieved = norm_bytes / (norm_t.median_ms / 1000) / 1e9

    gate = torch.randn(4096, 17408, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    swiglu_t = benchmark(lambda: submission.swiglu(gate, up), "swiglu",
                         warmup=5, runs=20)
    swiglu_ref_t = benchmark(lambda: F.silu(gate) * up, "swiglu_torch",
                             warmup=5, runs=20)
    swiglu_speedup = swiglu_ref_t.median_ms / swiglu_t.median_ms

    c.check(
        "fusion pays once the intermediate no longer fits in L2",
        lambda: big_speedup > 1.05,
        f"{big_speedup:.2f}x at 16384 rows",
    )
    c.check(
        "fusion buys little while the intermediate fits in L2",
        lambda: small_speedup < big_speedup,
        f"{small_speedup:.2f}x at 4096 rows against {big_speedup:.2f}x at 16384",
    )
    c.check(
        "the fused SwiGLU beats the unfused pair",
        lambda: swiglu_speedup > 1.1,
        f"{swiglu_speedup:.2f}x",
    )
    c.check(
        "rms_norm reaches at least 60% of what a plain copy achieves",
        lambda: achieved / COPY_CEILING_GBS > 0.6,
        f"{achieved:.0f} GB/s, {achieved / COPY_CEILING_GBS:.0%} of the copy "
        f"ceiling, {achieved / PEAK_BANDWIDTH_GBS:.0%} of the rating",
    )

    c.metric("fusion_speedup_past_l2", round(big_speedup, 2))
    c.metric("fusion_speedup_within_l2", round(small_speedup, 2))
    c.metric("achieved_gbs", round(achieved, 1))
    c.metric("vs_copy_ceiling", round(achieved / COPY_CEILING_GBS, 3))
    c.metric("vs_rated_peak", round(achieved / PEAK_BANDWIDTH_GBS, 3))
    c.metric("swiglu_speedup", round(swiglu_speedup, 2))
    return c.finish()
