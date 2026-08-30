import torch
import torch.nn.functional as F
from lab_common import Checks, close, require_cuda

PEAK_BANDWIDTH_GBS = 2039.0
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

    gate = torch.randn(4096, 17408, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    c.check(
        "swiglu matches silu(gate) * up",
        lambda: close(submission.swiglu(gate, up).float(),
                      (F.silu(gate.float()) * up.float()), 5e-2),
    )

    from engine.bench import benchmark

    x = torch.randn(4096, HIDDEN, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    w = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)

    def unfused():
        total = x + residual
        return submission.rms_norm(total, w), total

    fused_t = benchmark(lambda: submission.rms_norm_residual(x, residual, w), "fused")
    unfused_t = benchmark(unfused, "unfused")
    fusion_speedup = unfused_t.median_ms / fused_t.median_ms

    norm_bytes = 2 * x.numel() * x.element_size()
    norm_t = benchmark(lambda: submission.rms_norm(x, w), "rms_norm")
    achieved = norm_bytes / (norm_t.median_ms / 1000) / 1e9

    def unfused_swiglu():
        return F.silu(gate) * up

    swiglu_t = benchmark(lambda: submission.swiglu(gate, up), "swiglu")
    swiglu_ref_t = benchmark(unfused_swiglu, "swiglu_torch")
    swiglu_speedup = swiglu_ref_t.median_ms / swiglu_t.median_ms

    c.check(
        "fusing the residual add is not slower than doing it separately",
        lambda: fusion_speedup > 0.95,
        f"{fusion_speedup:.2f}x",
    )
    c.check(
        "rms_norm reaches at least half of peak bandwidth",
        lambda: achieved / PEAK_BANDWIDTH_GBS > 0.5,
        f"{achieved:.0f} GB/s, {achieved / PEAK_BANDWIDTH_GBS:.0%} of peak",
    )

    c.metric("fusion_speedup", round(fusion_speedup, 2))
    c.metric("achieved_gbs", round(achieved, 1))
    c.metric("bandwidth_efficiency", round(achieved / PEAK_BANDWIDTH_GBS, 3))
    c.metric("swiglu_speedup", round(swiglu_speedup, 2))
    return c.finish()
