import torch
from lab_common import Checks, close, require_cuda

PEAK_BANDWIDTH_GBS = 2039.0


def reference(x, weight, eps=1e-6):
    dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps) * weight.float()).to(dtype)


def run(submission):
    c = Checks()
    if not c.require(submission, "rms_norm", "rms_norm_low_precision", "bytes_moved"):
        return c.finish()

    device = require_cuda()
    torch.manual_seed(0)

    for shape in [(4, 5120), (1, 5120), (128, 5120), (7, 4097)]:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        w = torch.randn(shape[-1], device=device, dtype=torch.float32)
        c.check(
            f"matches the reference at {shape} in float32",
            lambda x=x, w=w: close(submission.rms_norm(x, w), reference(x, w), 1e-5),
        )

    x = torch.randn(64, 5120, device=device, dtype=torch.bfloat16)
    w = torch.ones(5120, device=device, dtype=torch.bfloat16)
    out = submission.rms_norm(x, w)
    c.check("the output keeps the input dtype", lambda: out.dtype == torch.bfloat16,
            f"got {out.dtype}")

    # A row of zeros must not produce NaN: that is what eps is for.
    zeros = torch.zeros(2, 5120, device=device, dtype=torch.float32)
    c.check(
        "an all-zero row stays finite",
        lambda: torch.isfinite(submission.rms_norm(zeros, w.float())).all().item(),
    )

    # How much the float32 accumulation is worth, on realistic activations.
    big = torch.randn(32, 5120, device=device, dtype=torch.bfloat16) * 8
    wb = torch.ones(5120, device=device, dtype=torch.bfloat16)
    exact = reference(big.float(), wb.float())
    err_fp32 = (submission.rms_norm(big, wb).float() - exact).abs().max().item()
    err_bf16 = (
        submission.rms_norm_low_precision(big, wb).float() - exact
    ).abs().max().item()

    c.check(
        "float32 accumulation beats bfloat16 accumulation",
        lambda: err_fp32 < err_bf16,
        f"fp32 {err_fp32:.3e} vs bf16 {err_bf16:.3e}",
    )

    n = 8192 * 5120
    x = torch.randn(8192, 5120, device=device, dtype=torch.bfloat16)
    w = torch.ones(5120, device=device, dtype=torch.bfloat16)
    expected_bytes = 2 * n * 2
    c.check(
        "bytes_moved counts one read and one write",
        lambda: submission.bytes_moved(x) == expected_bytes,
        f"got {submission.bytes_moved(x):,}, expected {expected_bytes:,}",
    )

    from engine.bench import benchmark

    timing = benchmark(lambda: submission.rms_norm(x, w), "rms_norm", warmup=5, runs=20)
    achieved = expected_bytes / (timing.median_ms / 1000) / 1e9
    c.metric("fp32_error", round(err_fp32, 6))
    c.metric("bf16_error", round(err_bf16, 6))
    c.metric("median_ms", timing.median_ms)
    c.metric("achieved_gbs", round(achieved, 1))
    c.metric("bandwidth_efficiency", round(achieved / PEAK_BANDWIDTH_GBS, 3))
    return c.finish()
