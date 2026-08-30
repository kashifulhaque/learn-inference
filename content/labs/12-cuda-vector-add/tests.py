import os
import time

import torch
from lab_common import Checks, close, require_cuda

# An A100 80GB PCIe is rated at 1935 GB/s. A plain copy reaches about
# 1275, which is the ceiling a real kernel is measured against.
PEAK_BANDWIDTH_GBS = 1935.0
COPY_CEILING_GBS = 1275.0


def run(submission):
    c = Checks()
    if not c.require(submission, "cuda_source", "CPP_DECLARATIONS"):
        return c.finish()

    device = require_cuda()
    # Surface CUDA errors at the launch that caused them, not several calls later.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.manual_seed(0)

    from torch.utils.cpp_extension import load_inline

    print("Compiling. The first run takes 30 to 60 seconds.", flush=True)
    started = time.perf_counter()
    try:
        module = load_inline(
            name="lab12_kernels",
            cpp_sources=submission.CPP_DECLARATIONS,
            cuda_sources=submission.cuda_source(),
            functions=["vector_add", "strided_copy", "rms_norm"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        c.add("the CUDA source compiles", False, str(exc)[:2000])
        return c.finish()
    c.add("the CUDA source compiles", True,
          f"{time.perf_counter() - started:.1f}s")

    # Sizes that are not multiples of the block size exercise the tail guard.
    for n in (1024, 1_000_003, 1):
        a = torch.randn(n, device=device)
        b = torch.randn(n, device=device)
        out = module.vector_add(a, b)
        torch.cuda.synchronize()
        c.check(f"vector_add is correct at n={n:,}", lambda: close(out, a + b, 1e-5))

    n = 1 << 24
    a = torch.randn(n, device=device)
    strided = module.strided_copy(a, 32)
    torch.cuda.synchronize()
    index = (torch.arange(n, device=device, dtype=torch.long) * 32) % n
    c.check("strided_copy reads the expected elements",
            lambda: close(strided, a[index], 1e-6))

    rows, cols = 4096, 1024
    x = torch.randn(rows, cols, device=device)
    w = torch.randn(cols, device=device)
    mine = module.rms_norm(x, w, 1e-6)
    torch.cuda.synchronize()
    reference = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    c.check("rms_norm matches the PyTorch reference", lambda: close(mine, reference, 1e-4))

    big = torch.randn(rows, cols, device=device) * 100
    ref_big = big * torch.rsqrt(big.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    c.check("rms_norm stays accurate on large activations",
            lambda: close(module.rms_norm(big, w, 1e-6), ref_big, 1e-3))

    from engine.bench import benchmark

    b_vec = torch.randn(n, device=device)
    add_t = benchmark(lambda: module.vector_add(a, b_vec), "add")
    strided_t = benchmark(lambda: module.strided_copy(a, 32), "strided")

    add_bytes = 3 * n * 4          # two reads and one write
    copy_bytes = 2 * n * 4         # one read and one write
    coalesced_gbs = add_bytes / (add_t.median_ms / 1000) / 1e9
    strided_gbs = copy_bytes / (strided_t.median_ms / 1000) / 1e9
    ratio = coalesced_gbs / max(strided_gbs, 1e-9)

    c.check(
        "coalesced access is much faster than strided",
        lambda: ratio > 3.0,
        f"{ratio:.1f}x ({coalesced_gbs:.0f} vs {strided_gbs:.0f} GB/s)",
    )
    c.check(
        "the coalesced kernel reaches most of what a plain copy achieves",
        lambda: coalesced_gbs / COPY_CEILING_GBS > 0.6,
        f"{coalesced_gbs:.0f} GB/s, {coalesced_gbs / COPY_CEILING_GBS:.0%} of the "
        f"copy ceiling",
    )

    c.metric("coalesced_gbs", round(coalesced_gbs, 1))
    c.metric("strided_gbs", round(strided_gbs, 1))
    c.metric("coalescing_ratio", round(ratio, 1))
    c.metric("vs_copy_ceiling", round(coalesced_gbs / COPY_CEILING_GBS, 3))
    return c.finish()
