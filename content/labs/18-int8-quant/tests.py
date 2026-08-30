import torch
from lab_common import Checks, pick_device

OUT, IN = 512, 1024
GROUP = 128


def run(submission):
    c = Checks()
    needed = ("quantize_per_tensor", "quantize_per_channel", "quantize_per_group",
              "dequantize", "relative_error", "storage_bytes")
    if not c.require(submission, *needed):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)
    weight = torch.randn(OUT, IN, device=device)

    q_t, s_t = submission.quantize_per_tensor(weight)
    c.check("per-tensor quantization returns int8", lambda: q_t.dtype == torch.int8,
            f"got {q_t.dtype}")
    c.check("values stay inside the int8 range",
            lambda: q_t.min().item() >= -128 and q_t.max().item() <= 127)
    c.check("the largest weight maps near 127",
            lambda: q_t.abs().max().item() >= 126,
            f"got {q_t.abs().max().item()}")

    q_c, s_c = submission.quantize_per_channel(weight)
    c.check("per-channel scales have one entry per row",
            lambda: tuple(s_c.shape) == (OUT, 1), f"got {tuple(s_c.shape)}")

    q_g, s_g = submission.quantize_per_group(weight, GROUP)
    c.check("per-group scales have one entry per group",
            lambda: tuple(s_g.shape) == (OUT, IN // GROUP), f"got {tuple(s_g.shape)}")
    c.check("the quantized tensor keeps its original shape",
            lambda: tuple(q_g.shape) == (OUT, IN), f"got {tuple(q_g.shape)}")

    err_t = submission.relative_error(
        weight, submission.dequantize(q_t, s_t))
    err_c = submission.relative_error(
        weight, submission.dequantize(q_c, s_c))
    err_g = submission.relative_error(
        weight, submission.dequantize(q_g, s_g, GROUP))

    c.check("finer granularity gives lower error",
            lambda: err_g < err_c < err_t,
            f"tensor {err_t:.4f} > channel {err_c:.4f} > group {err_g:.4f}")
    c.check("group-wise error stays under 1%", lambda: err_g < 0.01,
            f"{err_g:.4%}")
    c.check("relative error of an exact reconstruction is zero",
            lambda: submission.relative_error(weight, weight) < 1e-9)

    # An outlier is what separates the granularities. One huge weight ruins a
    # per-tensor scale and barely touches its own group.
    spiked = torch.randn(OUT, IN, device=device)
    spiked[0, 0] = 500.0
    q_ts, s_ts = submission.quantize_per_tensor(spiked)
    q_gs, s_gs = submission.quantize_per_group(spiked, GROUP)
    err_ts = submission.relative_error(spiked, submission.dequantize(q_ts, s_ts))
    err_gs = submission.relative_error(
        spiked, submission.dequantize(q_gs, s_gs, GROUP))
    c.check("an outlier hurts a per-tensor scale far more than a per-group one",
            lambda: err_ts > 5 * err_gs,
            f"tensor {err_ts:.4f} vs group {err_gs:.4f}")

    # Reconstruction must land near the original, not merely be the right shape.
    reconstructed = submission.dequantize(q_g, s_g, GROUP)
    c.check("dequantize returns the original shape",
            lambda: tuple(reconstructed.shape) == (OUT, IN))
    c.check("dequantized weights track the originals",
            lambda: (reconstructed - weight).abs().max().item()
            < weight.abs().max().item() * 0.05,
            f"max abs error {(reconstructed - weight).abs().max().item():.4f}")

    bf16_bytes = OUT * IN * 2
    group_bytes = submission.storage_bytes((OUT, IN), GROUP)
    channel_bytes = submission.storage_bytes((OUT, IN))
    c.check("per-group storage is int8 plus one scale per group",
            lambda: group_bytes == OUT * IN + OUT * (IN // GROUP) * 2,
            f"got {group_bytes:,}")
    c.check("per-group storage is under 55% of bfloat16",
            lambda: group_bytes / bf16_bytes < 0.55,
            f"{group_bytes / bf16_bytes:.1%}")
    c.check("per-channel storage is smaller still",
            lambda: channel_bytes < group_bytes)

    # The output error is what actually matters, so check a real matmul.
    x = torch.randn(64, IN, device=device)
    exact = x @ weight.T
    approx = x @ submission.dequantize(q_g, s_g, GROUP).T
    output_error = ((approx - exact).norm() / exact.norm()).item()
    c.check("the layer output error stays under 1%", lambda: output_error < 0.01,
            f"{output_error:.4%}")

    c.metric("error_per_tensor", round(err_t, 5))
    c.metric("error_per_channel", round(err_c, 5))
    c.metric("error_per_group", round(err_g, 5))
    c.metric("output_error", round(output_error, 5))
    c.metric("compression", round(bf16_bytes / group_bytes, 3))
    return c.finish()
