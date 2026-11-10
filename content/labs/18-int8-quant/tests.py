import torch
from lab_common import Checks, pick_device

OUT, IN = 512, 1024
GROUP = 128


SMOOTH_TOKENS, SMOOTH_IN, SMOOTH_OUT = 128, 512, 256
OUTLIER_CHANNELS = (7, 150, 333)
OUTLIER_GAIN = 60.0


def _int8_per_tensor(tensor):
    scale = tensor.abs().max() / 127.0
    return (tensor / scale).round().clamp(-128, 127), scale


def _int8_per_row(tensor):
    scale = tensor.abs().amax(dim=-1, keepdim=True) / 127.0
    return (tensor / scale).round().clamp(-128, 127), scale


def _w8a8_error(x, weight):
    """Per-tensor int8 activations, per-output-row int8 weights, int32 sum."""
    x_q, x_scale = _int8_per_tensor(x)
    w_q, w_scale = _int8_per_row(weight)
    approx = (x_q @ w_q.T) * x_scale * w_scale.T
    exact = x @ weight.T
    return ((approx - exact).norm() / exact.norm()).item()


def _smooth_inputs():
    """Seed 18: 128 tokens, 512 inputs, 256 outputs, three outlier channels."""
    torch.manual_seed(18)
    x = torch.randn(SMOOTH_TOKENS, SMOOTH_IN)
    weight = torch.randn(SMOOTH_OUT, SMOOTH_IN) * 0.05
    clean = x.clone()
    x[:, list(OUTLIER_CHANNELS)] *= OUTLIER_GAIN
    return clean, x, weight


def _quantizer_checks(submission, c):
    needed = ("quantize_per_tensor", "quantize_per_channel", "quantize_per_group",
              "dequantize", "relative_error", "storage_bytes")
    if not c.require(submission, *needed):
        return

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


def _smooth_checks(submission, c):
    if not c.require(submission, "smooth"):
        return
    # SmoothQuant. W8A8 with one per-tensor activation scale collapses when a
    # few input channels are 60 times larger than the rest. Migrating the
    # difficulty into the weight by s = x_max^alpha / w_max^(1 - alpha) fixes it.
    # Plausible wrong implementations this section catches:
    #   - multiplying x by s and dividing the weight by s: the product is still
    #     preserved, but the outliers grow, so the alpha = 0.5 error is 0.2189
    #     instead of 0.0141;
    #   - swapping the exponents (x_max^(1 - alpha) / w_max^alpha): identical at
    #     alpha = 0.5, so the scale is also compared at alpha = 0.75;
    #   - dividing the weight by s instead of multiplying: the product breaks.
    clean_x, spiked_x, w_small = _smooth_inputs()
    err_clean = _w8a8_error(clean_x, w_small)
    err_spiked = _w8a8_error(spiked_x, w_small)
    c.check("planted outlier channels make per-tensor W8A8 error blow up",
            lambda: err_spiked > 5 * err_clean,
            f"expected spiked above 5 x clean; clean {err_clean:.6f}, spiked {err_spiked:.6f}")

    x_max = spiked_x.abs().amax(dim=0)
    w_max = w_small.abs().amax(dim=0)
    exact = spiked_x @ w_small.T

    def preserved():
        w_s, s = submission.smooth(w_small, x_max, w_max, 0.5)
        rel = ((((spiked_x / s) @ w_s.T) - exact).norm() / exact.norm()).item()
        return rel < 1e-5, (f"expected (x/s)(sW) == xW with relative error under 1e-5, "
                            f"got {rel:.3e}")
    c.check("smooth preserves the product (x / s)(s W) = x W", preserved)

    def scale_formula():
        _, s = submission.smooth(w_small, x_max, w_max, 0.75)
        expected = x_max.pow(0.75) / w_max.pow(0.25)
        rel = ((s.float() - expected).abs() / expected).max().item()
        return rel < 1e-5, (f"expected s = x_max^0.75 / w_max^0.25 at alpha 0.75, "
                            f"max relative gap {rel:.3e}, shape {tuple(s.shape)}")
    c.check("smooth returns s = x_max^alpha / w_max^(1 - alpha), one per input channel",
            scale_formula)

    def migration_helps():
        errors = {}
        for alpha in (0.0, 0.5):
            w_s, s = submission.smooth(w_small, x_max, w_max, alpha)
            errors[alpha] = _w8a8_error(spiked_x / s, w_s)
        lower = errors[0.5] < errors[0.0] and errors[0.5] < err_spiked
        close_enough = abs(errors[0.5] - 0.014136) < 0.05 * 0.014136
        return lower and close_enough, (
            f"expected alpha 0.5 error 0.014136 below alpha 0 error 0.088486 and "
            f"unsmoothed {err_spiked:.6f}; got alpha 0.5 {errors[0.5]:.6f}, "
            f"alpha 0 {errors[0.0]:.6f}")
    c.check("smoothing at alpha 0.5 lowers the W8A8 error on planted outliers",
            migration_helps)


def run(submission):
    c = Checks()
    for section in (_quantizer_checks, _smooth_checks):
        try:
            section(submission, c)
        except Exception as exc:  # noqa: BLE001
            c.add(f"{section.__name__} ran to the end", False,
                  f"{type(exc).__name__}: {exc}")
    return c.finish()
