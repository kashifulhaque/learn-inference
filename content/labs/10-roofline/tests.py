from lab_common import Checks, pick_device

HIDDEN, INTERMEDIATE = 5120, 17408
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256


def run(submission):
    c = Checks()
    needed = ("matmul_flops", "mlp_cost", "decode_attention_cost",
              "arithmetic_intensity", "ridge_point", "roofline")
    if not c.require(submission, *needed):
        return c.finish()

    c.check("a matmul counts two FLOPs per multiply-add",
            lambda: submission.matmul_flops(4, 8, 16) == 2 * 4 * 8 * 16,
            f"got {submission.matmul_flops(4, 8, 16)}")

    ridge = submission.ridge_point()
    c.check("the A100 80GB PCIe ridge point is about 161 FLOPs/byte",
            lambda: abs(ridge - 161.2) < 2.0, f"got {ridge:.1f}")

    flops1, bytes1 = submission.mlp_cost(1, HIDDEN, INTERMEDIATE)
    expected_flops = 3 * 2 * HIDDEN * INTERMEDIATE
    c.check("one token through the MLP does 3 matmuls' worth of FLOPs",
            lambda: flops1 == expected_flops,
            f"got {flops1:,}, expected {expected_flops:,}")
    c.check("the MLP reads 535 MB of weights",
            lambda: abs(bytes1 - 3 * HIDDEN * INTERMEDIATE * 2) < 1e6,
            f"got {bytes1 / 1e6:.1f} MB")

    intensity_1 = submission.arithmetic_intensity(*submission.mlp_cost(1, HIDDEN, INTERMEDIATE))
    intensity_2048 = submission.arithmetic_intensity(
        *submission.mlp_cost(2048, HIDDEN, INTERMEDIATE))

    c.check("decode at batch 1 sits near intensity 1",
            lambda: 0.5 < intensity_1 < 4, f"got {intensity_1:.2f}")
    c.check("prefill at 2048 tokens is far above the ridge point",
            lambda: intensity_2048 > 5 * ridge, f"got {intensity_2048:.0f}")

    r1 = submission.roofline(*submission.mlp_cost(1, HIDDEN, INTERMEDIATE))
    r2048 = submission.roofline(*submission.mlp_cost(2048, HIDDEN, INTERMEDIATE))
    c.check("the MLP at batch 1 is memory bound",
            lambda: r1["bound_by"] == "memory", f"got {r1['bound_by']}")
    c.check("the MLP at 2048 tokens is compute bound",
            lambda: r2048["bound_by"] == "compute", f"got {r2048['bound_by']}")
    c.check("the floor is the larger of the two limits",
            lambda: abs(r1["floor_ms"] - max(r1["compute_ms"], r1["memory_ms"])) < 1e-9)

    # Attention during decode: intensity is the GQA group size times two, and it
    # does not move with context length.
    short = submission.arithmetic_intensity(
        *submission.decode_attention_cost(1024, HEADS, KV_HEADS, HEAD_DIM))
    long = submission.arithmetic_intensity(
        *submission.decode_attention_cost(65536, HEADS, KV_HEADS, HEAD_DIM))
    c.check("decode attention intensity does not depend on context length",
            lambda: abs(short - long) < 1e-6, f"1k -> {short:.2f}, 64k -> {long:.2f}")
    c.check("decode attention intensity equals the GQA group size",
            lambda: abs(short - HEADS / KV_HEADS) < 0.1, f"got {short:.2f}")
    c.check("decode attention is memory bound",
            lambda: submission.roofline(
                *submission.decode_attention_cost(4096, HEADS, KV_HEADS, HEAD_DIM)
            )["bound_by"] == "memory")

    # A model with 24 KV heads would read six times the bytes for the same work.
    mqa = submission.arithmetic_intensity(
        *submission.decode_attention_cost(4096, HEADS, HEADS, HEAD_DIM))
    c.check("sharing KV heads raises intensity six-fold",
            lambda: abs(short / mqa - 6.0) < 0.1, f"ratio {short / mqa:.2f}")

    c.metric("ridge_point", round(ridge, 1))
    c.metric("decode_intensity", round(intensity_1, 2))
    c.metric("prefill_intensity", round(intensity_2048, 1))
    c.metric("decode_attention_intensity", round(short, 2))
    return c.finish()
