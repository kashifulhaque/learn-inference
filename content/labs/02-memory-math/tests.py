"""Checks lab 02 against the real Qwen3.8-27B geometry."""

from lab_common import Checks

CONFIG = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "full_attention_interval": 4,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "bytes_per_element": 2,
}

A100_80GB = 80 * 1024**3
WEIGHTS_BF16 = 53_800_000_000


def run(submission):
    c = Checks()
    needed = (
        "num_full_attention_layers", "mlp_params_per_layer", "embedding_params",
        "kv_bytes_per_token", "dense_kv_bytes_per_token", "recurrent_state_bytes",
        "breakeven_tokens", "max_batch_size",
    )
    if not c.require(submission, *needed):
        return c.finish()

    full = submission.num_full_attention_layers(CONFIG)
    c.check("16 layers use full attention", lambda: full == 16, f"got {full}")

    mlp = submission.mlp_params_per_layer(CONFIG)
    c.check(
        "one MLP block holds 267.4M parameters",
        lambda: mlp == 3 * 5120 * 17408,
        f"got {mlp:,}",
    )

    embed = submission.embedding_params(CONFIG)
    c.check(
        "embedding plus output projection is 2.54B parameters",
        lambda: embed == 2 * 248320 * 5120,
        f"got {embed:,}",
    )

    kv = submission.kv_bytes_per_token(CONFIG)
    c.check("KV cache grows 64 KiB per token", lambda: kv == 65536, f"got {kv:,} bytes")

    dense = submission.dense_kv_bytes_per_token(CONFIG)
    c.check(
        "an all-full-attention model would cost 256 KiB per token",
        lambda: dense == 262144,
        f"got {dense:,} bytes",
    )

    state = submission.recurrent_state_bytes(CONFIG)
    c.check(
        "the recurrent state is 147.8 MiB per sequence",
        lambda: abs(state - 154_927_104) < 1024,
        f"got {state / 1024**2:.1f} MiB",
    )

    breakeven = submission.breakeven_tokens(CONFIG)
    c.check(
        "the hybrid breaks even near 788 tokens",
        lambda: abs(breakeven - 788.0) < 2.0,
        f"got {breakeven:.1f}",
    )

    batch_32k = submission.max_batch_size(CONFIG, A100_80GB, WEIGHTS_BF16, 32768)
    batch_4k = submission.max_batch_size(CONFIG, A100_80GB, WEIGHTS_BF16, 4096)
    c.check(
        "batch at 32k context is between 5 and 12",
        lambda: 5 <= batch_32k <= 12,
        f"got {batch_32k}",
    )
    c.check(
        "a shorter context fits a larger batch",
        lambda: batch_4k > batch_32k,
        f"4k -> {batch_4k}, 32k -> {batch_32k}",
    )
    c.check(
        "an A100 40GB cannot hold the weights at all",
        lambda: submission.max_batch_size(
            CONFIG, 40 * 1024**3, WEIGHTS_BF16, 4096) == 0,
    )

    params = embed + 64 * mlp
    c.metric("params_billions", round(params / 1e9, 2))
    c.metric("kv_kib_per_token", kv // 1024)
    c.metric("recurrent_state_mib", round(state / 1024**2, 1))
    c.metric("breakeven_tokens", round(breakeven))
    c.metric("max_batch_32k", batch_32k)
    c.metric("max_batch_4k", batch_4k)
    return c.finish()
