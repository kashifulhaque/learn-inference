import torch
from lab_common import Checks, close, pick_device

FULL = [3, 7, 11, 15]
LINEAR = [i for i in range(16) if i not in FULL]
KV_HEADS, HEAD_DIM = 2, 16
STATE_SHAPE = (4, 8, 8)
CONV_SHAPE = (40, 3)


def build(submission, device, batch=1, max_seq=64):
    return submission.Cache(
        FULL, LINEAR, batch, max_seq, KV_HEADS, HEAD_DIM,
        STATE_SHAPE, CONV_SHAPE, device=device, dtype=torch.float32,
    )


def run(submission):
    c = Checks()
    if not c.require(submission, "Cache"):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)
    cache = build(submission, device)

    c.check("a new cache has length zero", lambda: cache.length == 0)
    c.check(
        "only full-attention layers get KV buffers",
        lambda: sorted(cache.k_cache) == FULL,
        f"got {sorted(getattr(cache, 'k_cache', {}))}",
    )
    c.check(
        "only linear-attention layers get recurrent state",
        lambda: sorted(cache.states) == LINEAR,
    )

    # Prefill, then a decode step, and check the prefix is intact and in order.
    k1 = torch.randn(1, KV_HEADS, 10, HEAD_DIM, device=device)
    v1 = torch.randn(1, KV_HEADS, 10, HEAD_DIM, device=device)
    keys, values = cache.append(3, k1, v1)
    c.check("the first append returns the prefix so far",
            lambda: tuple(keys.shape) == (1, KV_HEADS, 10, HEAD_DIM),
            f"got {tuple(keys.shape)}")
    c.check("the appended keys are stored exactly", lambda: close(keys, k1, 0))
    cache.advance(10)

    k2 = torch.randn(1, KV_HEADS, 1, HEAD_DIM, device=device)
    v2 = torch.randn(1, KV_HEADS, 1, HEAD_DIM, device=device)
    keys, values = cache.append(3, k2, v2)
    c.check("a decode step extends the prefix by one",
            lambda: keys.shape[2] == 11, f"got {keys.shape[2]}")
    c.check("earlier entries are unchanged", lambda: close(keys[:, :, :10], k1, 0))
    c.check("the new entry lands at the end", lambda: close(keys[:, :, 10:], k2, 0))
    c.check("values track keys", lambda: close(values[:, :, :10], v1, 0))

    # Layers must not share storage.
    other = torch.randn(1, KV_HEADS, 11, HEAD_DIM, device=device)
    cache.append(7, other, other)
    c.check(
        "layers keep separate buffers",
        lambda: not torch.allclose(cache.k_cache[3][:, :, :11], other),
    )

    c.check(
        "overflowing the cache raises",
        lambda: _raises(
            lambda: build(submission, device, max_seq=4).append(
                3,
                torch.randn(1, KV_HEADS, 5, HEAD_DIM, device=device),
                torch.randn(1, KV_HEADS, 5, HEAD_DIM, device=device),
            )
        ),
    )

    fresh = build(submission, device)
    state = torch.randn(1, *STATE_SHAPE, device=device)
    conv = torch.randn(1, *CONV_SHAPE, device=device)
    fresh.set_linear_state(0, state, conv)
    c.check("the recurrent state round trips",
            lambda: close(fresh.recurrent_state(0), state, 0))
    c.check("the convolution window round trips",
            lambda: close(fresh.conv_state(0), conv, 0))
    c.check("a full-attention layer has no recurrent state",
            lambda: fresh.recurrent_state(3) is None)

    report = fresh.memory_bytes()
    expected_kv = 2 * len(FULL) * 1 * KV_HEADS * 64 * HEAD_DIM * 4
    c.check(
        "memory_bytes counts the KV buffers",
        lambda: report["kv"] == expected_kv,
        f"got {report['kv']:,}, expected {expected_kv:,}",
    )
    c.check("total is the sum of both parts",
            lambda: report["total"] == report["kv"] + report["recurrent"])

    # The end-to-end check: the engine's model, driven with this cache.
    error = _equivalence(submission, device, c)

    c.metric("equivalence_error", float(f"{error:.3e}"))
    c.metric("cache_bytes", report["total"])
    return c.finish()


def _raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False


def _equivalence(submission, device, c) -> float:
    from engine.config import ModelConfig
    from engine.model import HybridLanguageModel

    tiny = {
        "text_config": {
            "hidden_size": 64, "num_hidden_layers": 8, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128,
            "vocab_size": 256, "rms_norm_eps": 1e-6, "partial_rotary_factor": 0.5,
            "full_attention_interval": 4, "attn_output_gate": True,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "max_position_embeddings": 512,
            "dtype": "float32",
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        }
    }
    config = ModelConfig.from_dict(tiny)
    model = HybridLanguageModel(config).to(device).eval()
    ids = torch.randint(0, 256, (1, 24), device=device)

    conv_channels = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    cache = submission.Cache(
        config.full_attention_layers, config.linear_attention_layers, 1, 64,
        config.num_key_value_heads, config.head_dim,
        (config.linear_num_value_heads, config.linear_key_head_dim,
         config.linear_value_head_dim),
        (conv_channels, config.linear_conv_kernel_dim - 1),
        device=device, dtype=torch.float32,
    )

    with torch.no_grad():
        reference = model(ids)
        model(ids[:, :20], cache=cache, last_token_only=True)
        worst = 0.0
        for position in range(20, 24):
            logits = model(ids[:, position:position + 1], cache=cache,
                           last_token_only=True)
            worst = max(worst, (logits[:, -1] - reference[:, position]).abs().max().item())

    c.check(
        "the engine's model decodes correctly with your cache",
        lambda: (worst < 1e-4, f"max abs error {worst:.2e}"),
    )
    return worst
