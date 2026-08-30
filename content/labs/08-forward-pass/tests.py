"""Checks lab 08 against the engine's own hybrid model at a small size."""

import torch
from lab_common import Checks, pick_device

TINY = {
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


def run(submission):
    c = Checks()
    if not c.require(submission, "logit_metrics", "first_divergent_layer",
                     "prefill_decode_equivalence"):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)

    # Identical inputs: every metric must report perfect agreement.
    logits = torch.randn(2, 10, 512, device=device)
    same = submission.logit_metrics(logits, logits)
    c.check("identical logits correlate at 1.0",
            lambda: abs(same["correlation"] - 1.0) < 1e-5,
            f"got {same['correlation']:.6f}")
    c.check("identical logits agree on every argmax",
            lambda: abs(same["top1_agreement"] - 1.0) < 1e-9)
    c.check("identical logits have zero KL",
            lambda: abs(same["mean_kl"]) < 1e-6, f"got {same['mean_kl']:.2e}")

    # A small perturbation: still highly correlated, KL small but positive.
    noisy = logits + torch.randn_like(logits) * 0.01
    near = submission.logit_metrics(noisy, logits)
    c.check("a small perturbation stays above 0.999 correlation",
            lambda: near["correlation"] > 0.999, f"got {near['correlation']:.6f}")
    c.check("a small perturbation has positive KL",
            lambda: 0 < near["mean_kl"] < 0.01, f"got {near['mean_kl']:.2e}")

    # Unrelated logits: correlation near zero, agreement near chance.
    other = torch.randn_like(logits)
    far = submission.logit_metrics(other, logits)
    c.check("unrelated logits correlate near zero",
            lambda: abs(far["correlation"]) < 0.1, f"got {far['correlation']:.4f}")
    c.check("unrelated logits rarely agree on the argmax",
            lambda: far["top1_agreement"] < 0.1, f"got {far['top1_agreement']:.3f}")

    # Bisection: plant a divergence at a known layer.
    reference_states = [torch.randn(2, 6, 64, device=device) for _ in range(8)]
    mine_states = [s.clone() for s in reference_states]
    for i in range(5, 8):
        mine_states[i] = mine_states[i] + 0.5
    c.check(
        "the first divergent layer is found",
        lambda: submission.first_divergent_layer(mine_states, reference_states) == 5,
        f"got {submission.first_divergent_layer(mine_states, reference_states)}",
    )
    c.check(
        "identical states report no divergence",
        lambda: submission.first_divergent_layer(
            reference_states, reference_states) is None,
    )

    # The real test: prefill plus decode must equal a full forward pass.
    from engine.cache import HybridCache
    from engine.config import ModelConfig
    from engine.model import HybridLanguageModel

    config = ModelConfig.from_dict(TINY)
    model = HybridLanguageModel(config).to(device).eval()
    ids = torch.randint(0, 256, (1, 37), device=device)

    error = submission.prefill_decode_equivalence(
        model,
        ids,
        30,
        lambda: HybridCache(config, 1, 128, device=device, dtype=torch.float32),
    )
    c.check(
        "prefill then single steps matches one full forward pass",
        lambda: (error < 1e-4, f"max abs error {error:.2e}"),
    )

    c.metric("correlation", round(near["correlation"], 6))
    c.metric("top1_agreement", round(near["top1_agreement"], 4))
    c.metric("prefill_decode_error", float(f"{error:.3e}"))
    return c.finish()
