import torch
import torch.nn.functional as F
from lab_common import Checks, close, pick_device

HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
GROUP = HEADS // KV_HEADS


def run(submission):
    c = Checks()
    needed = ("repeat_kv", "causal_mask", "attention", "apply_output_gate",
              "kv_bytes_per_token")
    if not c.require(submission, *needed):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)

    x = torch.randn(2, KV_HEADS, 5, HEAD_DIM, device=device)
    repeated = submission.repeat_kv(x, GROUP)
    c.check(
        "repeat_kv produces one head per query head",
        lambda: tuple(repeated.shape) == (2, HEADS, 5, HEAD_DIM),
        f"got {tuple(repeated.shape)}",
    )
    c.check(
        "query head h reads KV head h // group",
        lambda: all(
            torch.equal(repeated[:, h], x[:, h // GROUP]) for h in range(HEADS)
        ),
    )

    mask = submission.causal_mask(4, 4, device)
    c.check(
        "the prefill mask is upper triangular above the diagonal",
        lambda: torch.equal(
            mask, torch.triu(torch.ones(4, 4, dtype=torch.bool, device=device), 1)
        ),
    )
    decode_mask = submission.causal_mask(1, 10, device)
    c.check(
        "a single decode query sees the whole prefix",
        lambda: not decode_mask.any().item(),
        f"masked {int(decode_mask.sum())} of 10 positions",
    )
    chunk_mask = submission.causal_mask(3, 10, device)
    c.check(
        "a chunk behind a cached prefix masks only the future",
        lambda: chunk_mask.sum().item() == 3,
        f"masked {int(chunk_mask.sum())}, expected 3",
    )

    q = torch.randn(2, HEADS, 16, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(2, KV_HEADS, 16, HEAD_DIM, device=device, dtype=torch.float32)
    v = torch.randn(2, KV_HEADS, 16, HEAD_DIM, device=device, dtype=torch.float32)

    mine = submission.attention(q, k, v, causal=True)
    reference = F.scaled_dot_product_attention(
        q,
        submission.repeat_kv(k, GROUP),
        submission.repeat_kv(v, GROUP),
        is_causal=True,
        scale=HEAD_DIM**-0.5,
    )
    err = (mine - reference).abs().max().item()
    c.check(
        "causal attention matches scaled_dot_product_attention",
        lambda: (err < 1e-4, f"max abs error {err:.2e}"),
    )

    mine_nc = submission.attention(q, k, v, causal=False)
    ref_nc = F.scaled_dot_product_attention(
        q, submission.repeat_kv(k, GROUP), submission.repeat_kv(v, GROUP),
        is_causal=False, scale=HEAD_DIM**-0.5)
    c.check("non-causal attention matches too", lambda: close(mine_nc, ref_nc, 1e-4))

    # The decode path: one query against a cached prefix must equal the matching
    # row of a full prefill.
    full = submission.attention(q, k, v, causal=True)
    step = submission.attention(q[:, :, -1:], k, v, causal=True)
    c.check(
        "a decode step matches the last row of the prefill",
        lambda: close(step[:, :, 0], full[:, :, -1], 1e-4),
    )

    # Row sums: attention output is a convex combination, so with all-ones values
    # every output entry must be exactly 1.
    ones_v = torch.ones_like(v)
    out_ones = submission.attention(q, k, ones_v, causal=True)
    c.check(
        "attention weights sum to one",
        lambda: close(out_ones, torch.ones_like(out_ones), 1e-4),
    )

    attn_out = torch.randn(2, 4, HEADS * HEAD_DIM, device=device)
    gate = torch.randn(2, 4, HEADS * HEAD_DIM, device=device)
    gated = submission.apply_output_gate(attn_out, gate)
    c.check(
        "the output gate is a sigmoid applied elementwise",
        lambda: close(gated, attn_out * torch.sigmoid(gate), 1e-6),
    )
    c.check(
        "a strongly negative gate suppresses the output",
        lambda: submission.apply_output_gate(
            torch.ones(1, 1, 4, device=device),
            torch.full((1, 1, 4), -20.0, device=device),
        ).abs().max().item() < 1e-6,
    )

    per_token = submission.kv_bytes_per_token(KV_HEADS, HEAD_DIM, 16, 2)
    c.check(
        "the cache grows 64 KiB per token",
        lambda: per_token == 65536,
        f"got {per_token:,}",
    )
    c.check(
        "24 KV heads would cost six times as much",
        lambda: submission.kv_bytes_per_token(24, HEAD_DIM, 16, 2) == 6 * 65536,
    )

    c.metric("sdpa_error", float(f"{err:.3e}"))
    c.metric("cache_bytes_per_token", per_token)
    return c.finish()
