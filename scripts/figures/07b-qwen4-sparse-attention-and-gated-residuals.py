"""Numbers for chapter 07b.

Every number is arithmetic from content/labs/07b-qwen4-blocks/solution.py
on one of two inputs:

* The Qwen3.8-2.4T-A95B Hugging Face card (hidden 8192, 92 layers, 23 full
  attention layers, 64 query heads, 4 KV heads, head width 256, 3 Gated
  DeltaNet layers per attention layer with 128 value heads, head width 128,
  512 experts with 10 routed and 1 shared active, expert width 2048).
* The Qwen4-Max blog (2.4T backbone, 71B active, 328B n-gram parameters).

Qwen published no block size, top-k, indexer width, gate equations, or n-gram
hashing scheme for Qwen4. The block size 64, top-k 64, and index width 64 below
are this chapter's illustrative choices. Nothing here is a measurement.

Run from the repository root:

    python3 scripts/figures/07b-qwen4-sparse-attention-and-gated-residuals.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sol", ROOT / "content/labs/07b-qwen4-blocks/solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

# Qwen3.8-2.4T-A95B card.
ATTN_LAYERS, LAYERS, HIDDEN = 23, 92, 8192
Q_HEADS, KV_HEADS, D_HEAD = 64, 4, 256
GDN_LAYERS, GDN_V_HEADS, GDN_DIM = 69, 128, 128
ROUTED, SHARED, D_EXPERT = 10, 1, 2048
# Illustrative design, not Qwen's.
BLOCK, TOP_K, D_IDX = 64, 64, 64
CONTEXTS = [32768, 131072, 262144, 1_000_000]
GIB = 1024**3


def cost(t, block=BLOCK, top_k=TOP_K):
    return sol.sparse_decode_cost(t, block, top_k, Q_HEADS, KV_HEADS, D_HEAD, D_IDX)


def main() -> None:
    per_token_layer = 2 * KV_HEADS * D_HEAD * 2
    per_token = per_token_layer * ATTN_LAYERS
    print("== KV cache, Qwen3.8 card ==")
    print(f"bytes per token per attention layer {per_token_layer:,}")
    print(f"bytes per token over {ATTN_LAYERS} layers {per_token:,} = {per_token / 1024:.1f} KiB")
    print(f"at 1,000,000 tokens {per_token * 1_000_000:,} B = {per_token * 1_000_000 / 1e9:.3f} GB = {per_token * 1_000_000 / GIB:.2f} GiB")
    print(f"course 27B model, 16 layers: {16 * per_token_layer:,} B per token")

    print("\n== GDN state, Qwen3.8 card (d_k x d_v per value head, no conv state) ==")
    elems = GDN_LAYERS * GDN_V_HEADS * GDN_DIM * GDN_DIM
    print(f"elements {elems:,}; bf16 {elems * 2 / 1e6:.1f} MB = {elems * 2 / GIB:.3f} GiB; fp32 {elems * 4 / 1e6:.1f} MB = {elems * 4 / GIB:.3f} GiB")

    print(f"\n== Sparse step per layer: block {BLOCK}, top-k {TOP_K}, d_idx {D_IDX}, bf16 ==")
    print(f"{'T':>10} {'blocks':>7} {'dense B':>15} {'indexer B':>10} {'kv B':>11} {'total B':>11} {'ratio':>8} {'dense GFLOP':>12} {'sparse MFLOP':>13}")
    for t in CONTEXTS:
        r = cost(t)
        mflop = (r["indexer_flops"] + r["attention_flops"]) / 1e6
        print(f"{t:>10,} {r['n_blocks']:>7,} {r['dense_bytes']:>15,} {r['indexer_bytes']:>10,} {r['kv_bytes']:>11,} {r['total_bytes']:>11,} {r['byte_ratio']:>8.1f} {r['dense_flops'] / 1e9:>12.3f} {mflop:>13.3f}")
    r = cost(1_000_000)
    print(f"all {ATTN_LAYERS} layers at 1M: dense {r['dense_bytes'] * ATTN_LAYERS / 1e9:.3f} GB, sparse {r['total_bytes'] * ATTN_LAYERS / 1e6:.3f} MB")
    print(f"extra cache for pooled index keys per layer at 1M: {r['extra_cache_bytes']:,} B = {r['extra_cache_bytes'] / per_token_layer / 1_000_000 * 100:.2f}% of that layer's KV")
    print(f"decode attention FLOPs, all layers, 1M dense {r['dense_flops'] * ATTN_LAYERS / 1e12:.3f} TFLOP; sparse {(r['indexer_flops'] + r['attention_flops']) * ATTN_LAYERS / 1e9:.3f} GFLOP")

    print("\n== Block-size sweep at 1,000,000 tokens, 4096 kept tokens ==")
    print(f"{'block':>6} {'top_k':>6} {'blocks':>7} {'indexer B':>10} {'kv B':>11} {'total B':>11} {'ratio':>8}")
    for block in (16, 32, 64, 128, 256, 512):
        r = cost(1_000_000, block, 4096 // block)
        print(f"{block:>6} {4096 // block:>6} {r['n_blocks']:>7,} {r['indexer_bytes']:>10,} {r['kv_bytes']:>11,} {r['total_bytes']:>11,} {r['byte_ratio']:>8.1f}")

    print("\n== Gated residual, hidden 8192, bf16 ==")
    g = sol.gated_residual_cost(4, HIDDEN, 2)
    print(g)
    moe = (ROUTED + SHARED) * 3 * HIDDEN * D_EXPERT * 2
    print(f"MoE layer FLOPs per token (11 experts x 3 matrices x 8192 x 2048 x 2) {moe:,}")
    print(f"gate FLOPs / MoE FLOPs = {g['gate_flops'] / moe:.3e}")

    print("\n== N-gram embedding, Qwen4-Max blog numbers ==")
    backbone, active, ngram = 2.4e12, 71e9, 328e9
    print(f"total with n-gram {backbone + ngram:,.0f}; n-gram share of backbone {ngram / backbone * 100:.2f}%")
    print(f"if bf16: n-gram {ngram * 2 / 1e9:.0f} GB; backbone {backbone * 2 / 1e12:.1f} TB")
    print(f"71B vs Qwen3.8's 95B: {(95 - 71) / 95 * 100:.2f}% fewer")
    row = HIDDEN * 2
    print(f"one table row, hidden 8192 bf16: {row:,} B; active weights bf16 {active * 2 / 1e9:.0f} GB; row/active {row / (active * 2):.2e}")
    a = sol.ngram_accounting(1 << 20, HIDDEN, 1, 2)
    print(f"table of 2**20 rows x 8192: params {a['params']:,}, bytes {a['stored_bytes']:,}, read {a['bytes_read_per_token']:,} B per token, flops {a['flops_per_token']:,}, dense-equivalent flops {a['dense_flops_per_token']:,}")

    print("\n== Decode step at batch 1, 1,000,000 tokens ==")
    weights = active * 2
    dense_kv = per_token * 1_000_000
    sparse_kv = cost(1_000_000)["total_bytes"] * ATTN_LAYERS
    print(f"active weights if bf16 (71B x 2 B): {weights / 1e9:.0f} GB")
    print(f"dense KV {dense_kv / 1e9:.3f} GB = {dense_kv / weights:.3f} of the weight read; sparse {sparse_kv / 1e6:.3f} MB = {sparse_kv / weights:.5f}")
    print(f"rated 1935 GB/s floors: dense {dense_kv / 1935e9 * 1e3:.2f} ms, sparse {sparse_kv / 1935e9 * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
