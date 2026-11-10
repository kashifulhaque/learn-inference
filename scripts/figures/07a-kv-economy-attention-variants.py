"""Numbers for chapter 07a.

Every number is config arithmetic from content/labs/07a-kv-economy/solution.py
or a seeded synthetic draw. DeepSeek-V2's dimensions are the paper's
(arXiv 2405.04434, section 3.1.2). The MLA rows for this course's model are
hypothetical, because the model has no MLA layers. Nothing here is an A100
measurement: the roofline times are floors from the rated 312 TFLOP/s and
1935 GB/s.

Run from the repository root:

    python3 scripts/figures/07a-kv-economy-attention-variants.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "07a-kv-economy" / "solution.py"

PEAK_FLOPS = 312e12
PEAK_BYTES = 1935e9
CONTEXT = 32768
WINDOW = 4096
SINKS = 4


def load():
    spec = importlib.util.spec_from_file_location("kv_economy_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = {"layers": 60, "n_heads": 128, "n_kv_heads": 128, "d_head": 128, "d_c": 512, "d_rope": 64}
THIS = {"layers": 16, "n_heads": 24, "n_kv_heads": 4, "d_head": 256, "d_c": 1024, "d_rope": 128}
THIS_MLA_V2_ABS = {**THIS, "d_c": 512, "d_rope": 64}


def kib(n):
    return f"{n / 1024:,.2f} KiB"


def main() -> None:
    m = load()
    print("== DeepSeek-V2 (paper dimensions)")
    mha_el = 2 * V2["n_heads"] * V2["d_head"] * V2["layers"]
    mla_el = (V2["d_c"] + V2["d_rope"]) * V2["layers"]
    print(f"MHA elements per token {mha_el:,}; MLA {mla_el:,}; ratio {mha_el / mla_el:.2f}")
    print(f"MHA bytes {m.gqa_kv_bytes(V2):,}; MLA bytes {m.mla_kv_bytes(V2):,} ({m.mla_kv_bytes(V2) / 1024:.2f} KiB)")
    print(f"GQA-equivalent groups: {mla_el / (2 * V2['d_head'] * V2['layers']):.2f}")
    print(f"per layer MLA elements {V2['d_c'] + V2['d_rope']} = {(V2['d_c'] + V2['d_rope']) / V2['d_head']} d_h")
    for t in (4096, 131072):
        a = m.mla_decode_cost(V2, t, "absorbed")
        n = m.mla_decode_cost(V2, t, "naive")
        g = m.gqa_decode_cost(V2, t)
        print(
            f"T={t}: absorbed {a['flops']:,} FLOPs, {a['bytes']:,} B, I={a['intensity']:.3f}; "
            f"naive {n['flops']:,} FLOPs, I={n['intensity']:,.1f}; MHA I={g['intensity']:.3f}; "
            f"naive/absorbed {n['flops'] / a['flops']:.1f}x"
        )
    t = 4096
    per_ctx_abs = 2 * 128 * (512 + 64) + 2 * 128 * 512
    print(f"absorbed FLOPs per context token per layer {per_ctx_abs:,}; fixed {4 * 128 * 128 * 512:,}")
    print(f"absorbed limit intensity {per_ctx_abs / (576 * 2):.3f} FLOPs/byte; ridge {PEAK_FLOPS / PEAK_BYTES:.1f}")
    # Roofline floor per context token per layer (bytes from the cache only).
    mha_t = max(4 * 128 * 128 / PEAK_FLOPS, 2 * 128 * 128 * 2 / PEAK_BYTES)
    mla_t = max(per_ctx_abs / PEAK_FLOPS, 576 * 2 / PEAK_BYTES)
    print(f"floor per context token per layer: MHA {mha_t * 1e9:.3f} ns, MLA absorbed {mla_t * 1e9:.3f} ns, ratio {mha_t / mla_t:.1f}x")
    print(f"MLA compute time {per_ctx_abs / PEAK_FLOPS * 1e9:.3f} ns vs memory time {576 * 2 / PEAK_BYTES * 1e9:.3f} ns")
    fixed_cross = (4 * 128 * 128 * 512) / per_ctx_abs
    print(f"fixed cost equals the context term at T={fixed_cross:.1f}")

    print("\n== This model (16 full-attention layers, 24 query heads, d_head 256)")
    rows = []

    def row(name, cache_bytes_32k, per_token, flops, tail):
        t = max(flops / PEAK_FLOPS, cache_bytes_32k / PEAK_BYTES)
        rows.append((name, per_token, cache_bytes_32k, flops, flops / cache_bytes_32k, t))

    mha = {**THIS, "n_kv_heads": 24}
    for name, cfg in (("MHA 24 KV heads", mha), ("GQA 4 KV heads (this model)", THIS)):
        c = m.gqa_decode_cost(cfg, CONTEXT)
        row(name, c["bytes"], m.gqa_kv_bytes(cfg), c["flops"], "")
    for name, cfg in (("MLA d_c 1024, d_rope 128 (V2 proportions)", THIS), ("MLA d_c 512, d_rope 64 (V2 absolute)", THIS_MLA_V2_ABS)):
        c = m.mla_decode_cost(cfg, CONTEXT, "absorbed")
        row(name, c["bytes"], m.mla_kv_bytes(cfg), c["flops"], "")
    wbytes = m.window_kv_bytes(THIS, CONTEXT, WINDOW, SINKS)
    wflops = m.gqa_decode_cost(THIS, WINDOW + SINKS)["flops"]
    row("Window 4096 plus 4 sinks, all layers", wbytes, m.gqa_kv_bytes(THIS), wflops, "")
    half_global = {**THIS, "layers": 8}
    g = m.gqa_decode_cost(half_global, CONTEXT)
    loc_bytes = m.window_kv_bytes(half_global, CONTEXT, WINDOW, 0)
    loc_flops = m.gqa_decode_cost(half_global, WINDOW)["flops"]
    row("Alternating local and global (8 and 8)", g["bytes"] + loc_bytes, m.gqa_kv_bytes(THIS), g["flops"] + loc_flops, "")
    row("H2O or SnapKV budget of 4096 tokens", m.window_kv_bytes(THIS, CONTEXT, 4096, 0), m.gqa_kv_bytes(THIS), m.gqa_decode_cost(THIS, 4096)["flops"], "")

    for name, per_token, b32, fl, inten, t in rows:
        print(
            f"{name}: per token {per_token:,} B ({per_token / 1024:g} KiB); at 32,768 {b32:,} B "
            f"({b32 / 2**20:,.3f} MiB); FLOPs {fl:,} ({fl / 1e9:.3f} G); I={inten:.3f}; floor {t * 1e6:.2f} us"
        )

    print("\n== Rolling buffer")
    print(f"slots {WINDOW + SINKS}; bytes {m.window_kv_bytes(THIS, CONTEXT, WINDOW, SINKS):,} = {m.window_kv_bytes(THIS, CONTEXT, WINDOW, SINKS) / 2**20:.3f} MiB; full {m.gqa_kv_bytes(THIS) * CONTEXT / 2**30:.0f} GiB")
    print(f"full / window = {CONTEXT / (WINDOW + SINKS):.3f}")
    print(f"Mistral-style reach with {THIS['layers']} layers x {WINDOW}: {THIS['layers'] * WINDOW:,}; Mistral 7B 32 x 4096 = {32 * 4096:,}")

    print("\n== Worked ring (window 4, sinks 2): slots for tokens 0..9")
    cache = m.RollingKVCache(4, 2, 1)
    slots = [cache.append(torch.zeros(1), torch.zeros(1)) for _ in range(10)]
    print(slots)
    _, _, pos = cache.gather()
    print("held after 10 tokens:", [int(x) for x in pos])

    print("\n== Planted sink (seed 1, 200 tokens, d 16, window 32)")
    n, d, window = 200, 16, 32
    g = torch.Generator().manual_seed(1)
    u = torch.randn(d, generator=g, dtype=torch.float64)
    u = u / u.norm()
    keys = torch.randn(n, d, generator=g, dtype=torch.float64)
    values = torch.randn(n, d, generator=g, dtype=torch.float64)
    keys[0] = 6 * math.sqrt(d) * u
    queries = 2 * u + 0.3 * torch.randn(n, d, generator=g, dtype=torch.float64)
    full = torch.softmax(keys @ queries[-1] / math.sqrt(d), dim=0)
    print(f"attention mass on token 0 under full attention: {full[0].item():.4f}")
    out_full = full @ values
    for sinks in (0, 1, 4):
        cache = m.RollingKVCache(window, sinks, d)
        for t in range(n):
            cache.append(keys[t], values[t])
        err = ((cache.attend(queries[-1]) - out_full).norm() / out_full.norm()).item()
        print(f"sinks {sinks}: relative error {err:.6f}")

    print("\n== H2O on planted heavy hitters (n 128, budget 24, recent 8)")
    for seed, heavy in ((0, (3, 17, 40, 41, 77, 90)), (5, (1, 9, 22, 58, 63, 100))):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(128, 128, generator=g, dtype=torch.float64) * 0.5
        for k in heavy:
            logits[:, k] += 4.0
        tt = torch.arange(128).unsqueeze(1)
        kk = torch.arange(128).unsqueeze(0)
        logits = (logits - 0.04 * (tt - kk).clamp(min=0)).masked_fill(kk > tt, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        oracle = torch.topk(attn.sum(0), 24).indices.tolist()
        kept = m.evict_h2o(logits, 24, 8)
        window_keep = list(range(128 - 24, 128))
        mass = lambda ks: attn[:, ks].sum().item()
        print(
            f"seed {seed}: heavy kept {sorted(set(heavy) & set(kept))}; oracle mass {mass(oracle):.3f}; "
            f"H2O {mass(kept):.3f} ({mass(kept) / mass(oracle):.4f}); window {mass(window_keep):.3f} "
            f"({mass(window_keep) / mass(oracle):.4f}); total mass {attn.sum().item():.0f}"
        )
        print("  kept:", kept)
        print("  oracle:", sorted(oracle))


if __name__ == "__main__":
    main()
