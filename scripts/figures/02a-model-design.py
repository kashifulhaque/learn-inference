"""Numbers for chapter 2a.

Every figure is arithmetic on the fitted loss of Hoffmann et al. (digits as
reported by Besiroglu et al.) and on chapter 2's geometry, computed by
content/labs/02a-model-design/solution.py. Nothing here is a measurement, and
the loss values are in the fit's own units, not this model's loss.

Run from the repository root:

    python3 scripts/figures/02a-model-design.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "02a-model-design" / "solution.py"

CONFIG = {
    "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24,
    "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 17408,
    "vocab_size": 248320, "full_attention_interval": 4, "attn_output_gate": True,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "bytes_per_element": 2,
}
N0 = 26_895_979_520


def load():
    spec = importlib.util.spec_from_file_location("model_design_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frontier(m, c, n):
    """Return (D, loss) where an n-parameter model is compute-optimal under c."""
    a = c["beta"] / (c["alpha"] + c["beta"])
    g = (c["alpha"] * c["A"] / (c["beta"] * c["B"])) ** (1 / (c["alpha"] + c["beta"]))
    d = (n / g) ** (1 / a) / n
    return d, m.chinchilla_loss(n, d, c)


def main() -> None:
    m = load()
    print("Parameters, chapter 2:", f"{m.param_count(CONFIG):,}")

    print("\nCompute-optimal tokens per parameter at chapter 2's N, by fit")
    for name, c in (("Hoffmann", m.HOFFMANN), ("Besiroglu", m.BESIROGLU)):
        d, ell = frontier(m, c, N0)
        a = c["beta"] / (c["alpha"] + c["beta"])
        print(f"  {name:10s} exponent a={a:.4f}  D={d:.4e}  D/N={d / N0:.2f}  loss={ell:.6f}")
    for n in (1e9, 10e9):
        d, _ = frontier(m, m.HOFFMANN, n)
        print(f"  Hoffmann at N={n:.0e}: D/N={d / n:.2f}")

    for name, c in (("Hoffmann", m.HOFFMANN), ("Besiroglu", m.BESIROGLU)):
        d0, ell = frontier(m, c, N0)
        print(f"\n{name}: lifetime optimum at the loss of the compute-optimal 26.9B model ({ell:.6f})")
        print("  served tokens | N* (B) | D* (T) | D*/N* | FLOPs at N* | FLOPs at 26.9B | saved")
        for v in (0, 1e11, 1e12, 1e13, 1e14, 1e15):
            n, d = m.optimal_with_inference(ell, v, c)
            best = m.lifetime_flops(n, d, v)
            base = m.lifetime_flops(N0, d0, v)
            print(f"  {v:13.0e} | {n / 1e9:6.3f} | {d / 1e12:6.3f} | {d / n:7.1f} | {best:.3e} | {base:.3e} | {100 * (1 - best / base):5.1f}%")

    _, ell = frontier(m, m.HOFFMANN, N0)
    print("\nWeighting a served token (Hoffmann, 1e13 served tokens)")
    for w in (1.0, 5.0, 50.0):
        n, d = m.optimal_with_inference(ell, 1e13, m.HOFFMANN, w)
        print(f"  weight {w:4.0f}: N*={n / 1e9:.3f}B  D*={d / 1e12:.3f}T")

    shapes = {
        "this model (3:1, 4 KV heads)": CONFIG,
        "24 KV heads": {**CONFIG, "num_key_value_heads": 24},
        "8 KV heads": {**CONFIG, "num_key_value_heads": 8},
        "2 KV heads": {**CONFIG, "num_key_value_heads": 2},
        "1:1 hybrid": {**CONFIG, "full_attention_interval": 2},
        "15:1 hybrid": {**CONFIG, "full_attention_interval": 16},
        "all full attention, 4 KV": {**CONFIG, "full_attention_interval": 1},
        "all full attention, 24 KV": {**CONFIG, "num_key_value_heads": 24, "full_attention_interval": 1},
        "128k vocabulary": {**CONFIG, "vocab_size": 128000},
        "32 layers, wider": m.match_params(CONFIG, 32, N0),
        "96 layers, narrower": m.match_params(CONFIG, 96, N0),
    }
    for batch, ctx in ((1, 1024), (32, 4096), (32, 32768)):
        print(f"\nDecode bytes per token, batch {batch}, context {ctx:,}")
        for k in m.rank_shapes(shapes, batch, ctx):
            s = shapes[k]
            print(f"  {k:30s} params {m.param_count(s) / 1e9:7.3f}B  KV {m.kv_bytes_per_token(s) // 1024:5d} KiB/token  "
                  f"bytes {m.inference_cost_per_token(s, batch, ctx):16,.0f}  floor {m.decode_floor_ms(s, batch, ctx):8.3f} ms")
    for k in ("32 layers, wider", "96 layers, narrower"):
        s = shapes[k]
        print(f"  {k}: hidden {s['hidden_size']}, intermediate {s['intermediate_size']}, "
              f"{m.num_full_attention_layers(s)} full-attention layers")

    print("\nContext where one sequence's KV read equals its share of the weights")
    for b in (1, 8, 32, 128):
        print(f"  batch {b:3d}: {m.kv_equals_weights_context(CONFIG, b):,.0f} tokens")
    print("  all full attention, 4 KV heads, batch 32:",
          f"{m.kv_equals_weights_context({**CONFIG, 'full_attention_interval': 1}, 32):,.0f} tokens")

    w = m.param_count(CONFIG) * 2
    print(f"\nDecode floor at batch 1, context 1: {m.decode_floor_ms(CONFIG, 1, 1):.3f} ms; weights alone {1e3 * w / 1935e9:.3f} ms")
    for k, e in (("input embedding row gather ignored: ", 248320 * 5120 * 2),):
        print(f"{k}weight read would drop by {e:,} bytes ({100 * e / w:.2f}%)")


if __name__ == "__main__":
    main()
