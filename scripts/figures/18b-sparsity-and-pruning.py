"""Numbers for the sparsity and pruning chapter.

Nothing here is an A100 measurement. The pruning errors are one seeded
synthetic layer from the lab. The byte counts and the decode floor are
arithmetic on 53.8 GB of weights at the measured 1275 GB/s copy rate, which
is an upper bound on what any sparse or int4 format can win.

Run from the repository root:

    python3 scripts/figures/18b-sparsity-and-pruning.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location("sol", ROOT / "content/labs/18b-sparsity/solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

WEIGHT_BYTES = 53.8e9  # bf16 weights, chapter 2
COPY_GBS = 1275e9  # measured copy bandwidth, chapter 0
DENSE_TFLOPS = 312e12  # rated bf16 tensor core, chapter 10


def layer():
    g = torch.Generator().manual_seed(18)
    weight = torch.randn(32, 64, generator=g, dtype=torch.float64)
    inputs = torch.randn(256, 64, generator=g, dtype=torch.float64)
    inputs[:, [3, 17, 40, 58]] *= 12.0
    return weight, inputs


def main() -> None:
    print("== Worked group of four (chapter: magnitude against Wanda) ==")
    w = torch.tensor([[0.9, -0.1, 0.4, -0.7]], dtype=torch.float64)
    x = torch.tensor([[1.0, 1.0, 3.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    _, m_mag = sol.prune_2_4(w, sol.magnitude_scores(w))
    _, m_wan = sol.prune_2_4(w, sol.wanda_scores(w, x))
    print("wanda scores", sol.wanda_scores(w, x).tolist())
    print("magnitude keeps", m_mag.tolist(), "wanda keeps", m_wan.tolist())
    out_ref = x @ w.T
    for name, mask in (("magnitude", m_mag), ("wanda", m_wan)):
        err = float((out_ref - x @ (w * mask).T).norm() / out_ref.norm())
        print(f"{name} relative error on that row {err:.6f}")

    print("\n== Seeded layer: 32 outputs, 64 inputs, 256 tokens, seed 18 ==")
    weight, inputs = layer()
    mag = sol.magnitude_scores(weight)
    wan = sol.wanda_scores(weight, inputs)
    norms = inputs.norm(dim=0)
    outl = [3, 17, 40, 58]
    others = [j for j in range(64) if j not in outl]
    print(f"outlier channel norm mean {norms[outl].mean():.4f}, other channels mean {norms[others].mean():.4f}")
    print(f"{'sparsity':>9} {'magnitude':>10} {'wanda':>10}")
    for sparsity in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        row = []
        for scores in (mag, wan):
            pruned, _ = sol.prune_rowwise(weight, scores, sparsity)
            row.append(sol.output_error(weight, pruned, inputs))
        print(f"{sparsity:>9.2f} {row[0]:>10.6f} {row[1]:>10.6f}")
    for name, scores in (("magnitude", mag), ("wanda", wan)):
        pruned, mask = sol.prune_2_4(weight, scores)
        print(f"2:4 {name}: error {sol.output_error(weight, pruned, inputs):.6f}, valid {sol.is_valid_2_4(mask)}")
    _, mask = sol.prune_rowwise(weight, wan, 0.5)
    print("50% rowwise Wanda mask is a valid 2:4 mask:", sol.is_valid_2_4(mask))
    # Share of kept weights that sit in outlier channels.
    pm, _ = sol.prune_rowwise(weight, mag, 0.5)
    pw, _ = sol.prune_rowwise(weight, wan, 0.5)
    print(
        "kept in outlier channels (of 128): magnitude",
        int((pm[:, outl] != 0).sum()),
        "wanda",
        int((pw[:, outl] != 0).sum()),
    )

    print("\n== Compressed format ==")
    pruned, mask = sol.prune_2_4(weight, wan)
    values, indices = sol.compress_2_4(pruned, mask)
    print("values", tuple(values.shape), "indices", tuple(indices.shape), "roundtrip", torch.equal(sol.decompress_2_4(values, indices, 64), pruned))
    vb, mb = sol.compressed_bytes(4096, 4096, 2)
    n = 4096 * 4096
    print(f"4096x4096 bf16: dense {2 * n} B, values {vb} B, metadata {mb} B, total {vb + mb} B")
    print(f"metadata share of the compressed matrix {mb / (vb + mb):.6f}")
    print(f"bytes per weight: dense bf16 2, 2:4 bf16 {(vb + mb) / n}, int4 g128 {0.5 + 2 / 128}")

    print("\n== Decode weight-read floor if every byte were a prunable linear weight ==")
    for name, per_weight in (("dense bf16", 2.0), ("2:4 bf16", 1.125), ("int4 g128", 0.5 + 2 / 128)):
        bytes_read = WEIGHT_BYTES * per_weight / 2.0
        ms = bytes_read / COPY_GBS * 1e3
        print(f"{name:>10}: {bytes_read / 1e9:7.3f} GB, {ms:6.3f} ms, {2.0 / per_weight:.3f}x fewer bytes than dense")
    print(f"sparse tensor core peak at twice the dense rate: {2 * DENSE_TFLOPS / 1e12:.0f} TFLOP/s")

    print("\n== Distillation loss ==")
    s = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    for temperature in (1.0, 2.0, 4.0):
        s.grad = None
        loss = sol.kd_loss(s, t, temperature)
        loss.backward()
        print(f"T={temperature:.0f}: loss {float(loss.detach()):.6f}, grad {s.grad.tolist()}")
    p_t = torch.softmax(t, -1)
    print("teacher softmax at T=1", p_t.tolist())
    pinsker = math.sqrt(0.5 * float(sol.kd_loss(s.detach(), t, 1.0)))
    tv = float(0.5 * (p_t - torch.softmax(s.detach(), -1)).abs().sum())
    print(f"TV {tv:.6f}, Pinsker bound sqrt(KL/2) {pinsker:.6f}, acceptance 1 - TV = {1 - tv:.6f}")


if __name__ == "__main__":
    main()
