"""Numbers for chapter 20a.

Every value comes from content/labs/20a-tree-speculation/solution.py. The
acceptance table comes from a Zipf target and a noisy draft, in closed form.
Nothing here is an A100 measurement or a measured acceptance rate.

Run from the repository root:

    python3 scripts/figures/20a-tree-speculation.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "content" / "labs" / "20a-tree-speculation" / "solution.py"


def load():
    spec = importlib.util.spec_from_file_location("tree_solution", SOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load()
    gib = 1024**3

    print("== Tree mask and positions, parents [-1, 0, 0, 1, 1, 2, 3], depth0 7")
    parents = [-1, 0, 0, 1, 1, 2, 3]
    print("depths", m.tree_depths(parents))
    print("positions", m.tree_positions(parents, 7).tolist())
    print("mask rows", [m.tree_mask(parents)[i].nonzero().flatten().tolist() for i in range(7)])
    print("leaf paths", m.leaf_paths(parents))
    print("mask entries set", int(m.tree_mask(parents).sum()), "of 49")

    print("\n== Branch acceptance, p = (0.5, 0.3, 0.2), q = (0.2, 0.3, 0.5)")
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.2, 0.3, 0.5])
    a = m.branch_acceptance(p, q, 3)
    print("a_k", [round(x, 4) for x in a], "sum", round(sum(a), 4), "left", round(1 - sum(a), 4))
    print("chain would accept", round(float(torch.minimum(p, q).sum()), 4))

    print("\n== Acceptance table: Zipf 0.9 target, 32 tokens, draft logit noise 0.8 to 1.8")
    g = torch.Generator().manual_seed(20)
    logp = -0.9 * torch.log(torch.arange(1, 33).float())
    pz = torch.softmax(logp, 0)
    table = []
    for sigma in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
        qz = torch.softmax(logp + torch.randn(32, generator=g) * sigma, 0)
        table.append(m.branch_acceptance(pz, qz, 4))
        print("sigma", sigma, "total variation", round(0.5 * float((pz - qz).abs().sum()), 3),
              "row", [round(x, 3) for x in table[-1]])

    print("\n== Best tree against a chain, same node budget (chain stops at depth 6)")
    for n in (1, 2, 4, 6, 8, 12, 16):
        tree = m.best_tree(table, n)
        chain = [-1] + list(range(min(n, 6)))
        et = m.expected_accepted_length(tree, table)
        ec = m.expected_accepted_length(chain, table)
        print(f"budget {n:2d}: tree {et:.3f} chain {ec:.3f} tokens/round {1 + et:.3f} "
              f"depth {max(m.tree_depths(tree))} parents {tree}")

    print("\n== Adaptive draft length, alpha 0.8, c 0.05")
    for alpha in (0.5, 0.7, 0.8, 0.9):
        print(f"alpha {alpha}: continuous optimum {m.optimal_gamma_continuous(alpha, 0.05):.3f}, "
              f"adaptive_gamma at batch 1 = {m.adaptive_gamma(1, alpha, 0.05)}, "
              f"speedup {m.chain_speedup(1, alpha, m.adaptive_gamma(1, alpha, 0.05), 0.05):.3f}")
    for b in (1, 8, 16, 32, 64, 96, 128, 160):
        g_ = m.adaptive_gamma(b, 0.8, 0.05)
        print(f"batch {b:3d}: gamma {g_:2d} speedup {m.chain_speedup(b, 0.8, g_, 0.05):.3f} "
              f"| gamma 4 speedup {m.chain_speedup(b, 0.8, 4, 0.05):.3f}")
    last_fixed = max(b for b in range(1, 2000) if m.chain_speedup(b, 0.8, 4, 0.05) >= 1.0)
    last_adapt = max(b for b in range(1, 2000) if m.adaptive_gamma(b, 0.8, 0.05) > 0)
    print("largest batch with speedup >= 1: fixed gamma 4 =", last_fixed, ", adaptive =", last_adapt)
    print("verify tokens per pass at batch 32, gamma 4:", 32 * 5, "; ridge", round(m.PEAK_FLOPS * m.WEIGHT_BYTES / (2 * m.ACTIVE_PARAMS * m.BANDWIDTH), 1))

    print("\n== Node budget under load, same table, c 0.05")
    for b in (1, 4, 8, 16, 32, 64, 128):
        free = m.adaptive_node_budget(b, table, 0.05)
        held = m.adaptive_node_budget(b, table, 0.05, 32, m.STATE_BYTES)
        print(f"batch {b:3d}: budget {free:2d} (speedup {m.tree_speedup(b, table, max(free, 1), 0.05):.3f}) "
              f"| with per-node state writes budget {held:2d}")
    for n in (4, 8, 16):
        print(f"batch 32, budget {n:2d}: speedup {m.tree_speedup(32, table, n, 0.05):.3f}, "
              f"with state writes {m.tree_speedup(32, table, n, 0.05, m.STATE_BYTES):.3f}")

    print("\n== Per-node recurrent state")
    s = m.STATE_BYTES
    print("one state", s, "bytes =", round(s / 2**20, 1), "MiB")
    for n in (5, 9, 17):
        print(f"{n:2d} nodes: {m.snapshot_bytes(n):,} bytes = {m.snapshot_bytes(n) / gib:.3f} GiB; "
              f"write time {m.snapshot_bytes(n) / m.BANDWIDTH * 1e3:.3f} ms per sequence")
    flat = sum(len(path) for path in m.leaf_paths(parents))
    print("7-node example: flattened path tokens", flat, "against", len(parents), "tree tokens")
    for n in (4, 8, 16):
        tree = m.best_tree(table, n)
        flat = sum(len(path) for path in m.leaf_paths(tree))
        print(f"best tree, {n:2d} nodes: {len(tree)} tokens as a tree, {flat} as flattened paths")
    print("write time per node", round(s / m.BANDWIDTH * 1e3, 4), "ms; ch 20 round trip", round(2 * s / m.BANDWIDTH * 1e3, 4), "ms")


if __name__ == "__main__":
    main()
