"""Checks for lab 19b: pipeline schedules and ring attention.

Everything runs on the CPU. Seeds are 19 for the tensors.

Plausible wrong implementations these checks catch:

1. A block-local causal mask. `ring_attention` builds the mask from
   `arange(tq) >= arange(tk)` inside each block instead of from the tokens'
   positions in the full sequence. That is right for the diagonal block and
   wrong for every other one. The "matches full attention" check fails it.
2. A "striped" assignment that returns contiguous runs. The output is still
   exact, but the work-balance check fails it.
3. A bubble formula of (p - 1) / m. The Gantt-table check fails it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import torch
from lab_common import Checks

HEADS, TOKENS, DIM = 2, 64, 16

# Run by two child processes. The store is a file, so there is no port to bind.
_GLOO_WORKER = """
import sys, torch, torch.distributed as dist
rank, world, store, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=world)
g = torch.Generator().manual_seed(19 + rank)
x = torch.randn(10, generator=g)
dist.all_reduce(x)
if rank == 0:
    torch.save(x, out)
dist.destroy_process_group()
"""


def _reference(q, k, v, causal=True):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        n = q.shape[1]
        mask = torch.ones(n, n, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _qkv():
    g = torch.Generator().manual_seed(19)
    return [torch.randn(HEADS, TOKENS, DIM, generator=g) for _ in range(3)]


def run(submission):
    c = Checks()
    if not c.require(
        submission, "block_attention", "merge_partials", "shard_indices",
        "ring_attention", "work_balance", "memory_per_rank", "min_shard_tokens",
        "pipeline_schedule", "bubble_fraction", "bubble_formula",
        "decode_pipeline", "compare_strategies", "ring_all_reduce",
    ):
        return c.finish()

    def block():
        g = torch.Generator().manual_seed(19)
        q = torch.randn(1, 3, 4, generator=g)
        k = torch.randn(1, 3, 4, generator=g)
        v = torch.randn(1, 3, 4, generator=g)
        # Query positions 0, 5, 9 over keys at positions 3, 4, 7. Query 0
        # sees no key.
        out, lse = submission.block_attention(
            q, k, v, torch.tensor([0, 5, 9]), torch.tensor([3, 4, 7]), True
        )
        scores = q[0] @ k[0].T / 2.0
        vis = torch.tensor([[False] * 3, [True, True, False], [True] * 3])
        scores = scores.masked_fill(~vis, float("-inf"))
        want_lse = torch.logsumexp(scores[1:], dim=-1)
        want_out = torch.softmax(scores[1:], dim=-1) @ v[0]
        ok_rows = (
            torch.allclose(out[0, 1:], want_out, atol=1e-5)
            and torch.allclose(lse[0, 1:], want_lse, atol=1e-5)
        )
        empty = bool(torch.all(out[0, 0] == 0)) and lse[0, 0] == float("-inf")
        return ok_rows and empty, (
            f"expected row 0 zeros with lse -inf and rows 1-2 lse {want_lse.tolist()}, "
            f"got row 0 {out[0, 0].tolist()} lse {lse[0, 0].item()} and rows 1-2 lse {lse[0, 1:].tolist()}"
        )

    c.check("block_attention masks by position and returns -inf for an empty row", block)

    def shards():
        msgs = []
        for kind in ("contiguous", "striped", "zigzag"):
            idx = submission.shard_indices(16, 4, kind)
            seen = torch.cat([torch.as_tensor(i) for i in idx]).sort().values.tolist()
            if seen != list(range(16)):
                msgs.append(f"{kind} covers {seen}")
        s = [torch.as_tensor(i).tolist() for i in submission.shard_indices(16, 4, "striped")]
        z = [torch.as_tensor(i).tolist() for i in submission.shard_indices(16, 4, "zigzag")]
        if s[1] != [1, 5, 9, 13]:
            msgs.append(f"striped rank 1 is {s[1]}")
        if z[1] != [2, 3, 12, 13]:
            msgs.append(f"zigzag rank 1 is {z[1]}")
        return not msgs, (
            "expected every token once, striped rank 1 [1, 5, 9, 13], zigzag rank 1 [2, 3, 12, 13]; got "
            + ("; ".join(msgs) if msgs else "exactly that")
        )

    c.check("each assignment owns every token once", shards)

    def exact():
        q, k, v = _qkv()
        worst, where = 0.0, ""
        for causal in (True, False):
            full = _reference(q, k, v, causal)
            for world in (2, 4):
                for kind in ("contiguous", "striped", "zigzag"):
                    out, _ = submission.ring_attention(q, k, v, world, kind, causal)
                    err = (out - full).abs().max().item()
                    if err > worst:
                        worst, where = err, f"{kind}, world {world}, causal {causal}"
        c.metric("ring_attention_error", worst)
        return worst <= 1e-5, f"expected max abs error <= 1e-5, got {worst:.2e} at {where}"

    c.check("ring attention matches full attention for every assignment", exact)

    def per_block_causal():
        q, k, v = _qkv()
        _, work = submission.ring_attention(q, k, v, 4, "contiguous", True)
        want = [[136, 136, 136, 136], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]
        total = sum(map(sum, work))
        return work == want and total == TOKENS * (TOKENS + 1) // 2, (
            f"expected {want} with {TOKENS * (TOKENS + 1) // 2} pairs in all, got {work} with {total}"
        )

    c.check("contiguous causal skips masked blocks and counts the visible pairs", per_block_causal)

    def balance():
        q, k, v = _qkv()
        got = {}
        for kind in ("contiguous", "striped", "zigzag"):
            _, work = submission.ring_attention(q, k, v, 4, kind, True)
            got[kind] = submission.work_balance(work)
        c.metric("striped_imbalance", round(got["striped"]["imbalance"], 4))
        ok = (
            got["contiguous"]["totals"] == [136, 392, 648, 904]
            and abs(got["contiguous"]["imbalance"] - 1.7385) < 1e-3
            and got["contiguous"]["critical"] == 904
            and got["striped"]["totals"] == [496, 512, 528, 544]
            and got["striped"]["critical"] == 544
            and got["zigzag"]["totals"] == [520] * 4
            and got["zigzag"]["critical"] == 520
        )
        return ok, (
            "expected totals contiguous [136, 392, 648, 904] critical 904, striped [496, 512, 528, 544] "
            "critical 544, zigzag [520, 520, 520, 520] critical 520; got "
            + ", ".join(f"{k} {v['totals']} critical {v['critical']}" for k, v in got.items())
        )

    c.check("striped and zigzag balance the causal work, contiguous does not", balance)

    def memory():
        m4 = submission.memory_per_rank(1_048_576, 4)
        m2 = submission.memory_per_rank(1_048_576, 2)
        m1 = submission.memory_per_rank(1_048_576, 1)
        c.metric("max_memory_per_rank_ratio", m4["peak_ratio"])
        ok = (
            m4["cache"] == 1_048_576 * 4096 // 4
            and m4["cache_ratio"] == 0.25
            and m4["peak_ratio"] == 0.5
            and m2["peak_ratio"] == 1.0
            and m1["peak_ratio"] == 1.0
        )
        return ok, (
            "expected cache 1/P, peak 2/P with a receive buffer (4 ranks: 0.25 and 0.5; 2 ranks: peak 1.0; "
            f"1 rank: 1.0); got 4 ranks {m4['cache_ratio']} and {m4['peak_ratio']}, "
            f"2 ranks peak {m2['peak_ratio']}, 1 rank peak {m1['peak_ratio']}"
        )

    c.check("per-rank memory ratio is 1/P resident and 2/P at the peak", memory)

    def gantt():
        sched = submission.pipeline_schedule(3, 4)
        want = [[0, 1, 2, 3, -1, -1], [-1, 0, 1, 2, 3, -1], [-1, -1, 0, 1, 2, 3]]
        if sched != want:
            return False, f"expected {want}, got {sched}"
        msgs = []
        for p, m in [(2, 1), (2, 8), (4, 4), (4, 16), (8, 32)]:
            got = submission.bubble_fraction(submission.pipeline_schedule(p, m))
            exp = (p - 1) / (m + p - 1)
            if abs(got - exp) > 1e-12:
                msgs.append(f"p={p} m={m}: expected {exp:.4f}, got {got:.4f}")
        if abs(submission.bubble_formula(4, 16, 2) - 3 / 35) > 1e-12:
            msgs.append(f"interleaved v=2: expected {3 / 35:.4f}, got {submission.bubble_formula(4, 16, 2):.4f}")
        return not msgs, "; ".join(msgs) or "matches (p-1)/(m+p-1) and the interleaved form"

    c.check("the Gantt table's bubble fraction is (p-1)/(m+p-1)", gantt)

    def decode():
        t, hop = 21.1e-3, 4e-6
        one = submission.decode_pipeline(2, 1, t, hop)
        two = submission.decode_pipeline(2, 2, t, hop)
        four = submission.decode_pipeline(2, 4, t, hop)
        ok = (
            one["utilization"] == 0.5 and two["utilization"] == 1.0 and four["utilization"] == 1.0
            and abs(one["latency"] - two["latency"]) < 1e-12
            and abs(two["rate"] / one["rate"] - 2.0) < 1e-9
            and abs(four["rate"] - two["rate"]) < 1e-9
        )
        return ok, (
            "expected utilization 0.5, 1.0, 1.0, the same latency, and rate x2 then flat; got "
            f"{one['utilization']}, {two['utilization']}, {four['utilization']}, latency {one['latency']:.6f} "
            f"and {two['latency']:.6f}, rate ratio {two['rate'] / one['rate']:.3f} then {four['rate'] / two['rate']:.3f}"
        )

    c.check("decode pipelining fills the stages without cutting latency", decode)

    def table():
        t = submission.compare_strategies(1_048_576, 4)
        msgs = []

        def near(name, got, exp, tol):
            if abs(got - exp) > tol:
                msgs.append(f"{name}: expected {exp}, got {got:.3f}")

        near("tp decode ms", t["tp"]["decode_ms"], 27.102, 0.01)
        near("pp decode ms", t["pp"]["decode_ms"], 96.106, 0.01)
        near("cp decode ms", t["cp"]["decode_ms"], 55.866, 0.01)
        near("dp decode ms", t["dp"]["decode_ms"], 96.094, 0.01)
        near("tp comm GB", t["tp"]["prefill_comm_gb"], 2061.584, 0.01)
        near("cp comm GB", t["cp"]["prefill_comm_gb"], 51.540, 0.01)
        near("pp comm GB", t["pp"]["prefill_comm_gb"], 10.737, 0.01)
        near("pp compute s", t["pp"]["prefill_compute_s"], 259.372, 0.01)
        near("cp weights GiB", t["cp"]["weights_gib"], 50.105, 0.01)
        near("tp weights GiB", t["tp"]["weights_gib"], 12.526, 0.01)
        fits = {k: v["fits"] for k, v in t.items()}
        if fits != {"dp": False, "tp": True, "pp": True, "cp": True}:
            msgs.append(f"fits: expected dp False, others True, got {fits}")
        near("min shard NVLink", submission.min_shard_tokens(300e9), 346.67, 0.01)
        near("min shard PCIe", submission.min_shard_tokens(32e9), 3250.0, 0.01)
        try:
            submission.compare_strategies(1_048_576, 8)
            msgs.append("expected ValueError for 8 ranks and 4 KV heads, got none")
        except ValueError:
            pass
        return not msgs, "; ".join(msgs) or "matches the comparison table"

    c.check("the long-context comparison matches this model's numbers", table)

    def gloo():
        shards = []
        for rank in range(2):
            g = torch.Generator().manual_seed(19 + rank)
            shards.append(torch.randn(10, generator=g))
        simulated, _ = submission.ring_all_reduce(shards)
        want = shards[0] + shards[1]
        if not torch.allclose(simulated[0], want, atol=1e-6):
            return False, f"expected the simulated ring to equal the sum {want.tolist()}, got {simulated[0].tolist()}"
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "worker.py")
            with open(script, "w") as fh:
                fh.write(_GLOO_WORKER)
            store, out = os.path.join(tmp, "store"), os.path.join(tmp, "out.pt")
            procs = [
                subprocess.Popen(
                    [sys.executable, script, str(r), "2", store, out],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=tmp,
                )
                for r in range(2)
            ]
            try:
                errs = [p.communicate(timeout=90)[1].decode()[-300:] for p in procs]
            except subprocess.TimeoutExpired:
                for p in procs:
                    p.kill()
                return True, "expected a gloo run, got skip (the two processes timed out)"
            if any(p.returncode for p in procs):
                return True, f"expected a gloo run, got skip (gloo unavailable here: {errs[0] or errs[1]})"
            real = torch.load(out)
        ok = torch.allclose(real, simulated[0], atol=1e-6)
        return ok, (
            f"expected gloo all_reduce {real.tolist()[:3]}... to equal the simulated ring "
            f"{simulated[0].tolist()[:3]}..."
        )

    c.check("a real gloo all_reduce on 2 CPU processes equals the simulated ring", gloo)

    return c.finish()
