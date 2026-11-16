"""Checks for the three Qwen4-style blocks.

Plausible wrong implementations, each run against these checks:

* `select_blocks` that scores single tokens and keeps the best tokens. It
  returns token ids, not block ids, so the block-id and masked-dense checks
  fail.
* `sparse_attention_step` that softmaxes over all T tokens and zeroes the
  dropped ones afterwards. The weights no longer sum to 1 over the kept
  tokens, so it disagrees with masked dense attention.
* `gated_residual` that applies `write` to the stream before adding `f(h)`
  or forgets the skip term `X`. The identity check fails.
* `ngram_hash` that sums the ids. It is deterministic and in range, but
  (a, b) and (b, a) collide, so the order and uniformity checks fail.
* `ngram_indices` that hashes the whole prefix. Index t then changes when an
  id far before t changes, so the window check fails.
"""

from __future__ import annotations

import math

import torch

from lab_common import Checks


def _call(submission, name, *args, **kwargs):
    try:
        return getattr(submission, name)(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _dense_masked(q, K, V, keep):
    """Reference: dense attention with every token outside `keep` masked out."""
    scores = (q @ K.T) / math.sqrt(q.shape[-1])
    mask = torch.full((K.shape[0],), float("-inf"), dtype=q.dtype)
    mask[keep] = 0.0
    return torch.softmax(scores + mask, dim=-1) @ V


def _cfg():
    """The cost example: 65,536 tokens, 64-token blocks, 16 blocks kept."""
    return dict(
        n_tokens=65536, block=64, top_k=16, n_heads=64, n_kv_heads=4,
        d_head=256, d_idx=64,
    )


def run(submission) -> dict:
    c = Checks()
    if not c.require(
        submission,
        "pool_blocks", "select_blocks", "block_token_indices", "sparse_attention_step",
        "sparse_decode_cost", "gated_residual", "dynamic_gates", "gated_residual_cost",
        "ngram_hash", "ngram_indices", "ngram_embedding", "inject_ngram", "ngram_accounting",
    ):
        return c.finish()
    f64 = torch.float64

    # ---- Part 1: block-sparse attention --------------------------------

    def pooling():
        g = torch.Generator().manual_seed(1)
        k = torch.randn(10, 3, generator=g, dtype=f64)
        got, err = _call(submission, "pool_blocks", k, 4)
        if err:
            return False, f"expected shape (3, 3), got {err}"
        want = torch.stack([k[0:4].mean(0), k[4:8].mean(0), k[8:10].mean(0)])
        if tuple(got.shape) != (3, 3):
            return False, f"expected shape (3, 3), got {tuple(got.shape)}"
        worst = float((got - want).abs().max())
        return worst < 1e-12, f"expected the mean of each block, max error {worst:.1e} (last block has 2 rows)"

    c.check("pool_blocks averages each micro-block, including a short last block", pooling)

    def selection():
        g = torch.Generator().manual_seed(2)
        k = torch.randn(64, 8, generator=g, dtype=f64) * 0.1
        q = torch.randn(8, generator=g, dtype=f64)
        # Blocks of 8 tokens. Plant blocks 5 and 2 with keys along q.
        k[40:48] += q
        k[16:24] += 0.5 * q
        got, err = _call(submission, "select_blocks", q, k, 8, 2)
        if err:
            return False, f"expected block ids [2, 5], got {err}"
        ids = [int(i) for i in got]
        return ids == [2, 5], f"expected block ids [2, 5], got {ids}"

    c.check("select_blocks returns the top-k block ids, ascending", selection)

    def ties_and_all():
        k = torch.ones(24, 2, dtype=f64)
        q = torch.ones(2, dtype=f64)
        got, err = _call(submission, "select_blocks", q, k, 4, 3)
        if err:
            return False, f"expected [0, 1, 2] on a tie, got {err}"
        if [int(i) for i in got] != [0, 1, 2]:
            return False, f"expected [0, 1, 2] on a tie, got {[int(i) for i in got]}"
        got, err = _call(submission, "select_blocks", q, k, 4, 99)
        if err:
            return False, f"expected all 6 blocks, got {err}"
        return [int(i) for i in got] == list(range(6)), f"expected all 6 blocks, got {[int(i) for i in got]}"

    c.check("select_blocks breaks ties toward the lower block and returns all blocks when k is large", ties_and_all)

    def matches_masked_dense():
        g = torch.Generator().manual_seed(3)
        n_tokens, block, d = 203, 16, 12  # 13 blocks, the last one has 11 tokens
        q = torch.randn(5, d, generator=g, dtype=f64)
        K = torch.randn(n_tokens, d, generator=g, dtype=f64)
        V = torch.randn(n_tokens, d, generator=g, dtype=f64)
        for ids in ([0, 3, 12], [12], [1, 2, 3, 4]):
            got, err = _call(submission, "sparse_attention_step", q, K, V, torch.tensor(ids), block)
            if err:
                return False, f"expected masked dense attention for blocks {ids}, got {err}"
            keep = torch.cat([torch.arange(b * block, min((b + 1) * block, n_tokens)) for b in ids])
            want = _dense_masked(q, K, V, keep)
            worst = float((got - want).abs().max())
            if worst > 1e-9:
                return False, f"expected masked dense attention for blocks {ids}, max error {worst:.2e}"
        return True, "matches dense attention restricted to the same blocks, including the short last block"

    c.check("sparse_attention_step equals dense attention restricted to the same blocks", matches_masked_dense)

    def all_blocks_is_dense():
        g = torch.Generator().manual_seed(4)
        q = torch.randn(4, 8, generator=g, dtype=f64)
        K = torch.randn(40, 8, generator=g, dtype=f64)
        V = torch.randn(40, 8, generator=g, dtype=f64)
        got, err = _call(submission, "sparse_attention_step", q, K, V, torch.arange(5), 8)
        if err:
            return False, f"expected full dense attention, got {err}"
        want = _dense_masked(q, K, V, torch.arange(40))
        worst = float((got - want).abs().max())
        return worst < 1e-9, f"expected full dense attention with every block kept, max error {worst:.2e}"

    c.check("keeping every block gives full dense attention", all_blocks_is_dense)

    def needle():
        g = torch.Generator().manual_seed(5)
        n_tokens, block, d = 512, 16, 32
        q = torch.randn(1, d, generator=g, dtype=f64)
        K = torch.randn(n_tokens, d, generator=g, dtype=f64) * 0.3
        V = torch.randn(n_tokens, d, generator=g, dtype=f64)
        K[200:208] += 2.0 * q[0]  # a needle inside block 12
        # Here the indexer sees the keys themselves.
        ids, err = _call(submission, "select_blocks", q[0], K, block, 4)
        if err:
            return False, f"expected the needle block among the 4 kept, got {err}"
        if 12 not in [int(i) for i in ids]:
            return False, f"expected block 12 among the kept blocks, got {[int(i) for i in ids]}"
        sparse = submission.sparse_attention_step(q, K, V, ids, block)
        dense = _dense_masked(q, K, V, torch.arange(n_tokens))
        rel = float((sparse - dense).norm() / dense.norm())
        return rel < 0.05, f"expected relative error under 0.05 against full attention, got {rel:.4f}"

    c.check("top-4 of 32 blocks stays close to full attention when one block holds the mass", needle)

    def cost_numbers():
        got, err = _call(submission, "sparse_decode_cost", **_cfg())
        if err:
            return False, f"expected the cost dict, got {err}"
        want = {
            "n_blocks": 1024,
            "kept_tokens": 1024,
            "indexer_bytes": 131072,
            "kv_bytes": 4194304,
            "total_bytes": 4325376,
            "dense_bytes": 268435456,
            "indexer_flops": 131072,
            "attention_flops": 67108864,
            "dense_flops": 4294967296,
        }
        for key, value in want.items():
            if got.get(key) != value:
                return False, f"expected {key} = {value:,}, got {got.get(key)}"
        ratio = 268435456 / 4325376
        if abs(got.get("byte_ratio", 0) - ratio) > 1e-9:
            return False, f"expected byte_ratio {ratio:.3f}, got {got.get('byte_ratio')}"
        c.metric("sparse_byte_ratio", round(got["byte_ratio"], 3))
        return True, f"dense {want['dense_bytes']:,} B against {want['total_bytes']:,} B, a ratio of {ratio:.2f}"

    c.check("sparse_decode_cost gives the bytes and FLOPs for 65,536 tokens", cost_numbers)

    def cost_scaling():
        base = submission.sparse_decode_cost(**_cfg())
        double_t = submission.sparse_decode_cost(**{**_cfg(), "n_tokens": 131072})
        half_block = submission.sparse_decode_cost(**{**_cfg(), "block": 32, "top_k": 32})
        small = submission.sparse_decode_cost(**{**_cfg(), "n_tokens": 512})
        if double_t["dense_bytes"] != 2 * base["dense_bytes"]:
            return False, "expected dense bytes to double with the context"
        if double_t["kv_bytes"] != base["kv_bytes"]:
            return False, f"expected kept KV bytes to stay {base['kv_bytes']:,}, got {double_t['kv_bytes']:,}"
        if double_t["indexer_bytes"] != 2 * base["indexer_bytes"]:
            return False, "expected indexer bytes to double with the context"
        if half_block["indexer_bytes"] != 2 * base["indexer_bytes"] or half_block["kv_bytes"] != base["kv_bytes"]:
            return False, "expected half the block size to double the indexer bytes and keep the KV bytes"
        if small["kept_tokens"] != 512:
            return False, f"expected kept_tokens capped at the context 512, got {small['kept_tokens']}"
        return True, "KV bytes stay flat, indexer bytes grow with T / block, kept tokens cap at T"

    c.check("sparse cost: KV bytes stay flat, the indexer scan grows with T / block", cost_scaling)

    # ---- Part 2: gated residual ----------------------------------------

    def fn_factory(d):
        g = torch.Generator().manual_seed(6)
        w = torch.randn(d, d, generator=g, dtype=f64)
        return lambda h: torch.tanh(w @ h)

    def identity_streams():
        d, n = 6, 4
        f = fn_factory(d)
        x = torch.randn(d, generator=torch.Generator().manual_seed(7), dtype=f64)
        X = x.repeat(n, 1)
        got, err = _call(
            submission, "gated_residual", X, f,
            torch.full((n,), 1.0 / n, dtype=f64), torch.ones(n, dtype=f64),
        )
        if err:
            return False, f"expected x + f(x) in every stream, got {err}"
        want = (x + f(x)).repeat(n, 1)
        worst = float((got - want).abs().max())
        c.metric("gated_identity_error", worst)
        return worst < 1e-12, f"expected x + f(x) in every stream, max error {worst:.1e}"

    c.check("equal streams with read = 1/n and write = 1 reduce to a plain residual", identity_streams)

    def one_hot():
        d, n = 6, 4
        f = fn_factory(d)
        X = torch.randn(n, d, generator=torch.Generator().manual_seed(8), dtype=f64)
        read = torch.zeros(n, dtype=f64)
        read[2] = 1.0
        write = torch.zeros(n, dtype=f64)
        write[2] = 1.0
        got, err = _call(submission, "gated_residual", X, f, read, write)
        if err:
            return False, f"expected stream 2 to become x + f(x) and the rest to stay, got {err}"
        want = X.clone()
        want[2] = X[2] + f(X[2])
        worst = float((got - want).abs().max())
        return worst < 1e-12, f"expected stream 2 to become x + f(x) and the rest to stay, max error {worst:.1e}"

    c.check("one-hot read and write gates act as an ordinary residual on one stream", one_hot)

    def read_mixes():
        d, n = 5, 4
        X = torch.randn(n, d, generator=torch.Generator().manual_seed(9), dtype=f64)
        read = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=f64)
        seen = {}

        def probe(h):
            seen["h"] = h.clone()
            return torch.zeros_like(h)

        got, err = _call(submission, "gated_residual", X, probe, read, torch.ones(n, dtype=f64))
        if err:
            return False, f"expected the layer input to be read @ X, got {err}"
        worst = float((seen["h"] - read @ X).abs().max())
        same = float((got - X).abs().max())
        return worst < 1e-12 and same < 1e-12, (
            f"expected the layer input read @ X and unchanged streams for f = 0, errors {worst:.1e} and {same:.1e}"
        )

    c.check("the read gate mixes the streams into the layer input, and f = 0 leaves them unchanged", read_mixes)

    def gates():
        d, n = 7, 4
        X = torch.randn(n, d, generator=torch.Generator().manual_seed(10), dtype=f64)
        zero = torch.zeros(d, dtype=f64)
        out, err = _call(submission, "dynamic_gates", X, zero, zero)
        if err:
            return False, f"expected uniform read and write 0.5 for zero weights, got {err}"
        read, write = out
        if not (torch.allclose(read, torch.full((n,), 0.25, dtype=f64)) and torch.allclose(write, torch.full((n,), 0.5, dtype=f64))):
            return False, f"expected read 0.25 and write 0.5, got {read.tolist()} and {write.tolist()}"
        w = torch.randn(d, generator=torch.Generator().manual_seed(11), dtype=f64)
        read, write = submission.dynamic_gates(X, w, -w)
        ok = abs(float(read.sum()) - 1) < 1e-12 and bool((read > 0).all()) and bool(((write > 0) & (write < 1)).all())
        if not ok:
            return False, "expected read to sum to 1 and write in (0, 1)"
        ok = torch.allclose(read, torch.softmax(X @ w, dim=0)) and torch.allclose(write, torch.sigmoid(-(X @ w)))
        return ok, "expected read = softmax(X @ w_read) and write = sigmoid(X @ w_write)"

    c.check("dynamic_gates gives a softmax read gate and a sigmoid write gate", gates)

    def gate_cost():
        got, err = _call(submission, "gated_residual_cost", 4, 8192, 2)
        if err:
            return False, f"expected the cost dict, got {err}"
        want = {
            "stream_bytes": 65536, "plain_bytes": 16384, "stream_ratio": 4,
            "rw_bytes": 131072, "plain_rw_bytes": 32768, "gate_flops": 262144,
        }
        for key, value in want.items():
            if got.get(key) != value:
                return False, f"expected {key} = {value:,}, got {got.get(key)}"
        return True, "4 streams of 8192 in bfloat16 hold 64 KiB per token, 4 times a plain residual"

    c.check("gated_residual_cost counts 4 streams at 4 times the residual bytes", gate_cost)

    # ---- Part 3: hashed n-gram embedding -------------------------------

    def determinism():
        ids = [5, 17, 17, 400, 3, 3, 3, 90, 12, 5]
        a, err = _call(submission, "ngram_indices", ids, 3, 1024)
        if err:
            return False, f"expected 10 indices in [0, 1024), got {err}"
        b = submission.ngram_indices(list(ids), 3, 1024)
        if tuple(a.shape) != (10,) or not torch.equal(a, b):
            return False, "expected the same indices on a second call"
        if int(a.min()) < 0 or int(a.max()) >= 1024:
            return False, f"expected indices in [0, 1024), got min {int(a.min())} and max {int(a.max())}"
        # Equal windows give equal rows: the three 3s at positions 4 to 6 end in
        # windows (17, 400, 3), (400, 3, 3), (3, 3, 3), so only position
        # pairs with equal windows may match.
        d, _ = _call(submission, "ngram_indices", [9, 9, 9, 9, 9], 2, 1024)
        if int(d[1]) != int(d[2]) or int(d[2]) != int(d[4]):
            return False, "expected equal windows (9, 9) to share a row"
        # Position 0 sees n - 1 pad ids before it, so its window is (PAD, 9).
        pad = getattr(submission, "PAD", -1)
        if int(d[0]) != submission.ngram_hash((pad, 9), 1024):
            return False, "expected position 0 to hash the window (PAD, id) for n = 2"
        return True, "deterministic, in range, equal windows share a row, and the start is padded"

    c.check("ngram_indices is deterministic and depends only on the window", determinism)

    def window_only():
        n, size = 3, 4096
        g = torch.Generator().manual_seed(12)
        ids = torch.randint(0, 1000, (30,), generator=g).tolist()
        base = submission.ngram_indices(ids, n, size)
        later = ids[:20] + [(t + 1) % 1000 for t in ids[20:]]
        far = ids[:10] + [(ids[10] + 1) % 1000] + ids[11:]  # position 10, 9 steps before 19
        after = submission.ngram_indices(later, n, size)
        before = submission.ngram_indices(far, n, size)
        if not torch.equal(base[:20], after[:20]):
            return False, "expected indices up to position 19 to ignore later ids"
        if not torch.equal(base[13:], before[13:]):
            return False, f"expected indices from position 13 on to ignore an id at position 10 for n = {n}"
        # Position 10 is inside the windows ending at 10, 11, and 12.
        if torch.equal(base[10:13], before[10:13]):
            return False, "expected a changed id to change the windows that contain it"
        return True, "causal, and only the last n ids matter"

    c.check("index t depends on ids t - n + 1 to t and on nothing else", window_only)

    def hash_quality():
        n_buckets = 64
        total = 0
        counts = [0] * n_buckets
        for a in range(150):
            for b in range(150):
                counts[submission.ngram_hash((a, b), n_buckets)] += 1
                total += 1
        mean = total / n_buckets
        if max(counts) > 1.2 * mean or min(counts) < 0.8 * mean:
            return False, f"expected bucket loads within 20% of {mean:.0f}, got {min(counts)} to {max(counts)}"
        same = sum(
            submission.ngram_hash((a, b), 1 << 20) == submission.ngram_hash((b, a), 1 << 20)
            for a in range(100) for b in range(100) if a != b
        )
        pairs = 100 * 99
        if same > 0.01 * pairs:
            return False, f"expected (a, b) and (b, a) to differ, {same} of {pairs} pairs collide"
        return True, f"bucket loads {min(counts)} to {max(counts)} around {mean:.0f}, and order changes the row"

    c.check("ngram_hash spreads 22,500 bigrams evenly and is order-sensitive", hash_quality)

    def embedding_and_inject():
        d, size = 8, 256
        table = torch.randn(size, d, generator=torch.Generator().manual_seed(13), dtype=f64)
        ids = [3, 1, 4, 1, 5]
        emb, err = _call(submission, "ngram_embedding", ids, table, 2)
        if err:
            return False, f"expected shape (5, 8), got {err}"
        idx = submission.ngram_indices(ids, 2, size)
        if tuple(emb.shape) != (5, d) or not torch.equal(emb, table[idx]):
            return False, "expected one table row per position, taken at ngram_indices"
        h = torch.randn(5, d, generator=torch.Generator().manual_seed(14), dtype=f64)
        for layer in (1, 3, 4):
            if not torch.equal(submission.inject_ngram(h, emb, layer), h):
                return False, f"expected hidden unchanged at layer {layer}"
        if not torch.allclose(submission.inject_ngram(h, emb, 2), h + emb):
            return False, "expected hidden + emb at layer 2"
        return True, "adds the rows at layer 2 and nowhere else"

    c.check("the lookup returns table rows and is added at layer 2 only", embedding_and_inject)

    def accounting():
        got, err = _call(submission, "ngram_accounting", 1 << 20, 64, 2, 2)
        if err:
            return False, f"expected the accounting dict, got {err}"
        want = {
            "params": 134217728,
            "stored_bytes": 268435456,
            "bytes_read_per_token": 256,
            "flops_per_token": 128,
            "dense_flops_per_token": 268435456,
        }
        for key, value in want.items():
            if got.get(key) != value:
                return False, f"expected {key} = {value:,}, got {got.get(key)}"
        bigger = submission.ngram_accounting(1 << 24, 64, 2, 2)
        if bigger["params"] != 16 * got["params"]:
            return False, "expected 16 times the parameters for 16 times the rows"
        if bigger["flops_per_token"] != got["flops_per_token"] or bigger["bytes_read_per_token"] != got["bytes_read_per_token"]:
            return False, "expected FLOPs and bytes read per token to stay the same when the table grows"
        c.metric("ngram_flop_ratio", got["flop_ratio"])
        return True, "16x the parameters, the same 128 FLOPs and 256 bytes read per token"

    c.check("n-gram parameters grow with the table, FLOPs and bytes read per token don't", accounting)
    return c.finish()
