import math

import torch
from lab_common import Checks, close, pick_device

KV_HEADS, HEAD_DIM, HEADS = 2, 16, 8
BLOCK_SIZE = 16


def run(submission):
    c = Checks()
    try:
        _run(c, submission)
    except Exception as exc:  # noqa: BLE001 - an unfinished stub must still report
        c.add("the checks ran to completion", False, f"{type(exc).__name__}: {exc}")
    return c.finish()


def _run(c, submission):
    if not c.require(submission, "PagedCache", "paged_decode_attention",
                     "memory_waste", "preemption_cost"):
        return

    device = pick_device()
    torch.manual_seed(0)

    cache = submission.PagedCache(64, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    c.check("a new pool has every block free", lambda: cache.free_blocks == 64,
            f"got {cache.free_blocks}")

    c.check("20 tokens need 2 blocks of 16",
            lambda: cache.blocks_needed("a", 20) == 2,
            f"got {cache.blocks_needed('a', 20)}")
    c.check("exactly 16 tokens need 1 block",
            lambda: cache.blocks_needed("a", 16) == 1)

    table = cache.allocate("a", 20)
    c.check("allocation returns a table of 2 blocks", lambda: len(table) == 2,
            f"got {len(table)}")
    c.check("allocation removes them from the free list",
            lambda: cache.free_blocks == 62, f"got {cache.free_blocks}")

    # Growing by one token inside the tail block must not take a new block.
    cache.allocate("a", 1)
    c.check("appending inside the tail block takes no new block",
            lambda: cache.free_blocks == 62, f"got {cache.free_blocks}")
    # Crossing into the third block must.
    cache.allocate("a", 12)
    c.check("crossing a block boundary takes a new block",
            lambda: cache.free_blocks == 61, f"got {cache.free_blocks}")

    # Blocks need not be contiguous, and a write-then-read must be exact.
    fresh = submission.PagedCache(64, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    fresh.allocate("s1", 40)
    k = torch.randn(40, KV_HEADS, HEAD_DIM, device=device)
    v = torch.randn(40, KV_HEADS, HEAD_DIM, device=device)
    fresh.write("s1", torch.arange(40, device=device), k, v)
    gk, gv = fresh.gather("s1")
    c.check("a write-then-gather round trip is exact",
            lambda: close(gk, k, 0)[0] and close(gv, v, 0)[0])

    # Two sequences must not overwrite each other.
    fresh.allocate("s2", 30)
    k2 = torch.randn(30, KV_HEADS, HEAD_DIM, device=device)
    fresh.write("s2", torch.arange(30, device=device), k2, k2)
    c.check("a second sequence leaves the first intact",
            lambda: close(fresh.gather("s1")[0], k, 0))

    before = fresh.free_blocks
    fresh.release("s2")
    c.check("release returns every block",
            lambda: fresh.free_blocks == before + math.ceil(30 / BLOCK_SIZE),
            f"{before} -> {fresh.free_blocks}")

    small = submission.PagedCache(2, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    c.check("running out of blocks raises", lambda: _raises(
        lambda: small.allocate("big", 100)))

    # Attention through the block table must equal attention over the gathered
    # prefix.
    pool = submission.PagedCache(64, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    seq_ids, lengths = ["p1", "p2", "p3"], [37, 5, 64]
    for seq_id, length in zip(seq_ids, lengths):
        pool.allocate(seq_id, length)
        kk = torch.randn(length, KV_HEADS, HEAD_DIM, device=device)
        vv = torch.randn(length, KV_HEADS, HEAD_DIM, device=device)
        pool.write(seq_id, torch.arange(length, device=device), kk, vv)

    q = torch.randn(3, HEADS, HEAD_DIM, device=device)
    mine = submission.paged_decode_attention(pool, seq_ids, q)

    group = HEADS // KV_HEADS
    expected = []
    for index, seq_id in enumerate(seq_ids):
        keys, values = pool.gather(seq_id)
        keys = keys.repeat_interleave(group, dim=1).float()
        values = values.repeat_interleave(group, dim=1).float()
        scores = torch.einsum("hd,thd->ht", q[index].float(), keys) * HEAD_DIM**-0.5
        expected.append(torch.einsum("ht,thd->hd", scores.softmax(-1), values))
    expected = torch.stack(expected)

    err = (mine.float() - expected).abs().max().item()
    c.check("paged attention matches attention over the gathered prefix",
            lambda: (err < 1e-4, f"max abs error {err:.2e}"))
    c.check("the output has one row per sequence",
            lambda: tuple(mine.shape) == (3, HEADS, HEAD_DIM),
            f"got {tuple(mine.shape)}")

    # Waste on a realistic mix of lengths.
    lengths = [37, 5, 64, 200, 1200, 18, 450, 12, 3000, 90]
    contiguous, paged = submission.memory_waste(lengths, 32768, BLOCK_SIZE)
    c.check("a contiguous cache wastes most of its reservation",
            lambda: contiguous > 0.95, f"{contiguous:.1%}")
    c.check("paging wastes only the tail block",
            lambda: paged < 0.05, f"{paged:.2%}")
    c.check("paging wastes less than contiguous", lambda: paged < contiguous)

    _check_sharing(c, submission, device)
    _check_preemption(c, submission)

    c.metric("waste_contiguous", round(contiguous, 4))
    c.metric("waste_paged", round(paged, 4))
    c.metric("paged_error", float(f"{err:.3e}"))


def _raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False


def _min_refcount(cache):
    """Every refcount reachable from the pool must be non-negative."""
    return min(cache.refcount(b) for b in range(cache.num_blocks))


def _check_sharing(c, submission, device):
    """Fork, copy-on-write, and reference counts.

    Wrong implementations these checks catch:
    - NeverCopy: append_token writes through a shared tail, so the parent's
      cached K changes and the two tables keep one tail block.
    - CopyOnEveryAppend: after a fork, every append takes a new block, even once
      the tail is private, so the free count falls by one per token.
    - CopyFullTail: copies a shared tail even when it is full, wasting a block.
    """
    torch.manual_seed(1)
    kv = (KV_HEADS, HEAD_DIM)

    cache = submission.PagedCache(64, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    cache.allocate("p", 20)  # one full block and a 4-token tail
    pk = torch.randn(20, *kv, device=device)
    pv = torch.randn(20, *kv, device=device)
    cache.write("p", torch.arange(20, device=device), pk, pv)
    free_before = cache.free_blocks

    cache.fork("p", "c")
    c.check("fork copies no block",
            lambda: cache.free_blocks == free_before,
            f"expected {free_before} free, got {cache.free_blocks}")
    c.check("fork shares the same block table",
            lambda: cache.tables["c"] == cache.tables["p"],
            f"parent {cache.tables['p']}, child {cache.tables['c']}")
    c.check("every shared block has refcount 2",
            lambda: all(cache.refcount(b) == 2 for b in cache.tables["p"]),
            f"got {[cache.refcount(b) for b in cache.tables['p']]}")

    nk = torch.randn(*kv, device=device)
    nv = torch.randn(*kv, device=device)
    cache.append_token("c", nk, nv)
    c.check("the first divergent append copies exactly one block",
            lambda: cache.free_blocks == free_before - 1,
            f"expected {free_before - 1} free, got {cache.free_blocks}")
    c.check("the full block stays shared and the tail splits",
            lambda: (cache.tables["c"][0] == cache.tables["p"][0]
                     and cache.tables["c"][1] != cache.tables["p"][1]),
            f"parent {cache.tables['p']}, child {cache.tables['c']}")
    c.check("refcounts after divergence are [2, 1] for the shared prefix",
            lambda: ([cache.refcount(b) for b in cache.tables["p"]] == [2, 1]
                     and cache.refcount(cache.tables["c"][1]) == 1),
            f"parent blocks {[cache.refcount(b) for b in cache.tables['p']]}, "
            f"child tail {cache.refcount(cache.tables['c'][1])}")

    gk, gv = cache.gather("p")
    c.check("the parent's cached K and V are untouched by the child's write",
            lambda: close(gk, pk, 0)[0] and close(gv, pv, 0)[0]
            and cache.lengths["p"] == 20,
            f"parent length {cache.lengths['p']}, expected 20")
    ck, _ = cache.gather("c")
    c.check("the child holds the shared prefix plus its own token",
            lambda: close(ck[:20], pk, 0)[0] and close(ck[20], nk, 0)[0]
            and cache.lengths["c"] == 21)

    # The parent now writes its own token at position 20. NeverCopy shares one
    # tail block, so this overwrites the child's token.
    ok_k = torch.randn(*kv, device=device)
    cache.append_token("p", ok_k, ok_k)
    c.check("a sibling's write to the same position does not reach the child",
            lambda: close(cache.gather("c")[0][20], nk, 0)[0]
            and close(cache.gather("p")[0][20], ok_k, 0)[0]
            and cache.free_blocks == free_before - 1,
            f"expected {free_before - 1} free, got {cache.free_blocks}")

    # 11 more tokens fill the child's private tail (positions 21..31): no new
    # block. A copy-on-every-append implementation fails here.
    for _ in range(11):
        cache.append_token("c", torch.randn(*kv, device=device),
                           torch.randn(*kv, device=device))
    c.check("appends into a private tail take no block",
            lambda: cache.free_blocks == free_before - 1,
            f"expected {free_before - 1} free, got {cache.free_blocks}")
    cache.append_token("c", nk, nv)  # position 32 starts a third block
    c.check("crossing the boundary takes one new block",
            lambda: cache.free_blocks == free_before - 2,
            f"expected {free_before - 2} free, got {cache.free_blocks}")

    # A shared tail that is exactly full needs a new block, not a copy.
    full = submission.PagedCache(8, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    full.allocate("p", 16)
    full.fork("p", "c")
    full.append_token("c", nk, nv)
    c.check("a full shared tail gets a fresh block, and the old one stays shared",
            lambda: (full.free_blocks == 8 - 2 and full.refcount(full.tables["p"][0]) == 2
                     and len(full.tables["c"]) == 2),
            f"expected 6 free, got {full.free_blocks}")

    # Release in both orders: counts never go negative, blocks return only at 0.
    free_mid = cache.free_blocks
    cache.release("p")
    shared = cache.tables["c"][0]
    c.check("releasing the parent keeps the block the child still holds",
            lambda: (cache.refcount(shared) == 1 and shared not in cache.free
                     and cache.free_blocks == free_mid + 1),
            f"refcount {cache.refcount(shared)}, "
            f"expected {free_mid + 1} free, got {cache.free_blocks}")
    cache.release("c")
    c.check("releasing the last owner returns every block",
            lambda: cache.free_blocks == 64 and sorted(cache.free) == list(range(64)),
            f"expected 64 free, got {cache.free_blocks}")
    c.check("no refcount is negative or left over",
            lambda: all(cache.refcount(b) == 0 for b in range(64)),
            f"min {_min_refcount(cache)}")

    # No free block means a divergent append raises and changes nothing.
    tight = submission.PagedCache(1, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=device)
    tight.allocate("p", 4)
    tight.fork("p", "c")
    raised = _raises(lambda: tight.append_token("c", nk, nv))
    c.check("a divergent append with no free block raises",
            lambda: raised)
    c.check("the failed append leaves lengths and counts unchanged",
            lambda: (tight.lengths["c"] == 4 and tight.refcount(tight.tables["c"][0]) == 2),
            f"length {tight.lengths['c']}, refcount "
            f"{tight.refcount(tight.tables['c'][0])}")


def _check_preemption(c, submission):
    """Swap moves KV bytes out and back. Recompute moves none and costs FLOPs.

    Numbers are chapter 2's: 64 KiB of KV per token, 154,927,104 bytes of
    recurrent state, and 26.9e9 parameters.
    """
    kv_per_token = 65_536
    state = 154_927_104
    params = 26_900_000_000

    cost = submission.preemption_cost(2000, kv_per_token, BLOCK_SIZE, params, state)
    one_way = 2000 * kv_per_token + state  # 2000 is a whole number of blocks
    c.check("swap bytes out for 2000 tokens plus recurrent state",
            lambda: cost["swap_bytes_out"] == 285_999_104,
            f"expected 285,999,104, got {cost['swap_bytes_out']:,}")
    c.check("swap bytes in equal swap bytes out",
            lambda: cost["swap_bytes_in"] == one_way,
            f"expected {one_way:,}, got {cost['swap_bytes_in']:,}")
    c.check("swap total is out plus in",
            lambda: cost["swap_bytes_total"] == 571_998_208,
            f"expected 571,998,208, got {cost['swap_bytes_total']:,}")
    c.check("recompute moves zero bytes",
            lambda: cost["recompute_bytes"] == 0,
            f"got {cost['recompute_bytes']}")
    c.check("recompute costs 2 * params * tokens FLOPs",
            lambda: cost["recompute_flops"] == 107_600_000_000_000,
            f"expected 107,600,000,000,000, got {cost['recompute_flops']:,}")

    # 2001 tokens occupy 126 blocks, and a swap moves whole blocks.
    odd = submission.preemption_cost(2001, kv_per_token, BLOCK_SIZE, params)
    expected = 126 * BLOCK_SIZE * kv_per_token
    c.check("swap rounds up to whole blocks",
            lambda: odd["swap_bytes_out"] == expected,
            f"expected {expected:,}, got {odd['swap_bytes_out']:,}")
    c.check("recompute FLOPs follow tokens, not blocks",
            lambda: odd["recompute_flops"] == 2 * params * 2001,
            f"expected {2 * params * 2001:,}, got {odd['recompute_flops']:,}")
