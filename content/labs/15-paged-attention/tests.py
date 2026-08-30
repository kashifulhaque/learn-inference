import math

import torch
from lab_common import Checks, close, pick_device

KV_HEADS, HEAD_DIM, HEADS = 2, 16, 8
BLOCK_SIZE = 16


def run(submission):
    c = Checks()
    if not c.require(submission, "PagedCache", "paged_decode_attention",
                     "memory_waste"):
        return c.finish()

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

    c.metric("waste_contiguous", round(contiguous, 4))
    c.metric("waste_paged", round(paged, 4))
    c.metric("paged_error", float(f"{err:.3e}"))
    return c.finish()


def _raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False
