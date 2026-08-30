---
title: Paged attention
slug: 15-paged-attention
part: "Part 4 — Kernels"
summary: Applying virtual memory to the KV cache, and the kernel that follows a block table.
minutes: 80
gpu: true
objectives:
  - Explain the three kinds of waste a contiguous KV cache creates.
  - Implement a block allocator with a block table per sequence.
  - Write a decode attention kernel that reads through the block table.
lab: 15-paged-attention
---

# Paged attention

The contiguous cache from chapter 9 reserves `max_seq_len` per sequence. That's
correct and it wastes most of your memory. Paging fixes it with the same idea an
operating system uses for RAM.

## Three kinds of waste

**Reserved but unused.** A sequence that might reach 32k tokens reserves 2 GiB. If
it stops at 300 tokens, 98% of that reservation was never touched — and it was
unavailable to anyone else the whole time.

**Internal fragmentation.** Even at the reservation's end, the last partially
filled region is wasted.

**External fragmentation.** Sequences finishing at different times leave gaps too
small for the next request, so allocation fails while plenty of total memory is
free.

Measured on production traces, contiguous caches waste 60 to 80% of KV memory.
Since the cache sets the batch size and batch size sets throughput, that's a
direct throughput loss.

## Blocks

Divide the cache into fixed-size blocks, typically 16 tokens. A sequence gets a
*block table*: an ordered list of the physical blocks holding its tokens. Blocks
need not be adjacent.

```python
shape = (num_blocks, block_size, num_kv_heads, head_dim)
self.k_blocks = {layer: torch.zeros(shape) for layer in full_attention_layers}
self.free = list(range(num_blocks))
self.tables: dict[str, BlockTable] = {}
```

A sequence holds only the blocks it has filled. Waste is bounded by
`block_size - 1` tokens in the tail block — under 0.4% at block size 16 for any
sequence over 4k tokens. External fragmentation disappears entirely, because every
block is interchangeable.

Allocation becomes a list pop:

```python
def allocate(self, seq_id, extra_tokens):
    table = self.tables.setdefault(seq_id, BlockTable(blocks=[]))
    need = self.blocks_needed(seq_id, extra_tokens)
    if need > len(self.free):
        raise OutOfBlocks(...)
    for _ in range(need):
        table.blocks.append(self.free.pop())
    table.length += extra_tokens
    return table
```

## Mapping positions to slots

Logical position *p* lives at block `p // block_size`, offset `p % block_size`:

```python
def slot_indices(self, seq_id, positions):
    blocks = torch.tensor(self.tables[seq_id].blocks)
    return blocks[positions // self.block_size] * self.block_size + (
        positions % self.block_size
    )
```

That's a page table lookup. The block size is the page size, and the trade is the
familiar one: smaller blocks waste less and cost more indirection.

## The kernel

Decode attends one query against the whole cached prefix. There's no score matrix
worth tiling and no reuse of Q across queries, so the kernel is bound entirely by
how fast it reads K and V. It walks the block table:

```python
for b in range(0, MAX_BLOCKS):
    if b < num_blocks:
        physical = tl.load(block_table_ptr + seq * stride_tb + b * stride_ts)
        token_pos = b * BLOCK_SIZE + offs_s
        valid = token_pos < context_len

        k = tl.load(k_cache_ptr + physical * stride_kb + ..., mask=valid[:, None])
        scores = tl.sum(k * q[None, :], axis=1)
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        correction = tl.exp(m_i - m_new)
        p = tl.where(valid, tl.exp(scores - m_new), 0.0)

        l_i = l_i * correction + tl.sum(p, axis=0)
        acc = acc * correction + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
```

Same online softmax as chapter 14. The only difference is where the keys come
from.

The loop runs to `MAX_BLOCKS`, a compile-time constant, with a runtime guard,
because Triton needs a static trip count. Sequences shorter than the maximum skip
their extra iterations cheaply.

Within a block, the 16 tokens are contiguous, so each block read is coalesced.
That's why block size matters for speed as well as waste: at block size 1 every
read is a separate transaction.

## Prefix sharing

Two requests with the same system prompt can point at the same physical blocks for
the shared prefix. Add a reference count per block, free only at zero, and copy a
block on write when a shared sequence diverges.

For a service where every request carries the same 500-token system prompt, this
removes that prompt's cache from all but one request, and skips recomputing its
prefill. It's the largest single win available to a chat serving stack, and it
falls out of the block abstraction almost for free.

## Preemption

When blocks run out, the scheduler must free some. Two options.

*Swap* the victim's blocks to host memory and copy them back later. Preserves the
work; costs PCIe bandwidth, which is roughly 60 times slower than HBM.

*Recompute* — drop the blocks and prefill the sequence again when it's readmitted.
Wastes the work; costs no transfer.

Recomputation usually wins, because prefill is compute bound and fast while the
PCIe transfer is not. `Scheduler._preempt` recomputes, and evicts the most recently
admitted sequence so the ones closest to finishing get to finish.

## Lab

Implement the block allocator, the position-to-slot mapping, and the paged decode
kernel. The harness checks that a write-then-gather round trip is exact, that the
paged kernel matches naive attention on the same data, and that allocation and
release conserve blocks. It also reports memory waste against a contiguous cache
on a realistic mix of sequence lengths.

## Further reading

- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180)
