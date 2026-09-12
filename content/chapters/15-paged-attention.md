---
title: Paged attention
slug: 15-paged-attention
part: "Part 4 — Kernels"
summary: Deriving the fragmentation numbers, building a block allocator, and writing a decode kernel that follows a block table.
minutes: 130
gpu: true
objectives:
  - Quantify the three kinds of waste a contiguous KV cache creates on a real length distribution.
  - Derive the page size trade-off between internal fragmentation and indirection.
  - Implement a block allocator with a per-sequence block table and a free list.
  - Write a decode attention kernel that reads through the block table.
  - Explain what prefix sharing saves, and what the recurrent layers cost that paging cannot fix.
lab: 15-paged-attention
---

# Paged attention

The cache from chapter 9 works and wastes most of your memory. It reserves
`max_seq_len` slots per sequence at admission time, because attention wants to
read a contiguous range and a contiguous range has to be reserved before you know
how long the sequence will be.

Operating systems solved this problem in the 1960s. A process gets an address
space that looks contiguous, backed by physical pages that are not, with a page
table translating between them. Paged attention is the same idea applied to the
KV cache: fixed-size blocks, a per-sequence block table, and a kernel that
follows the table instead of a pointer.

This chapter does the arithmetic rather than asserting the conclusion. Paging is
not obviously worth an indirection on the hottest read in the engine, and the
case for it is quantitative.

## Before you start

**The KV cache from chapter 9.** Sixteen of this model's 64 layers use full
attention, and each one stores a key and a value vector per KV head per token.
The per-token cost across those 16 layers is 64 KiB, derived in chapter 2 and
re-derived below. The other 48 layers keep a fixed-size recurrent state and
contribute nothing to cache growth.

**The memory budget from chapter 2.** Weights take 53.8 GB of the A100's 80 GB.
After the CUDA context, an activation workspace, and fragmentation headroom,
about 19 GB is left for the cache. That is 18,120 MiB, and every number in this
chapter is measured against it.

**Paging vocabulary.** A *page* (here, a *block*) is the allocation unit. A
*page table* (here, a *block table*) maps logical positions to physical ones.
*Internal fragmentation* is space wasted inside an allocated page. *External
fragmentation* is free space that cannot be used because it is in the wrong-sized
pieces. *Copy-on-write* means two owners share one physical page until one of
them writes, at which point it gets a private copy.

**The online softmax from chapter 14.** The decode kernel here streams blocks and
maintains the same running maximum, denominator, and accumulator. If that
derivation is not solid, go back; this chapter assumes it.

## What a contiguous cache actually wastes

Three kinds of waste, in order of size.

**Reserved but unused.** A request that might reach 32,768 tokens reserves for
32,768 tokens. At 64 KiB per token that is

$$
32768 \times 64\ \text{KiB} = 2\ \text{GiB}
$$

per sequence, committed at admission, whatever the request turns out to be.

**Internal fragmentation.** Even at the end of a sequence's real length, the
final partially filled unit is unusable by anyone else.

**External fragmentation.** Sequences finish at different times and leave holes.
A new request needing a contiguous run of 2 GiB can fail while 6 GiB is free in
pieces.

Put numbers on the first one using the length mix the lab measures, which is a
plausible chat trace: prompts and completions of 37, 5, 64, 200, 1200, 18, 450,
12, 3000, and 90 tokens.

| | Tokens reserved | Memory | Waste |
|---|---|---|---|
| Contiguous, 32k max | 327,680 | 20 GiB | 98.45% |
| Paged, 16-token blocks | 5,152 | 322 MiB | 1.48% |
| Actually used | 5,076 | 317 MiB | — |

The contiguous row is $10 \times 32768$ slots. The paged row rounds each length
up to a multiple of 16 and sums: 48, 16, 64, 208, 1200, 32, 464, 16, 3008, 96.
The waste figures are $1 - 5076/327680$ and $1 - 5076/5152$.

Read the memory column again. Ten concurrent requests under a contiguous cache
need 20 GiB, which is more than the 18,120 MiB the card has left after the
weights. Ten. The same ten sequences under paging need 322 MiB, and the card
would hold hundreds of them. This is not an optimization; it is the difference
between a serving engine and a demo.

Since the cache sets the batch size, and batch size sets throughput, that waste
converts directly into throughput you do not get.

## Blocks and the block table

Divide the pool into fixed-size blocks. A sequence gets an ordered list of the
physical blocks holding its tokens. The blocks need not be adjacent and need not
be in order.

```python
shape = (num_blocks, block_size, num_kv_heads, head_dim)
self.k_blocks = {layer: torch.zeros(shape) for layer in full_attention_layers}
self.v_blocks = {layer: torch.zeros(shape) for layer in full_attention_layers}
self.free = list(range(num_blocks))
self.tables: dict[str, BlockTable] = {}
```

The pool is allocated once, at startup, at the full size you intend to use. From
then on nothing is allocated or freed on the device: allocation is a list pop,
release is a list extend, and the CUDA allocator never sees another request. That
alone removes a class of latency spikes.

External fragmentation disappears completely, because every block is
interchangeable — any free block can serve any sequence. Internal fragmentation
is bounded by `block_size - 1` tokens in the tail block, and nothing else.

## What a page costs on this model

Derive the per-token cost rather than quoting it. For one full-attention layer,
one token stores a key and a value for each of the 4 KV heads, each of dimension
256, in bfloat16:

$$
2 \times 4 \times 256 \times 2\ \text{bytes} = 4096\ \text{bytes} = 4\ \text{KiB}.
$$

Across the 16 full-attention layers:

$$
16 \times 4\ \text{KiB} = 64\ \text{KiB per token}.
$$

A block of 16 tokens therefore costs 64 KiB per layer, and

$$
16 \times 64\ \text{KiB} = 1\ \text{MiB}
$$

across all 16 layers. One page, one mebibyte. That is a convenient number to
keep in your head.

How many pages does the card afford? Start from the whole 80 GB, subtract the
weights, then subtract the overheads chapter 2 budgets:

$$
80 - 53.8 = 26.2\ \text{GB},
$$

minus roughly 1 GB of CUDA context, 3 GB of activation workspace, and 2 GB of
fragmentation headroom, leaving about 19 GB. In binary units that is 18,120 MiB,
so

$$
18{,}120 \text{ pages of } 1\ \text{MiB} = 289{,}920 \text{ tokens}
$$

of KV cache. Call it 290,000 tokens, or 18,000 pages. Spend them how you like:
70 sequences at 4k context, 8 sequences at 32k, one at 262k with room to spare.
The pool is a single global budget, and the scheduler in chapter 16 is what
decides who gets it.

## Mapping positions to slots

Logical position $p$ lives in the sequence's block $\lfloor p / B \rfloor$ at
offset $p \bmod B$, where $B$ is the block size. Composing that with the block
table gives a flat index into the pool:

$$
\operatorname{slot}(p) = \text{table}[\lfloor p / B \rfloor] \cdot B + (p \bmod B).
$$

In code, vectorized over a tensor of positions:

```python
def slot_indices(self, seq_id, positions):
    blocks = torch.tensor(self.tables[seq_id].blocks)   # (num_blocks_held,)
    return blocks[positions // self.block_size] * self.block_size + (
        positions % self.block_size
    )
```

`positions` has shape `(tokens,)` and so does the result. One gather into the
block table, one multiply, one add. Writing a chunk of new KV is then a scatter:

```python
kb = self.k_blocks[layer].view(-1, num_kv_heads, head_dim)   # (num_blocks * B, kv_heads, dim)
kb[slots] = k                                                # k: (tokens, kv_heads, dim)
```

The `view` flattens the block and offset dimensions into one, which is what makes
a single flat index work. `gather` runs the same map in reverse to reconstruct a
contiguous prefix; the lab uses it to check the kernel, and a real decode never
calls it, because gathering defeats the point.

## Choosing the page size

The page size is the one free parameter, and it trades two costs against each
other.

**Smaller pages waste less.** Average internal fragmentation is $(B-1)/2$ tokens
per sequence, assuming lengths fall anywhere within a block with equal
probability.

**Larger pages cost less indirection.** The block table holds
$\lceil L/B \rceil$ entries, and the kernel does one dependent load per block.

At 32k context, in this model's units:

| Block size $B$ | Avg. tail waste | Table entries | Table bytes | Loads per decode |
|---|---|---|---|---|
| 1 | 0 | 32,768 | 128 KiB | 32,768 |
| 16 | 7.5 tokens (480 KiB) | 2,048 | 8 KiB | 2,048 |
| 64 | 31.5 tokens (1.97 MiB) | 512 | 2 KiB | 512 |
| 256 | 127.5 tokens (7.97 MiB) | 128 | 512 B | 128 |

Both columns are small compared to the 2 GiB of KV a 32k sequence holds, which
is the real finding: the page size barely matters for memory, within an order of
magnitude either side of 16. Even at $B = 256$ the average tail waste is 0.4% of
a 32k sequence.

The indirection is not a bandwidth problem either. At $B = 16$, the kernel reads
4 bytes of block table to locate 1 MiB of KV — a ratio of one to 262,144. It is a
*latency* problem: the address of the next load depends on the value of the
previous one, so the memory system cannot run ahead. Larger blocks amortize that
dependency over more useful bytes, and they let the compiler unroll the inner
loop.

Sixteen is the conventional choice and the one the lab uses. It keeps waste under
half a percent for any sequence past a few thousand tokens, and 16 tokens of one
head is 8 KiB of contiguous bytes, enough for the load to coalesce cleanly.

The block table's own memory is negligible and worth confirming once. At the
model's 262k maximum context, a table is
$262144/16 = 16{,}384$ entries of 4 bytes, or 64 KiB per sequence. A batch of 64
holds 4 MiB of tables against 18,120 MiB of pool: 0.02%. You never need to think
about it again.

## The free list

Allocation is a stack.

```python
def blocks_needed(self, seq_id, extra_tokens):
    table = self.tables.get(seq_id)
    current = table.length if table else 0
    have = len(table.blocks) * self.block_size if table else 0
    return max(0, -(-(current + extra_tokens - have) // self.block_size))

def allocate(self, seq_id, extra_tokens):
    table = self.tables.setdefault(seq_id, BlockTable(blocks=[]))
    need = self.blocks_needed(seq_id, extra_tokens)
    if need > len(self.free):
        raise OutOfBlocks(
            f"{seq_id} needs {need} blocks, {len(self.free)} free. Preempt a "
            "running sequence or lower the batch size."
        )
    for _ in range(need):
        table.blocks.append(self.free.pop())
    table.length += extra_tokens
    return table
```

`blocks_needed` is the piece worth reading twice. `have` is the capacity the
sequence already holds; `current + extra_tokens - have` is the shortfall in
tokens; `-(-x // B)` is an integer ceiling division. When a sequence grows by one
token inside its tail block, the shortfall is negative, the ceiling is negative,
and `max(0, ...)` returns zero. No new block. That is the common case during
decode: fifteen steps out of sixteen allocate nothing.

`self.free.pop()` takes from the end, so the free list is last-in, first-out. A
block released a moment ago is the next one handed out, which is the friendliest
order for L2 and the TLB. It also means a sequence's blocks are scattered and
descending, which is fine — the block table does not care.

**When allocation fails mid-decode.** This is the case that matters. A sequence
that has been running for 500 steps asks for its 501st token, crosses a block
boundary, and the free list is empty. The allocator raises; it does not wait, and
it does not partially allocate. The scheduler catches the failure and preempts
someone, which chapter 16 covers in detail.

Two consequences follow, and both are design decisions rather than accidents:

Failure must be *detectable before the write*. `blocks_needed` is a pure query,
so the scheduler can ask before committing. A design that discovered the shortage
halfway through writing K and V would leave the cache inconsistent.

Failure must be *rare*, because the recovery is expensive. The scheduler keeps a
watermark — 2% of the pool, about 362 pages here — unallocated, so that it hits
the wall with room to make a decision instead of with nothing left to move.

Releasing is the reverse, and it must return every block:

```python
def release(self, seq_id):
    table = self.tables.pop(seq_id, None)
    if table:
        self.free.extend(table.blocks)
```

A leak here does not crash. It shrinks capacity slowly until, days later, the
engine preempts constantly at a batch size it used to hold. The lab checks
conservation for this reason, and so should your production metrics.

## The kernel

Decode attends one query token against the whole cached prefix. There is no score
matrix worth tiling and no reuse of $Q$ across query rows, so the kernel is bound
entirely by how fast it reads $K$ and $V$. One program handles one
(sequence, head) pair.

```python
seq = tl.program_id(0)
head = tl.program_id(1)
kv_head = head // kv_group

context_len = tl.load(context_len_ptr + seq)             # scalar
offs_d = tl.arange(0, HEAD_DIM)                          # (HEAD_DIM,)
q = tl.load(q_ptr + seq * stride_qs + head * stride_qh + offs_d * stride_qd)
q = q.to(tl.float32) * scale                             # (HEAD_DIM,)

m_i = float("-inf")                                      # scalar
l_i = 0.0                                                # scalar
acc = tl.zeros([HEAD_DIM], dtype=tl.float32)             # (HEAD_DIM,)
```

The running state is two scalars and a $d$-vector, not the $(B_r,)$ vectors of
chapter 14, because there is exactly one query row. The scale is folded into $q$
once instead of into the scores every iteration.

```python
offs_s = tl.arange(0, BLOCK_SIZE)                        # (BLOCK_SIZE,)
num_blocks = tl.cdiv(context_len, BLOCK_SIZE)

for b in range(0, MAX_BLOCKS):
    if b < num_blocks:
        physical = tl.load(block_table_ptr + seq * stride_tb + b * stride_ts)
        token_pos = b * BLOCK_SIZE + offs_s               # (BLOCK_SIZE,)
        valid = token_pos < context_len                   # (BLOCK_SIZE,)
```

This is the indirection, in one line. `physical` is the block id from the table;
every address below is computed from it. The loop runs to `MAX_BLOCKS`, a
compile-time constant, with a runtime guard, because Triton needs a static trip
count. Sequences shorter than the maximum skip their extra iterations cheaply —
the guard is uniform across the program, so there is no warp divergence, just a
predicated branch over an empty body.

```python
        k_ptrs = (
            k_cache_ptr + physical * stride_kb
            + offs_s[:, None] * stride_ks + kv_head * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(..., mask=valid[:, None], other=0.0).to(tl.float32)
```

`k` and `v` are `(BLOCK_SIZE, HEAD_DIM)`. The mask kills the tail of the last
block, where `token_pos` runs past `context_len`.

```python
        scores = tl.sum(k * q[None, :], axis=1)          # (BLOCK_SIZE,)
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        correction = tl.exp(m_i - m_new)
        p = tl.where(valid, tl.exp(scores - m_new), 0.0)  # (BLOCK_SIZE,)

        l_i = l_i * correction + tl.sum(p, axis=0)
        acc = acc * correction + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
```

The score computation is a broadcast multiply and a reduction, not `tl.dot`: with
one query row there is no matmul shape for the tensor cores to exploit. Past that
line, this is chapter 14's online softmax verbatim — same running maximum, same
correction factor $e^{m_{\text{old}} - m_{\text{new}}}$, same two corrected
accumulators. The only thing that changed is where the keys came from.

```python
acc = acc / tl.where(l_i == 0.0, 1.0, l_i)
tl.store(out_ptr + seq * stride_os + head * stride_oh + offs_d * stride_od,
         acc.to(out_ptr.dtype.element_ty))
```

One division at the end, one store of $d$ elements.

Note what the mask does for correctness at the boundary. Invalid slots load as
zeros; their score becomes $-\infty$; $e^{-\infty - m}$ is zero, so they add
nothing to $\ell$ and nothing to the accumulator. The explicit
`tl.where(valid, p, 0.0)` is belt and braces against the case where `m_new`
itself is $-\infty$, which would otherwise produce NaN.

## Prefix sharing and copy-on-write

Once addressing is indirect, two sequences can point at the same physical block.
Add a reference count per block, free only when it reaches zero, and copy a block
before writing to it if its count is above one. That is copy-on-write, and it
buys the single largest saving available to a chat stack.

Take the common case: a service where every request carries the same 500-token
system prompt.

**Memory.** Each request's copy of that prefix costs
$500 \times 64\ \text{KiB} = 31.25\ \text{MiB}$. With 64 concurrent requests, 63
of them stop paying it:

$$
63 \times 31.25\ \text{MiB} = 1.92\ \text{GiB},
$$

about 11% of the 18,120 MiB pool, recovered for free.

**Compute.** The shared prefix also does not need prefilling again. Prefill costs
roughly $2 N T$ FLOPs for $N$ parameters and $T$ tokens, and this model has
26.9 billion parameters, so

$$
2 \times 26.9 \times 10^9 \times 500 = 26.9\ \text{TFLOP}
$$

per request, which is 86 ms of tensor-core time at the A100's 312 TFLOP/s — and
more in wall-clock, since no real kernel hits peak. Every request after the first
skips it. That is a direct cut to time-to-first-token, and it is the reason
prefix caching shows up in benchmarks as a latency win rather than only a memory
win.

Two details bite in practice.

*Sharing is page-aligned.* A 500-token prefix is 31 full blocks (496 tokens) plus
4 tokens in a partial block. Only the full blocks can be shared; the partial one
must be copied, because the next sequence will write its own token 497 into it.
Prefixes that are exact multiples of the block size share perfectly, and a system
prompt is a fixed string, so padding it is a legitimate trick.

*The match must be a prefix, not a substring.* Attention at position $p$ depends
on every token before $p$, so a cached block is only reusable if every preceding
block matched too. Implementations hash the token ids of the prefix up to and
including each block, and look the block up by that hash.

## The 48 layers that do not page

Everything so far applies to 16 of this model's 64 layers. The other 48 are
linear-attention layers, and they hold a fixed-size recurrent state per sequence:
147.8 MiB, derived in chapter 2, constant whatever the context length.

That state does not page, and the reason is structural rather than an
implementation gap. A KV cache is append-only: token $p$'s entry is written once
and read forever, so it can live anywhere. A recurrent state is read and
overwritten in full on every step. There is no per-token granularity to page, no
part of it that is cold, and nothing to share between two sequences that have
diverged by even one token.

So the per-sequence cost has two parts with different shapes:

$$
\text{cost}(L) = 147.8\ \text{MiB} + L \times 64\ \text{KiB}.
$$

The fixed term is worth converting into the pool's own units:

$$
\frac{147.8\ \text{MiB}}{64\ \text{KiB}} = 2365\ \text{tokens}.
$$

Admitting a sequence costs as much as 2365 tokens of KV before it has a single
token of context. Three consequences:

**Batch size has a hard ceiling from the fixed term alone.** With 18,120 MiB in
the pool, $18{,}120 / 147.8 = 122$ sequences fit with zero context each. No amount
of paging raises that.

**At realistic batch sizes the fixed term is half the budget.** Sixty-four
sequences hold $64 \times 147.8\ \text{MiB} = 9.24\ \text{GiB}$ of recurrent
state — 52% of the pool — leaving the other half for all their KV.

**Short requests are the expensive ones, per token.** A 100-token conversation
pays 147.8 MiB of state to store 6.25 MiB of KV. A 32k conversation pays the same
147.8 MiB against 2 GiB. This is the flip side of chapter 2's break-even
calculation: the hybrid architecture is a bet on long contexts, and paging makes
the bet look better only on the part of the cost that scales.

The practical consequence for the allocator is that admission control has to
charge both. The lab's pool tracks blocks only, which is the right simplification
for learning the mechanism; the engine's `HybridCache` allocates the recurrent
state per sequence at admission, and a production scheduler must count it against
the same budget.

## Preemption: recompute or swap

When the pool is empty and a running sequence needs a block, something has to
give. Two options, and the arithmetic is more interesting than the usual advice
suggests.

*Swap.* Copy the victim's blocks to host memory, free them, copy them back on
readmission. Preserves the work, costs PCIe bandwidth, which is roughly 60 times
slower than HBM — call it 21 GB/s against the measured 1275 GB/s.

*Recompute.* Drop the blocks, and prefill the sequence again when it is
readmitted. Wastes the work, costs no transfer.

Price both for a sequence holding 2000 tokens of context on this model.

Swap moves the KV and, for the hybrid, the recurrent state too:

$$
2000 \times 64\ \text{KiB} + 147.8\ \text{MiB} = 273\ \text{MiB},
$$

which at 21 GB/s is about 13 ms out and 13 ms back.

Recompute redoes the prefill:

$$
2 \times 26.9 \times 10^9 \times 2000 = 107.6\ \text{TFLOP},
$$

which is 345 ms at 312 TFLOP/s, and considerably more in practice.

By that arithmetic swapping wins by an order of magnitude, and the textbook
answer — recompute, always — is not obviously right on this hardware and this
model. What the arithmetic leaves out is why the reference still recomputes:

- The blocks are scattered. A swap is a gather plus a transfer, and many small
  transfers get nowhere near 21 GB/s.
- The swap-in sits on the critical path. The sequence cannot run until the last
  byte lands, whereas a recomputed prefill splits into chunks that ride along
  with other work, using tensor cores that decode steps leave idle anyway.
- Swapping needs pinned host memory proportional to the preempted set, and
  pinning is a global resource that competes with everything else on the host.
- Recompute is a few lines. Swap is a subsystem.

`Scheduler._preempt` recomputes, and evicts the most recently admitted sequence
so that the sequences closest to finishing get to finish. One honest caveat about
the reference: it also clears the victim's generated tokens, so the sequence
restarts from the prompt. A production engine keeps those token ids and
recomputes their KV as part of the prefill; otherwise a user watching a stream
would see their output rewind.

## What goes wrong

**Off-by-one in `blocks_needed`.** Requesting exactly `block_size` tokens must
take one block, and `block_size + 1` must take two. An implementation that
computes `(tokens // block_size) + 1` allocates a spare block for every sequence
whose length is an exact multiple — 6% extra memory, and no test fails. The lab
checks both boundaries.

**Forgetting that `length` accumulates.** `blocks_needed` compares against the
sequence's *current* length, not against `extra_tokens` alone. An implementation
that ignores the existing length reallocates from scratch on every decode step
and exhausts the pool in seconds.

**Releasing a sequence twice.** `self.free.extend(table.blocks)` run twice puts
duplicate ids in the free list, and two sequences then write to the same block.
The symptom is one conversation's tokens appearing in another's output, which
looks like a sampling bug and is not. Popping the table before extending, as the
reference does, makes the second release a no-op.

**Assuming block order.** The free list is LIFO, so a sequence's physical block
ids descend and skip around. Any code that assumes `table.blocks` is sorted, or
that block $n+1$ follows block $n$ in memory, is wrong the first time a sequence
is released and its blocks are reused.

**A stale `context_len`.** The kernel reads `context_len` to bound the loop and
build the mask. Passing the length before this step's token was written drops the
newest key; passing it after, when the write has not happened yet, reads
uninitialized memory. The symptom is a model that generates fluently and ignores
its most recent token.

## Check your understanding

**Why does paging eliminate external fragmentation rather than just reduce it?**

Because every request is for exactly one block, and every free block satisfies it.
External fragmentation is the mismatch between the shape of a request and the
shape of the free space, and fixed-size allocation removes the mismatch by
construction. What remains is internal fragmentation, bounded by `block_size - 1`
tokens per sequence.

**If a 16-token block costs 1 MiB across the layers, why is the block table only
4 bytes per entry?**

The table stores a block *id*, not a block. The id indexes a pool of at most
18,120 blocks here, which fits in far fewer than 32 bits. The 1 MiB is the data
the id points at, and the ratio between them is exactly the leverage the
indirection buys.

**Paging bounds waste at `block_size - 1` tokens per sequence. Why does the lab's
measured paged waste come out at 1.48% rather than near zero?**

Because the bound is per sequence and the mix contains short ones. Five of the
ten lengths are under 100 tokens, and a 5-token sequence occupies a 16-token
block: 69% waste on that one. The bound is tight; it is the short requests that
make the average visible. Sequences past a few thousand tokens waste under 0.4%.

**Would a block size of 1 be strictly better for memory?**

For memory, yes — zero internal fragmentation. For everything else, no. The block
table grows to one entry per token, 128 KiB per 32k sequence, and the kernel does
one dependent load per token instead of one per sixteen, with no contiguous run
to amortize the address arithmetic over. At that point you have rebuilt a
pointer-chasing linked list on the hottest read in the engine.

## Lab

Implement `PagedCache` with `blocks_needed`, `allocate`, `release`,
`slot_indices`, `write`, and `gather`; a `paged_decode_attention(cache, seq_ids,
q)` that attends each sequence's query against its own paged prefix; and a
`memory_waste(lengths, max_seq_len, block_size)` that returns the contiguous and
paged waste fractions.

The harness checks that a fresh pool has every block free, that 20 tokens need 2
blocks of 16 and exactly 16 need 1, that growing inside the tail block takes no
new block while crossing a boundary does, that a write-then-gather round trip is
bit-exact, that a second sequence leaves the first intact, that release returns
precisely the blocks taken, and that exhausting a small pool raises. Then it
compares paged decode attention against attention over the gathered prefix for
three sequences of 37, 5, and 64 tokens, requiring agreement to $10^{-4}$ and an
output of shape `(3, heads, head_dim)`. Finally it checks that the contiguous
waste on the ten-length mix exceeds 95% while the paged waste stays under 5%.

## Further reading

- [Efficient memory management for large language model serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM: easy, fast, and cheap LLM serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [SGLang: efficient execution of structured language model programs](https://arxiv.org/abs/2312.07104) — RadixAttention, which generalizes prefix sharing to a tree.
