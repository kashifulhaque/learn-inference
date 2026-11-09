---
title: Split-KV decode attention
slug: 14a-split-kv-decode-attention
part: "Part 4 — Kernels"
summary: Why a query of length 1 leaves most of the SMs idle, how partial softmax results merge, and how many KV splits that grid launches.
minutes: 100
gpu: true
objectives:
  - "Compute the SM occupancy of a query-length-1 attention grid, and the bandwidth that occupancy implies."
  - "Derive the log-sum-exp merge of partial softmax results, and show why that merge is associative."
  - "Choose the split count from batch, heads, KV length, and SM count, including the cost of the reduction."
  - "Say what split-KV changes in a paged decode kernel."
  - "Say what FlashAttention-2 changed versus FlashAttention-1, and what FlashAttention-3 needs from Hopper that Ampere does not have."
lab: 14a-split-kv-decode
---

# Split-KV decode attention

> [!TLDR]
> - A decode step has one query token. [Chapter 14](/c/14-flash-attention)'s grid is then one program per query head, so batch 1 on this model launches 24 programs. An A100 has 108 SMs, and 84 of them get no block.
> - If those 24 SMs share the measured 1275 GB/s copy ceiling and the idle SMs contribute none, the ceiling is 283.3 GB/s. Four KV splits raise the grid to 96 programs and the same ceiling to 1133.3 GB/s. Neither number is a measured kernel.
> - Each split returns a normalized partial output and a log-sum-exp. The merge is a weighted average. The weights are the chunks' shares of the softmax mass, and in exact arithmetic the merge is associative.
> - The split count stays 1 once `batch * heads` covers 80% of the SMs. Otherwise it is the smallest count whose wave efficiency is within 85% of the highest, capped by the number of 64-token KV blocks.
> - FlashAttention-2 parallelizes over the query length and splits queries across warps. FlashAttention-3 needs Hopper's TMA, asynchronous WGMMA, and FP8 tensor cores, which Ampere does not have.

[Chapter 14](/c/14-flash-attention) removes the score matrix. The grid it launches is one program per query tile and per head. During prefill that grid is large. During decode the query length is 1, so the tile axis contributes one block, and the whole launch is `batch * heads` programs.

On this model that is 24 programs at batch 1. The A100 has 108 streaming multiprocessors. Most of them sit idle while one program walks the whole KV cache. The fix is to split the key and value sequence as well, run those pieces in parallel, and merge them with the log-sum-exp the online softmax already computes.

The kernel that tiles Q, K, and V is chapter 14, and the same forward and backward kernels are the subject of the [GPU course's FlashAttention chapter](https://gpu.ifkash.dev/c/16-flash-attention). This chapter does not write that tile loop again. It counts the idle SMs, merges the partial results, and says what the split changes in the paged decode kernel.

## Before you start

**The decode grid, from [chapter 14](/c/14-flash-attention).** One program owns one query tile. The launch is `(ceil(q_len / block_m), batch * heads)`. At `q_len = 1` the first factor is 1, for any tile height of at least 1. This model has 24 query heads, so batch 1 launches 24 programs. The 4 KV heads and the head dimension 256 are the same geometry as [chapter 7](/c/07-grouped-query-attention).

**Online softmax.** A running row keeps a maximum $m$, a sum of exponentials $\ell$, and an accumulator. The log-sum-exp of that row is $m + \log \ell$. Chapter 14 divides the accumulator by $\ell$ once, at the end of the row. A split does that division at the end of its own chunk, and keeps the log-sum-exp so a later merge can put the chunks back together.

**SMs and bandwidth, from [chapter 10](/c/10-roofline).** The A100 has 108 SMs. A copy on this card measures 1275 GB/s. The rated 1935 GB/s is not the ceiling you compare a kernel against. Chapter 10's utilization for $b$ blocks is $b / (108 \cdot \lceil b / 108 \rceil)$.

**The block table, from [chapter 15](/c/15-paged-attention).** Decode attention reads keys and values through a block table. Logical block $b$ holds tokens `[b * block_size, (b + 1) * block_size)`. The kernel loads `block_table[seq, b]` and builds the address from that physical block. This chapter changes which logical blocks a program visits. It does not rebuild the allocator.

## A query of length 1 leaves most SMs idle

This section counts the blocks a decode attention kernel launches, and the bandwidth those blocks can use.

Flash-Decoding states the same gap for a grid that does not multiply by the head count. If the batch is smaller than the SM count, much of the GPU is idle, and a batch of 1 uses less than 1% of an A100's 108 SMs. That sentence is about the batch axis alone. This course's grid also has a head axis, so the same step is larger than one block and still far from 108.

Batch 1 and 24 query heads launch 24 blocks. One wave of 108 has room for all of them, so the utilization from chapter 10 is the following:

$$
u(24) = \frac{24}{108} = 0.222.
$$

==Eighty-four SMs receive no block.== The kernel is still memory bound. Chapter 10's decode-attention intensity does not grow with context. What grows is the KV read, and only the SMs that have a program issue that read.

Take the measured copy, 1275 GB/s, and give each occupied SM an equal share. Idle SMs contribute none. The ceiling this grid can touch is the following:

$$
1275 \times \frac{24}{108} = 283.\overline{3}\ \text{GB/s}.
$$

That is an assumption about occupancy, not a profile. A few SMs can sometimes feed the memory controllers well enough to beat a strict share, and a real kernel can also miss it. The number is the roofline you get if bandwidth scales with the fraction of SMs that have work. `scripts/figures/14a-split-kv-decode-attention.py` prints it.

One full-attention layer of this model stores $2 \times 4 \times 256 \times 2 = 4096$ bytes of KV per token. At 32,768 tokens that is 134,217,728 bytes. The time to move those bytes at the two ceilings is the following:

$$
\frac{134{,}217{,}728}{1275 \times 10^{9}} = 105.27\ \mu\text{s},
\qquad
\frac{134{,}217{,}728}{283.\overline{3} \times 10^{9}} = 473.71\ \mu\text{s}.
$$

The byte count did not change. The occupancy ceiling is 4.5 times slower because 108/24 = 4.5. Splitting the KV length is how you buy the missing programs without reading the cache twice. Each split reads a disjoint slice. The union is still one pass over K and V.

> [!EXAMPLE] Four splits on this grid
> Four splits launch $24 \times 4 = 96$ programs. Utilization is $96/108 = 0.889$, and the same copy-share ceiling is $1275 \times 96/108 = 1133.\overline{3}$ GB/s. The 32,768-token read then takes 118.43 µs instead of 473.71 µs. Nine splits would fill two exact waves, 216 programs, and the ceiling would be the full 1275 GB/s. The next section stops at four anyway.

Flash-Decoding's own measurements are on CodeLlama-34B, not on this model and not on this machine. On that benchmark the end-to-end generation speedup is up to 8× at long context, the attention kernel itself is up to 50× faster than FlashAttention, and the attention time stays roughly flat out to 32k tokens because the split grid keeps the GPU busy. Treat those as their results. The arithmetic you can recompute here is the 24/108 ceiling.

## The log-sum-exp merge

This section combines partial softmax outputs, and shows why the order of the chunks does not matter.

Split $i$ attends the query against its own keys and values and returns two things. $\hlb{O_i}$ is the softmax-weighted values inside the chunk, already divided by that chunk's sum of exponentials. $\hla{L_i}$ is the chunk's log-sum-exp, $m_i + \log \ell_i$. The merge treats those pairs as a partition of one row.

Add the chunks' probability mass in log space, then weight each partial output by its share of that mass:

$$
\boxed{
\hla{L} = \log \sum_i \exp(\hla{L_i}),
\qquad
\hlb{O} = \sum_i \exp(\hla{L_i} - \hla{L})\, \hlb{O_i}
}
$$

> [!INTUITION] A weighted average, not a mean
> $\exp(\hla{L_i} - \hla{L})$ is the fraction of the row's softmax mass that landed in chunk $i$. A long chunk with large scores gets a large weight. A short chunk with small scores gets a small one. Dividing the partial outputs by the split count throws that fraction away.

An empty chunk stores $\hla{L_i} = -\infty$ and a zero output. $\exp(-\infty)$ is 0, so the chunk adds nothing. Do not evaluate $\exp(-\infty - (-\infty))$, which is not a number. A chunk stored with log-sum-exp 0 is not empty. It is a real vote with weight $\exp(0 - \hla{L})$.

> [!EXAMPLE] Two chunks and one empty chunk
> Scores $0$ and $2$ with values $4$ and $6$ give partial output $5.761594$ and log-sum-exp $2.126928$. Score $1$ with value $10$ gives partial output $10$ and log-sum-exp $1$. An empty chunk has log-sum-exp $-\infty$. The merge returns output $6.798853$ and log-sum-exp $2.407606$, the same pair as a softmax over $[0, 2, 1]$ and values $[4, 6, 10]$. The unscaled mean of the two real partials is $(5.761594 + 10) / 2 = 7.880797$.

The merge is associative in exact arithmetic. Write the unnormalized mass and the unnormalized values of a chunk:

$$
Z_i = \exp(\hla{L_i}), \qquad U_i = \exp(\hla{L_i})\, \hlb{O_i}.
$$

Disjoint chunks add. $Z_{I \cup J} = Z_I + Z_J$ and $U_{I \cup J} = U_I + U_J$. Addition of real numbers does not depend on the grouping, so $(I \cup J) \cup K$ and $I \cup (J \cup K)$ produce the same $Z$ and the same $U$. The merged output is $U / Z$ and the merged log-sum-exp is $\log Z$. Floating-point addition is not that associative. The lab's tolerance is the gap, the same kind of gap chapter 14 already accepts between an online softmax and a one-pass softmax.

> [!DEEPDIVE] Why two groupings match
> Expand three chunks. The left grouping first forms $Z_{ab} = Z_a + Z_b$ and $U_{ab} = U_a + U_b$, then adds chunk $c$. The right grouping first forms $Z_{bc} = Z_b + Z_c$. Both finish at $Z_a + Z_b + Z_c$ and $U_a + U_b + U_c$. An empty chunk is the pair $(Z, U) = (0, 0)$, which is the additive identity, so it drops out of either grouping. FlashInfer writes the same operator on the pair $(\hlb{O}, \hla{L})$ and uses it to compose attention over arbitrary index sets, not only over contiguous KV slices.

## Choose the split count

This section picks a split count from the batch, the head count, the KV length, and the SM count, and prices the reduction that follows.

More splits fill more SMs. They also write one partial output and one log-sum-exp per query head, and a second kernel reads those partials back. FlashAttention's `num_splits_heuristic` balances the two. The launcher in that file passes twice the SM count, because it assumes two 128-thread blocks can sit on one SM. This chapter counts one block per SM, the same 108 lanes as chapter 10, and keeps the two fractions from the heuristic.

Define *wave efficiency* for `tasks = batch * heads` and a candidate split count $s$. Decode has one query tile, so `tasks` is the whole unsplit grid. The launched blocks are `tasks * s`. The wave count is $\lceil \text{tasks} \cdot s / \text{num\_sms} \rceil$, and the efficiency is the blocks divided by the slots in those waves:

$$
e(s) = \frac{\text{tasks} \cdot s}{\text{num\_sms} \cdot \lceil \text{tasks} \cdot s / \text{num\_sms} \rceil}.
$$

`choose_num_splits(batch, heads, kv_len, num_sms)` then follows four rules:

1. If any argument is less than 1, return 1.
2. If `batch * heads * 5 >= num_sms * 4`, return 1. That is the 80% test. It includes every case where `batch * heads` is at least `num_sms`.
3. Otherwise search $s$ from 1 to $\min(128, \text{num\_sms})$. Let $e^{\star}$ be the highest efficiency in that range. Take the smallest $s$ with $e(s) \ge 0.85\, e^{\star}$.
4. Cap that choice by the number of KV blocks, $\lceil \text{kv\_len} / 64 \rceil$. The block is 64 tokens because that is the split-KV tile FlashAttention uses once the head dimension is greater than 128, which is this model's 256.

The cap is what makes the count monotone in KV length. A longer cache has at least as many 64-token blocks, so the cap can rise and cannot fall. The 80% test and the 85% search do not look at `kv_len` at all.

> [!KEY] Stop at four splits for this decode grid
> For 24 heads and 108 SMs, one split has efficiency $24/108 = 0.222$. Four splits have $96/108 = 0.889$. Nine splits land on two full waves and efficiency 1. The cutoff is $0.85 \times 1 = 0.85$, and 0.889 clears it, so the search returns 4 rather than 9. Three splits sit at $72/108 = 0.667$ and do not clear it.

The reduction those four splits write is small next to a long cache. Each partial is a float32 vector of width 256 plus one float32 log-sum-exp, 1,028 bytes. Across 24 heads and 4 splits that is 98,688 bytes. The 32,768-token KV read is 134,217,728 bytes, so the partials are $7.35 \times 10^{-4}$ of it. The byte cost is not why the heuristic refuses nine splits. It refuses them because a second wave, and a second kernel, buy efficiency the 85% rule already treats as close enough. Chapter 10's 5 µs launch is real, and it is small beside the 118.43 µs memory ceiling, which is another way of saying the same thing. Occupancy sets the count. The reduction rides along.

The figure is that table for batch 1. The script prints every point.

```viz
14a-split-count
```

The following anchors all use 108 SMs:

| Batch | Heads | KV length | Splits | Why |
|---|---|---|---|---|
| 1 | 24 | 64 | 1 | One 64-token block, so the cap is 1 |
| 1 | 24 | 65 | 2 | Two blocks, and two splits still help occupancy |
| 1 | 24 | 256 | 4 | The cap has caught the occupancy choice |
| 1 | 24 | 65536 | 4 | Same choice. The cache is longer, the count is not |
| 1 | 108 | 65536 | 1 | `108 * 5 >= 108 * 4`, so the grid is already full |
| 1 | 87 | 65536 | 1 | 87 heads clear the 80% test. 86 heads do not, and the search returns 5 |
| 1 | 1 | 65536 | 92 | One head needs many splits before the waves fill |

FlashAttention's source also skips a candidate when it does not shorten the KV blocks relative to the previous candidate. That filter can step the count down by one when the length grows by a single block. This chapter drops it. The lab checks that a longer cache does not return a smaller count.

Variable-length batches need a different cut. Equal splits on a short sequence finish early and leave the long sequence's tail as the long pole. FlashInfer's scheduler plans on the CPU. It sets a KV chunk size from the total work divided by the CTA count, assigns long chunks first to the least-loaded CTA, and then `run`s a persistent kernel whose grid does not change. The same plan is reused for every layer of the step. The composition of those chunks is the merge in the preceding section. When the KV is paged, FlashInfer does not use TMA for the gather, because TMA does not support non-affine addresses. That limitation is the Hopper story in the last section.

## What split-KV changes in the paged kernel

This section says what changes in [chapter 15](/c/15-paged-attention)'s decode kernel, and what stays a block-table load.

Chapter 15 launches one program per sequence and per query head, and that program walks every logical block. Split-KV adds a third program id and shortens the walk. The block-table load is the same line. The range around it is not.

The following is the difference, not a second copy of the kernel:

```python
split = tl.program_id(2)
start = split * blocks_per_split
end = min(start + blocks_per_split, num_blocks)

for b in range(MAX_BLOCKS):
    if start <= b < end:
        physical = tl.load(block_table + seq * stride + b)
        # the same key and value loads, and the same online softmax
```

Three consequences follow from that range:

- The program writes a partial output and a log-sum-exp at index `split`, not the final attention output. A second kernel, or `merge_partials` on the workspace, produces the output chapter 15 stores.
- A split past the last logical block writes zeros and a log-sum-exp of $-\infty$. The merge then ignores it. Storing 0 instead of $-\infty$ invents a chunk.
- The logical blocks in one split stay contiguous, so the online softmax inside the split is the chapter 14 update. The merge does not require contiguity. Any partition of the keys works in exact arithmetic. Contiguity is what lets the inner loop stay the loop you already have.

[Chapter 10a](/c/10a-cuda-graphs-and-launch-overhead) captures a decode step only when the addresses and shapes stay fixed. The partial workspace is one of those shapes. Size it for the maximum split count you will launch, and let unused splits carry $-\infty$. The block table's contents can change between replays. Its address cannot.

## What FlashAttention-2 changed, and what FlashAttention-3 asks of Hopper

This section names the work-partitioning changes. The tile kernel those changes sit on is chapter 14 and the GPU course, not this section.

FlashAttention-1 gives one thread block to one head. The grid is `batch * heads` blocks, and FlashAttention-2 says that schedule fills the GPU when the product is at least about 80, against 108 SMs on an A100. A long sequence usually means a small batch, so that grid leaves SMs idle. The inner loop of that kernel also runs over query tiles while the outer loop runs over key tiles, and the partial output is written back to HBM on every outer step.

FlashAttention-2 makes three changes:

- **Parallelize the query length.** The outer loop over query-row blocks becomes the grid. Blocks in that grid do not communicate. A long prefill fills the SMs even when the batch and the head count do not. Decode undoes that axis, because the query length is 1. Split-KV puts the parallelism back, along K and V instead of along Q.
- **Spend fewer non-matmul FLOPs.** The A100's FP16 and BF16 matmul peak is 312 TFLOP/s. Its non-matmul FP32 peak is 19.5 TFLOP/s, so one non-matmul FLOP costs 16 matmul FLOPs. FlashAttention-2 keeps an unscaled accumulator and divides by $\ell$ once at the end of the row, which is the update chapter 14 already writes. It also stores the log-sum-exp for the backward pass instead of storing both $m$ and $\ell$.
- **Split Q across warps, not K and V.** FlashAttention-1's split-K layout makes every warp write a partial into shared memory and then reduce. FlashAttention-2 gives each warp a slice of Q and the whole K and V tile, so the warp's output slice needs no cross-warp add.

The paper records that the loop swap and the sequence-parallel grid first showed up in Triton's fused attention tutorial. Chapter 14 uses that order. The [GPU course's FlashAttention chapter](https://gpu.ifkash.dev/c/16-flash-attention) is where the forward and backward kernels are the assignment. FlashAttention-2's reported speedups, about 2× over FlashAttention and 50% to 73% of peak FLOP/s on A100, are that paper's measurements.

FlashAttention-3 starts from a different complaint. FlashAttention-2's algorithm is synchronous, and on an H100 the paper measures it at 35% utilization. The three techniques that raise that number need Hopper hardware Ampere does not have:

- **TMA.** Hopper's Tensor Memory Accelerator copies tiles between global memory and shared memory as its own unit. Ampere's asynchronous copies are a thread instruction, the `cp.async` path the [GPU course's pipelining chapter](https://gpu.ifkash.dev/c/12-pipelining-and-async-copies) writes. TMA also refuses non-affine addresses, which is why a paged KV load stays on the Ampere-style copy.
- **Asynchronous WGMMA.** Hopper's tensor-core instruction is issued by a warpgroup, reads operands from shared memory, and runs beside other work. Ampere's tensor cores are the synchronous `mma` instructions. FlashAttention-3 overlaps softmax on one score tile with WGMMA on the next tile. That overlap needs the asynchronous instruction. Producer warps issue TMA, and consumer warps issue WGMMA.
- **FP8 tensor cores.** Hopper's WGMMA has an FP8 path. FlashAttention-3 uses it with block quantization and incoherent processing. This course's A100 has no FP8 tensor cores, so that path stays a description.

The paper's speedups, 1.5× to 2× over FlashAttention-2 and up to 740 TFLOP/s in FP16, are H100 SXM5 results. This machine has no Hopper GPU, and the lab does not pretend to time one.

## What goes wrong

- **Averaging the partial outputs.** The unscaled mean in the worked example is 7.880797 against a softmax of 6.798853. The chunks do not hold equal probability mass. The lab fails that merge.
- **Storing an empty split as log-sum-exp 0.** Zero is a finite score. The merge then gives that split weight $\exp(0 - L)$ and pulls the output toward the zero vector.
- **Splitting a grid that is already full.** `batch * heads >= num_sms` returns 1. Extra splits add a workspace and a second kernel, and they do not wake an idle SM.
- **Letting the split count fall as the cache grows.** A filter that depends on the exact block count can do that. The cap by $\lceil \text{kv\_len} / 64 \rceil$ cannot. The monotone check is that cap.
- **Writing the final output from inside the split.** Each program has seen only its slice. The store belongs to the merge.

> [!RECAP]
> - Batch 1 and 24 heads launch 24 of 108 programs. The copy-share ceiling is 283.3 GB/s, against 1275 GB/s when every SM has work.
> - The merge is $\hlb{O} = \sum_i \exp(\hla{L_i} - \hla{L})\, \hlb{O_i}$ with $\hla{L} = \log \sum_i \exp(\hla{L_i})$. In exact arithmetic it is associative, because the unnormalized sums add.
> - An empty chunk is log-sum-exp $-\infty$. It adds the zero vector for $(Z, U)$.
> - For 24 heads and 108 SMs the heuristic returns 4 once the cache holds 256 tokens, and it returns 1 when `batch * heads` covers 80% of the SMs.
> - Four splits write 98,688 bytes of partials, $7.35 \times 10^{-4}$ of a 32,768-token KV read. Occupancy, not those bytes, sets the count.
> - FlashAttention-2 parallelizes query tiles and splits Q across warps. FlashAttention-3 needs TMA, asynchronous WGMMA, and FP8 tensor cores.

## Check your understanding

> [!QUESTION] Batch 1 launches 24 programs. Why is the copy-share ceiling 283.3 GB/s and not 1275?
> Because 24 of 108 SMs have a block. Giving those SMs an equal share of the measured 1275 GB/s copy, and the idle SMs none, is $1275 \times 24/108 = 283.\overline{3}$. A profile can land elsewhere. This is the occupancy ceiling, not a timed kernel.

> [!QUESTION] Two chunks return outputs 5.761594 and 10. Why is the merged output 6.798853 rather than 7.880797?
> The first chunk's log-sum-exp is 2.126928 and the second's is 1, so the first chunk holds more of the softmax mass. The merge weights each output by $\exp(L_i - L)$. The mean 7.880797 weights them equally.

> [!QUESTION] `choose_num_splits(1, 108, 65536, 108)` returns 1. Why does a 65,536-token cache not change that?
> `1 * 108` already covers the SMs, and the 80% test returns 1 before the KV length is consulted. Extra splits would add partials and a reduction, and every SM already has a program.

> [!QUESTION] Which three Hopper features does FlashAttention-3 use that an A100 does not have?
> TMA for tile copies, asynchronous WGMMA so softmax can overlap the next matmul, and FP8 tensor cores. Ampere has synchronous tensor-core instructions and `cp.async`, and it has no FP8 tensor cores. Paged addresses still use the Ampere-style copy, because TMA does not gather non-affine addresses.

## Lab

> [!TRY]
> Implement the merge and the split count. You pass when `merge_partials` matches a full softmax, including unequal chunks and a chunk whose log-sum-exp is $-\infty$, and when `choose_num_splits` is monotone in KV length, returns 1 once `batch * heads` covers the SMs, and matches the table in the split-count section. On a CUDA device, `split_kv_decode` also matches scaled dot-product attention. Without one, that check skips.

The CPU checks run anywhere. Implement the following:

- `merge_partials(outs, lses)`, with `outs` shaped `(splits, ..., dim)` and `lses` shaped `(splits, ...)`. Return the merged output and the merged log-sum-exp. Weight each partial by $\exp(L_i - L)$. An empty split is $-\infty$ and adds nothing.
- `choose_num_splits(batch, heads, kv_len, num_sms)`, the four rules in the split-count section. The KV block is 64 tokens. The search stops at $\min(128, \text{num\_sms})$.

On CUDA, also implement the following:

- `split_kv_decode(q, k, v, num_splits)`, with `q` shaped `(batch, heads, dim)` and `k`, `v` shaped `(batch, heads, kv_len, dim)`. One KV head per query head. Write a partial output and a log-sum-exp per split, then call `merge_partials`. The check compares the result with scaled dot-product attention at lengths 1, 17, 128, 4,096, and 65,536. It does not time the kernel.

The first check calls only `merge_partials`. A merge that adds the partial outputs without the log-sum-exp weights fails the unequal-chunk check.

## Further reading

- [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html). Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov, Stanford CRFM, 12 October 2023. Split the KV sequence, compute a partial output and a log-sum-exp per split, and reduce. The batch-of-1 occupancy claim, the CodeLlama-34B speedups, and the three-step procedure in this chapter are from that post. The same essay is on the [PyTorch blog](https://pytorch.org/blog/flash-decoding/).
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691). Query-length parallelism, the loop swap, the non-matmul FLOP cut, and splitting Q across warps rather than K and V. The 312 TFLOP/s and 19.5 TFLOP/s figures, and the 30% to 50% forward-pass utilization of FlashAttention-1, are that paper's A100 account.
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). Warp specialization, softmax overlapped with asynchronous WGMMA, and FP8 with block quantization and incoherent processing. The 35% H100 utilization and the hardware the algorithm requires are from this paper.
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005). The attention state $(O, \mathrm{LSE})$ composes by the merge in this chapter. `plan` builds a load-balanced chunk assignment, and `run` executes it on a fixed grid so a CUDA graph can capture the step. TMA is limited to affine KV addresses.
- [`num_splits_heuristic` in FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/csrc/flash_attn/flash_api.cpp). The 80% early return, the 85% efficiency cutoff, the 128-split cap, and the 64-token tile at head dimension above 128. This chapter counts one block per SM instead of the launcher's `multiProcessorCount * 2`.
- [FlashAttention in the GPU course](https://gpu.ifkash.dev/c/16-flash-attention), and [pipelining and asynchronous copies](https://gpu.ifkash.dev/c/12-pipelining-and-async-copies). The tiled kernel and the Ampere `cp.async` path. Hopper TMA and WGMMA are described there and used conceptually here.
