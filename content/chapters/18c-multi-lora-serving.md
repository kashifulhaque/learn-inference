---
title: Multi-LoRA serving
slug: 18c-multi-lora-serving
part: "Part 6 — Scaling"
summary: Serve many LoRA adapters from one base model in one batch with a gathered and a segmented kernel, count what the adapters cost, and keep them resident with a frequency-aware pool.
minutes: 70
gpu: false
objectives:
  - "Write the LoRA update with the alpha over r scale and show why a merged weight serves only one adapter."
  - "Write bgmv, a gathered per-row adapter product, and match a per-request loop."
  - "Write sgmv, which sorts rows by adapter and runs one pair of matmuls per segment."
  - "Count the FLOPs and bytes of the LoRA path against the base layer at ranks 8, 16, and 64."
  - "Size an adapter pool, price a host-to-GPU load at 32 GB/s, and grade an eviction policy on a seeded trace."
lab: 18c-multi-lora
---

# Multi-LoRA serving

> [!TLDR]
> - A LoRA layer adds a rank-$r$ update to a frozen layer: $y = xW + (xA)B \cdot \alpha/r$. Merging $AB$ into $W$ removes the extra work, but the merged weight belongs to one adapter.
> - A batch that mixes adapters cannot use a merged weight. `bgmv` gathers each row's adapter. `sgmv` sorts rows by adapter and runs one pair of matmuls per segment. Both equal a per-request loop.
> - On the up projection, a rank-16 path does $720{,}896$ FLOPs against $178{,}257{,}920$ for the base layer, $0.40\%$. The cost that matters is memory: the adapter must be resident.
> - A rank-16 adapter on every MLP matrix of all 64 layers is $138{,}412{,}032$ bytes. A host-to-GPU copy at 32 GB/s takes $4.325$ ms, about a tenth of a 42 ms decode step.
> - On the lab's seeded trace, a pool that evicts by lifetime frequency and pins the running batch spends $0.805$ of the PCIe time of the better of FIFO and LRU.

A fine-tuned model is often a small difference from a shared base. A product with one fine-tune per customer ends up with hundreds of adapters on one base model. You can't give each its own replica, because the base weights are 53.8 GB and the adapters are tens or hundreds of megabytes.

This chapter serves them together. The base weights stay once on the GPU. The adapters are small enough that the hard problems are a kernel that handles a batch where every row has a different adapter, and a memory policy for the adapters that don't fit. It uses [chapter 13a](/c/13a-skinny-decode-gemm)'s skinny products, and the eviction ideas from [chapter 15a](/c/15a-prefix-caching-and-radix-trees) and [chapter 16](/c/16-continuous-batching).

## Before you start

**The decode GEMM, from [chapter 13a](/c/13a-skinny-decode-gemm).** A linear layer with $M$ rows is memory bound until $M$ reaches the ridge, because the weight is read once for the whole step. The up projection has $K = 5120$ and $N = 17408$, which is $89{,}128{,}960$ weights.

**The model, from [chapter 02](/c/02-memory-arithmetic).** Hidden size $5120$, MLP width $17408$, $64$ layers, bfloat16. The MLP has gate, up, and down matrices, and the three hold $267{,}386{,}880$ weights per layer. This chapter puts adapters on the MLP only, to keep the arithmetic to three shapes.

**Eviction, from [chapter 15a](/c/15a-prefix-caching-and-radix-trees).** A cache can't evict an entry that a running request still uses. The reference count that protects a block there protects an adapter here.

**PCIe, from [chapter 17b](/c/17b-disaggregated-serving).** PCIe 4.0 x16 moves 32 GB/s one way. A decode step on this model has a floor of about 42 ms, from [chapter 00](/c/00-what-you-are-building).

## The update and the merge

This section answers what a LoRA layer computes and why you can fold it into the weight.

The idea in a picture: the frozen weight $W$ is a wide matrix, and the fine-tune is a thin detour beside it. The input goes down to $r$ numbers, comes back up to the output width, and the result is added to what $W$ produced.

For a layer with $W$ of shape $(d_\text{in}, d_\text{out})$, an adapter holds $\hla{A}$ of shape $(d_\text{in}, r)$ and $\hlb{B}$ of shape $(r, d_\text{out})$. Write the update as the base product plus the detour, scaled by $\alpha / r$:

$$
\boxed{y = xW + \hlc{\tfrac{\alpha}{r}}\,(x\hla{A})\hlb{B}}
$$

The scale $\hlc{\alpha / r}$ is a constant per adapter. $\alpha$ is a training hyperparameter, and $r$ is the rank, the second dimension of $A$. Hu and colleagues introduce this scale so that changing $r$ doesn't force you to retune the learning rate.

> [!EXAMPLE] A rank-2 layer with alpha 4
> Take $x = (1, 2)$, $W = \begin{pmatrix}1&2\\3&4\end{pmatrix}$, $A = \begin{pmatrix}1&1\\0&1\end{pmatrix}$, $B = \begin{pmatrix}1&0\\2&1\end{pmatrix}$, and $\alpha = 4$. The scale is $4/2 = 2$.
> The base product is $xW = (7, 10)$. The detour is $xA = (1, 3)$, then $(xA)B = (7, 3)$, then times $2$ is $(14, 6)$. The sum is $y = (21, 16)$.
> Scaling by $\alpha = 4$ instead gives $(28, 12)$ for the detour and $y = (35, 22)$. The lab's first check uses these numbers.

Matrix multiplication is associative, and $x(W + \tfrac{\alpha}{r}AB) = xW + \tfrac{\alpha}{r}(xA)B$. So you can precompute

$$
W' = W + \tfrac{\alpha}{r}AB
$$

and serve $xW'$ with no extra FLOPs and no extra kernels. This is *merging*. The merged weight has the same shape as $W$, and the detour costs nothing at run time.

## Why merging fails for a mixed batch

A merged weight is correct for one adapter and wrong for every other, so a batch with two adapters has no single $W'$.

Suppose rows 0 and 1 of a batch use adapters 0 and 1. The merged $W'_0$ gives row 1 the update of adapter 0. You have three options:

- Run each adapter's rows as a separate batch. Each merged GEMM then has fewer rows and reads the full weight again, so you lose the sharing that makes decode cheap.
- Swap the merged weight between steps. A merge or an un-merge rewrites $89$ million weights per matrix, and a rounding error accumulates over many swaps.
- Keep $W$ unmerged and add each row's detour. The base GEMM stays one batched product, and the detour is a small extra.

Serving systems take the third option when the batch is heterogeneous. Merged serving still wins when all traffic uses one adapter, and the dLoRA system (Wu and colleagues, OSDI 2024) merges and unmerges adapters as the workload changes, and finds that unmerged serving isn't always better when request types are skewed. This chapter builds the unmerged path.

> [!KEY] The base weight is shared and the detour is per row
> $xW$ is one GEMM for the whole batch. Only $(x_iA_{a(i)})B_{a(i)}$ depends on which adapter $a(i)$ row $i$ uses.

## Gather the adapters with bgmv

The detour needs a different $A$ and $B$ for each row. `bgmv`, a batched gather matrix-vector product, does it with an index. The Punica paper names it and its segmented sibling.

Stack the adapters into $A_\text{stack}$ of shape $(n, d_\text{in}, r)$ and $B_\text{stack}$ of shape $(n, r, d_\text{out})$. Row $i$ carries a request's adapter id $a(i)$. The delta for row $i$ is

$$
\Delta_i = \tfrac{\alpha}{r}\,\bigl(x_i A_\text{stack}[a(i)]\bigr)\,B_\text{stack}[a(i)],
$$

and the layer output is $y_i = x_iW + \Delta_i$. An id of $-1$ marks a row with no adapter, and its delta is zero. In PyTorch the gather is `A_stack[adapter_ids]`, which has shape $(T, d_\text{in}, r)$, followed by two batched matrix products.

==The index is the request's adapter id, not the row's position in the batch.== A tempting version indexes with `arange(T)`. It passes any test where row $i$ happens to use adapter $i$, and it fails when ids repeat or come in another order. The lab permutes the ids and also uses more rows than adapters.

The gather copies $T$ adapters into a temporary, which is slow. Real kernels read the adapter through a pointer and never copy it, but the arithmetic is the same.

## Sort the rows with sgmv

`sgmv` does the same product with fewer reads: it groups rows that share an adapter, so each adapter's weights are loaded once.

Sort the rows by adapter id. Rows with equal ids become one contiguous *segment*. For each segment, run $X_s A_a$ and then $(X_s A_a) B_a$ as two small matmuls, where $X_s$ is the segment's rows. Then write each row back to its original position.

A stable sort keeps requests in arrival order inside a segment, and the inverse permutation restores the batch order. The ids $[2, 0, -1, 2, 1, 0, 2]$ sort to the order $[2, 1, 5, 4, 0, 3, 6]$. The segments are adapter 0 on sorted positions 1 to 3, adapter 1 on 3 to 4, and adapter 2 on 4 to 7. The $-1$ row sorts first and belongs to no segment.

The benefit is reuse. A segment of $s$ rows reads the adapter once and does $s$ rows of work, so its intensity rises with $s$. `bgmv` reads one adapter per row. The Punica paper names both forms, a gathered one and a segmented one, and builds its kernels around them.

## What the LoRA path costs

The detour is tiny in FLOPs and has the same arithmetic intensity as the weight GEMM it sits beside, so it is cheap only when the adapter is already in GPU memory.

For one token on a layer of shape $(d_\text{in}, d_\text{out})$ at rank $r$, the detour is two skinny products, $d_\text{in} r$ and $r d_\text{out}$ multiply-adds:

$$
F_\text{lora} = 2r(d_\text{in} + d_\text{out}), \qquad F_\text{base} = 2 d_\text{in} d_\text{out}.
$$

The adapter weights are $2r(d_\text{in} + d_\text{out})$ bytes in bfloat16, the same count. So the detour does $1$ FLOP per byte at one token per adapter, which sits near [chapter 13a](/c/13a-skinny-decode-gemm)'s $0.9997$ and far below the ridge of $161.2$.

| Rank | LoRA FLOPs per token | Share of the base layer | Adapter bytes on this layer |
|---|---|---|---|
| 8 | 360,448 | 0.2022% | 360,448 |
| 16 | 720,896 | 0.4044% | 720,896 |
| 64 | 2,883,584 | 1.6176% | 2,883,584 |

The table is the up projection, with $F_\text{base} = 178{,}257{,}920$. The FLOP share is small at every rank. The part that grows is the number of adapters you must hold.

Gate, up, and down in all 64 layers make one MLP adapter. Its size is $64 \times r \times 2 \times (22528 + 22528 + 22528)$ bytes:

| Rank | Adapter bytes | Over PCIe at 32 GB/s | Fit in 4 GiB | Distinct adapters that read as many bytes as the MLP |
|---|---|---|---|---|
| 8 | 69,206,016 | 2.163 ms | 62 | 494.5 |
| 16 | 138,412,032 | 4.325 ms | 31 | 247.3 |
| 64 | 553,648,128 | 17.302 ms | 7 | 61.8 |

The MLP weights are $34{,}225{,}520{,}640$ bytes. A step whose batch uses 64 distinct rank-16 adapters reads $8{,}858{,}370{,}048$ bytes of adapter, $25.9\%$ of that, and it can't share a read between rows the way the base GEMM does. About 62 distinct rank-64 adapters read as many bytes as the base MLP. These numbers are config arithmetic from `scripts/figures/18c-multi-lora-serving.py`, not an A100 measurement.

> [!NOTE] The 42 ms comparison
> A rank-16 load is $4.325 / 42 = 10.3\%$ of a decode step, and a rank-64 load is $41.2\%$. If a request waits on a cold load, its first token pays that time. The pool in the next section exists to avoid it.

## Keep the adapters resident

The GPU holds only the adapters that running requests need, and every other adapter waits in host memory, so the pool's job is to choose what to drop.

S-LoRA stores all adapters in main memory and copies the ones a batch needs into GPU memory. Two rules keep this correct:

- **Pin the running batch.** An adapter used by a request in the current step can't be evicted. A policy that ignores this evicts it mid-step and reloads it a step later.
- **Free enough bytes.** Adapters have different ranks, so they have different sizes. A load of 528 MiB can need several evictions. Evict until the new adapter fits.

Among the unpinned adapters, you choose a victim. FIFO drops the oldest load. LRU drops the least recently used. Both lose to a *scan*: a burst of requests for many rarely used adapters pushes the popular ones out, and then every popular adapter reloads. [Chapter 15a](/c/15a-prefix-caching-and-radix-trees) evicts the least recently used unreferenced leaf, which is the right bias when recent use predicts reuse. Adapter traffic is more often skewed toward a few popular adapters.

The lab's frequency policy counts how many batches have used each adapter over the whole run, and evicts the lowest count, ties by recency. The count survives an eviction, so a popular adapter that was dropped returns with its history.

```viz
18c-adapter-pool
```

The trace is Zipf traffic over 48 adapters of ranks 8, 16, and 64 (18, 19, and 11 of them), four per batch, with a scan of cold adapters in 10 of every 30 batches. At a 4 GiB pool FIFO loads 484 times for $3016.9$ ms, LRU 493 times for $3118.6$ ms, and the frequency policy 340 times for $2428.7$ ms, a ratio of $0.805$ against the better plain policy. At 8 GiB FIFO beats LRU, so no single plain policy wins everywhere. Every figure is a simulation on that seeded trace, not a production measurement.

S-LoRA goes further than a pool of adapters. Its *unified paging* stores adapter weights and KV-cache blocks in one page pool, so an idle adapter's pages can become KV pages and the reverse. A separate adapter pool, as in this chapter, can sit half empty while the KV cache runs short. The lab does not build the unified pool, because the policy question is the same in both.

## What goes wrong

- **Scaling by alpha instead of alpha over r.** The update is $r$ times too large. With $\alpha = 32$ and $r = 8$ it is 8 times too large. The output is plausible, only biased, so no shape check catches it. The worked example does: $(35, 22)$ rather than $(21, 16)$.
- **Indexing adapters by batch position.** It works when row $i$ uses adapter $i$. Any other order, or more rows than adapters, gives wrong rows or an index error.
- **Forgetting to unsort.** `sgmv` computes rows in sorted order. If you don't scatter them back with the inverse permutation, each row gets another row's output.
- **Treating $-1$ as the last adapter.** In PyTorch, `A_stack[-1]` is the last adapter. A base-only row must return zero, not that adapter's delta.
- **Merging for the first request and serving everyone.** The merged weight is right for one adapter. Check it against a second.
- **Evicting a pinned adapter.** The victim comes from the adapters outside the current batch. A batch that can't fit is an error, not a reason to evict.

> [!RECAP]
> - The update is $y = xW + (xA)B \cdot \alpha/r$, where $r$ is the second dimension of $A$. Merging $AB$ into $W$ serves one adapter.
> - A mixed batch uses an unmerged base GEMM plus a per-row detour. `bgmv` gathers by adapter id. `sgmv` sorts by id and runs a matmul pair per segment.
> - The detour is $0.40\%$ of the up projection's FLOPs at rank 16, and it does 1 FLOP per byte. The cost is the resident adapter: $138{,}412{,}032$ bytes at rank 16.
> - A rank-16 MLP adapter loads in $4.325$ ms at 32 GB/s, and a rank-64 adapter in $17.302$ ms.
> - Pin the running batch's adapters. A frequency-based victim beats FIFO and LRU on a trace with a scan.

## Check your understanding

> [!QUESTION] A merged weight for adapter 0 serves a batch where rows 0 and 1 use adapters 0 and 1. What does row 1 get?
> Row 1 gets $xW + \tfrac{\alpha}{r}(xA_0)B_0$, the base output plus adapter 0's update, when it needs adapter 1's. The merged weight carries only one $AB$. The lab's merge check applies the weight merged for one adapter to a second adapter's rows and requires a miss larger than 1.

> [!QUESTION] The LoRA path is $0.40\%$ of the FLOPs but the chapter calls it a memory problem. Why?
> At one token per adapter the path does $2r(d_\text{in} + d_\text{out})$ FLOPs on $2r(d_\text{in} + d_\text{out})$ bytes, 1 FLOP per byte, against a ridge of $161.2$. Its time is its bytes over bandwidth, and the bytes are the whole adapter, read once per distinct adapter. The FLOP share is small, and the byte share grows with the number of adapters in the batch.

> [!QUESTION] Why does `sgmv` read fewer bytes than `bgmv` when ids repeat?
> `bgmv` reads the adapter once per row. `sgmv` groups the rows by adapter and reads it once per segment. With ids $[3, 3, 0, 4, -1, 1, 3, 0, 4, 2, 1]$, there are 11 rows and 5 distinct adapters, so `sgmv` reads 5 adapters instead of 10.

> [!QUESTION] Which unit bug does a rank-16 load time of 4.325 ms guard against?
> The load is $138{,}412{,}032 / (32 \times 10^9)$ seconds, which is $4.325$ ms. If you divide by $32 \times 2^{30}$ you get $4.03$ ms. If you forget that a bfloat16 weight is 2 bytes, the load is half that. PCIe's 32 GB/s uses $10^9$ bytes, which is the unit chapter 17b uses.

> [!QUESTION] The frequency policy wins at 4 GiB. Is it the best policy for every trace?
> No. It assumes that past popularity predicts future use. If popularity shifts, old high counts keep stale adapters resident, and a policy with decay does better. At 8 GiB on the lab's trace FIFO also beats LRU, so any policy's rank depends on the trace and the pool size.

## Lab

> [!TRY]
> Build the LoRA path and the pool on the CPU. You pass when `lora_forward` returns $(21, 16)$ on the worked example, `bgmv` and `sgmv` match a per-request loop with repeated and permuted ids and a base-only row, the cost functions return $720{,}896$ FLOPs and $138{,}412{,}032$ bytes, and the frequency policy spends at most 0.9 of the PCIe time of the better of FIFO and LRU on the seeded trace.

The lab runs on the CPU. Implement the following:

- `lora_forward(x, W, A, B, alpha)` and `merge_lora(W, A, B, alpha)`, using $\alpha / r$ with $r$ read from `A`. The first check calls only `lora_forward`.
- `bgmv(x, A_stack, B_stack, adapter_ids, alpha)`, which returns the delta only. An id of $-1$ gives a zero row.
- `segment_by_adapter(ids)`, which returns a stable order and the `(adapter, start, end)` segments, and `sgmv`, which returns rows in request order.
- `lora_cost`, `adapter_bytes`, and `adapter_load_ms`, using 2 bytes per value and 32 GB/s.
- `simulate_adapter_pool(trace, sizes, capacity_bytes, policy)` for `"fifo"`, `"lru"`, and `"freq"`, with the running batch pinned.

Scaling by alpha instead of alpha over r fails the worked example and every `bgmv` check. Indexing the stack by batch position fails the permuted-id check. An LRU victim passed off as `"freq"` fails the pool check.

## Further reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Hu and colleagues define the frozen weight plus a low-rank update, the $\alpha / r$ scale, and the merge that adds no inference latency.
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285). Adapters live in main memory, the running batch's adapters move to the GPU, and Unified Paging shares one pool between adapters of different ranks and KV tensors.
- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547). A kernel that batches the LoRA detour across adapters, with one copy of the base model on the GPU.
- [LoRA adapters in vLLM](https://docs.vllm.ai/en/latest/features/lora/). The `--enable-lora` flag, `--lora-modules`, and `--max-lora-rank`, which the page advises setting to the highest adapter rank.
- [dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang). OSDI 2024. It merges and unmerges adapters dynamically, with a credit-based batching algorithm to decide when.
