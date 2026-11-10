---
title: Pipeline and context parallelism
slug: 19b-pipeline-and-context-parallelism
part: "Part 6 — Scaling"
summary: A pipeline schedule as a Gantt table and its bubble, why decode barely pipelines, and ring attention with load-balanced causal sharding for a single request that is too long for one GPU.
minutes: 100
gpu: false
objectives:
  - "Build a pipeline schedule as a Gantt table, and derive and check its bubble fraction (p-1)/(m+p-1)."
  - "Compute decode utilization and inter-token latency through a pipeline with and without micro-batches."
  - "Shard attention over a simulated ring, merge the blocks with the log-sum-exp, and check the result against full attention."
  - "Show why a contiguous causal split leaves ranks idle, and balance it with striped or zigzag assignment."
  - "Compare tensor, pipeline, context, and data parallelism for one long request, with this model's memory, decode, and communication numbers."
lab: 19b-pipeline-and-ring
---

# Pipeline and context parallelism

> [!TLDR]
> - A pipeline puts layers on different GPUs. With `p` stages and `m` micro-batches the schedule has `m + p - 1` time slots, and the idle share is the bubble, `(p-1)/(m+p-1)`. Four stages and 16 micro-batches leave 15.8% idle.
> - A decode step has one token in flight, so a pipeline has one busy stage at a time: utilization is `1/p`, and a sequence's inter-token latency does not shrink (42.2 ms on this model, computed). More concurrent micro-batches fill the stages and raise throughput, not latency.
> - Ring attention shards the sequence, passes K and V around a ring, and merges the partial results with the log-sum-exp from [chapter 14a](/c/14a-split-kv-decode-attention). It is exact: the lab's error against full attention is below 1e-6.
> - A causal mask makes a contiguous split unbalanced. On 4 ranks the busiest rank does 1.74 times the mean, and the ring finishes in 57.5% of the ideal time. Striped or zigzag assignment brings the imbalance to 1.05 and 1.00.
> - For one 1,048,576-token request on four A100s, tensor parallelism has the shortest decode floor (27.1 ms), pipeline parallelism cuts memory but not latency (96.1 ms), and context parallelism shards the cache but keeps the weights whole (55.9 ms). All of these are computed floors, not measurements.

[Chapter 19](/c/19-tensor-parallelism) splits every matrix across GPUs and ends with one paragraph on the other two strategies. This chapter gives pipeline parallelism a schedule you can count, and adds the strategy that the 64 KiB-per-token cache forces on you at long context: split the *sequence*.

The cache is the reason. At 1,048,576 tokens, this model's 16 full-attention layers hold 64 GiB of keys and values for one request. One A100 has about 19 GiB left for cache after the weights. You need at least four GPUs to hold that request, and the choice among tensor, pipeline, and context parallelism decides how fast it decodes and how much traffic it puts on the link.

Everything here runs on one device. A rank is an index, a ring step is a loop iteration, and every time is arithmetic over the course's numbers: 1275 GB/s measured copy, 300 GB/s NVLink one way, 32 GB/s PCIe one way, and 312 TFLOP/s peak. No multi-GPU run produced anything in this chapter.

## Before you start

**The ring and the alpha-beta model, from [chapter 19](/c/19-tensor-parallelism).** A ring all-reduce over $P$ ranks takes $2(P-1)$ hops and sends $2(P-1)/P$ of the tensor per rank. One message costs $2(P-1)\alpha + \frac{2(P-1)}{P}\frac{S}{\beta}$, with $\alpha = 4\,\mu\text{s}$ per hop. Lab 19 simulates that ring, and this lab reuses it unchanged.

**The log-sum-exp merge, from [chapter 14a](/c/14a-split-kv-decode-attention).** Two partial attention results, each with a normalized output $O_i$ and a log-sum-exp $L_i$, combine as $L = \log\sum_i e^{L_i}$ and $O = \sum_i e^{L_i - L} O_i$. A block with no visible key has $L_i = -\infty$ and adds nothing.

**The causal mask, from [chapter 14](/c/14-flash-attention).** Query $i$ sees key $j$ only when $j \le i$. Chapter 14 skips whole tiles that are entirely masked. A ring skips whole blocks the same way.

**KV bytes, from [chapter 9](/c/09-the-kv-cache).** One full-attention layer stores $2 \times 4 \times 256 \times 2 = 4096$ bytes of K and V per token. Sixteen layers store 65,536 bytes, 64 KiB.

## A pipeline schedule is a table you can count

This section builds the pipeline's Gantt table and derives its bubble.

Cut the 64 layers into $p$ *stages*, one per GPU. Cut the work into $m$ *micro-batches*. For inference, a micro-batch is a group of sequences, or a chunk of one long prompt, as in [chapter 16](/c/16-continuous-batching)'s chunked prefill. Stage $s$ cannot start micro-batch $k$ until stage $s-1$ has finished it. Give each stage one time unit per micro-batch.

Stage $s$ then runs micro-batch $t - s$ at time $t$. The first micro-batch reaches the last stage at $t = p - 1$, and the last micro-batch leaves it at $t = m + p - 2$. That is $m + p - 1$ slots. Four stages and four micro-batches give this table, where `-` is idle:

| Time | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Stage 0 | 0 | 1 | 2 | 3 | - | - | - |
| Stage 1 | - | 0 | 1 | 2 | 3 | - | - |
| Stage 2 | - | - | 0 | 1 | 2 | 3 | - |
| Stage 3 | - | - | - | 0 | 1 | 2 | 3 |

The table has $p(m+p-1) = 28$ cells, and $pm = 16$ of them hold work. The rest, $p(p-1) = 12$, are the *bubble*:

$$
\boxed{\text{bubble} = \frac{p(p-1)}{p(m+p-1)} = \hlc{\frac{p-1}{m+p-1}}}
$$

For this table that is $3/7 = 0.4286$. The lab counts the idle cells in the table it builds and compares the count with this formula for five $(p, m)$ pairs.

```viz
19b-pipeline-gantt
```

More micro-batches shrink the bubble, because the fill and drain triangles stay the same size while the middle grows. At $p = 4$, one micro-batch leaves 75% idle, four leave 42.9%, and sixteen leave 15.8%. At $p = 8$ and $m = 32$ the bubble is 17.9%, so a deeper pipeline needs more micro-batches to hold the same fraction.

> [!NOTE] Interleaving cuts the bubble by its chunk count
> Megatron-LM's interleaved schedule gives each GPU $v$ smaller, non-adjacent chunks of layers, so a micro-batch visits each GPU $v$ times. The bubble time falls by a factor of $v$. In the fraction form the lab uses, that is $(p-1)/(vm + p - 1)$. At $p = 4$ and $m = 16$, $v = 2$ gives 8.57% and $v = 4$ gives 4.48%. The cost is $v$ times as many point-to-point sends. This is the training schedule from the paper in the reading list. For inference, which has no backward pass, the same table applies and the 1F1B memory argument does not.

For a single long prompt, the micro-batches are chunks of the prompt. At 1,048,576 tokens on four GPUs, 16 chunks leave a 15.8% bubble, and the compute floor rises from 218.4 s to 259.4 s at the 312 TFLOP/s peak. That is computed, and it treats every chunk as equal cost. Later chunks attend to more keys, so a real schedule is less even.

## Decode barely pipelines

This section applies the same table to decode, where each micro-batch is one token per sequence.

A decode step reads every weight once ([chapter 10](/c/10-roofline)). With the weights split over $p$ stages, each stage reads $53.8 \times 10^{9} / p$ bytes. At the measured 1275 GB/s that is 21.098 ms at $p = 2$ and 10.549 ms at $p = 4$. Between stages, the hidden state of one token, 10,240 bytes, crosses the link in $4\,\mu\text{s} + 0.034\,\mu\text{s} = 4.034\,\mu\text{s}$.

A sequence needs its token from stage $p-1$ before it can start the next step, so it cannot overlap with itself. With one micro-batch in flight, one stage works while $p - 1$ wait:

$$
\text{utilization} = \frac{\min(m_{\text{flight}},\, p)}{p}, \qquad
\text{inter-token latency} = p\,(t_{\text{stage}} + t_{\text{hop}})
$$

The latency does not depend on how many micro-batches are in flight, and it does not shrink with $p$. The computed numbers:

| Stages $p$ | In flight | Utilization | Inter-token latency | Trips per second |
|---|---|---|---|---|
| 2 | 1 | 0.50 | 42.204 ms | 23.69 |
| 2 | 2 | 1.00 | 42.204 ms | 47.39 |
| 4 | 1 | 0.25 | 42.212 ms | 23.69 |
| 4 | 4 | 1.00 | 42.212 ms | 94.76 |

Each *trip* is one micro-batch crossing every stage, so throughput scales with micro-batches in flight and stops at $p$. The 42.2 ms is the single-GPU weight read from chapter 19, plus a few microseconds of hops. A pipeline spends no less time reading weights for one sequence. It lets more sequences read them in parallel. ==Pipelining decode buys throughput and memory, not latency.==

> [!WARNING] Splitting a fixed batch into micro-batches makes decode slower
> The table above assumes $m$ independent batches. If you instead cut one batch into $m$ smaller micro-batches, every stage reads its weights once per micro-batch, because a decode step is bound by the weight read and not by the batch size. At $p = 4$ the step takes $(m + 3) \times 10.549$ ms: 42.20 ms at $m = 1$, 52.75 ms at $m = 2$, 73.84 ms at $m = 4$, and 116.04 ms at $m = 8$. Micro-batches help only when they add requests, not when they divide them.

That is the "bubble, and you fill it with micro-batches" statement in chapter 19, made precise. At low load, when there is one request, there is nothing to fill it with.

## Ring attention: shard the sequence, pass K and V

This section shards one attention layer over $P$ ranks and shows that the result is exact.

*Picture.* Cut the sequence into $P$ shards. Rank $r$ keeps the queries of shard $r$ for the whole computation, and it starts with the K and V of shard $r$. At each ring step it computes attention of its queries against whichever K and V block it holds, merges that into a running result, and passes the block to rank $r+1$. After $P$ steps every query has seen every key.

*Words.* At step $s$, rank $r$ holds the block that began on rank $(r - s) \bmod P$. Its partial result is a normalized output $O_s$ and a log-sum-exp $L_s$. The running pair folds in each new block with the merge from chapter 14a. The merge is associative, so the order in which blocks arrive does not matter, and the result equals attention over all keys at once.

The one detail is the mask. Query $i$ sees key $j$ only when $j \le i$, and $i$ and $j$ are positions in the *whole* sequence, not indices inside a block. The lab's `block_attention` takes each token's position and builds the mask from it. A block whose keys are all in the future of every query returns zeros and $L = -\infty$, and the ring skips it.

> [!EXAMPLE] Four ranks, 64 tokens, contiguous shards
> Each shard is 16 tokens. At step 0 every rank holds its own block, a 16 by 16 triangle with $16 \times 17 / 2 = 136$ visible pairs. At step 1 rank 0 holds the block from rank 3, whose keys all come after its queries, so it skips. Ranks 1 to 3 hold earlier blocks and compute all $16 \times 16 = 256$ pairs. The lab's check expects this work matrix, row per step and column per rank: `[[136,136,136,136], [0,256,256,256], [0,0,256,256], [0,0,0,256]]`, 2080 pairs in all, which is $64 \times 65 / 2$.

Against full attention on 2 heads, 64 tokens, and dimension 16, the maximum absolute error is $2.4 \times 10^{-7}$ for contiguous shards, $3.6 \times 10^{-7}$ for striped, and $2.4 \times 10^{-7}$ for zigzag. The ring changes the order of the float32 additions, nothing else.

### Memory per rank

The shard stays resident, so the *cache* per rank is $1/P$ of the full K and V. During the ring a rank also holds the block it is receiving while it computes on the current one, so the *peak* is $2/P$. The lab reports that peak as `max_memory_per_rank_ratio`:

| Ranks $P$ | Cache ratio | Peak ratio | K and V of one layer at 1,048,576 tokens, peak |
|---|---|---|---|
| 1 | 1 | 1 | 4.000 GiB |
| 2 | 0.5 | 1.0 | 4.000 GiB |
| 4 | 0.25 | 0.5 | 2.000 GiB |
| 8 | 0.125 | 0.25 | 1.000 GiB |

Two ranks save nothing at the peak, because the receive buffer is as large as the shard. Ring attention pays off from $P = 3$. The resident cache that a decode step reads is the $1/P$ column.

### When the transfer hides behind the compute

A ring step moves one shard of K and V while it computes on another. The transfer is free when the compute takes at least as long. For $b$ tokens per shard, a balanced causal block does $2 b^{2} \cdot 24 \cdot 256$ FLOPs, and the transfer is $4096\,b$ bytes per layer:

$$
\hlb{\frac{2\, b^{2} H D}{F}} \ge \hla{\frac{4096\, b}{\beta}}
\quad\Longrightarrow\quad
b \ge \frac{4096\, F}{2\, H D\, \beta}
$$

Here $H = 24$ query heads, $D = 256$ is the head dimension, $F = 312 \times 10^{12}$ FLOP/s, and $\beta$ is the link rate. At $\beta = 300$ GB/s the bound is 346.7 tokens per shard. At the PCIe rate of 32 GB/s it is 3,250 tokens. A kernel that runs below the peak rate lowers both bounds, because the compute takes longer. At 1,048,576 tokens on 4 ranks the shard is 262,144 tokens, so the ring is far above both. The bound matters for short prompts split across many ranks, not for the 1M-token case.

## Causal masking unbalances the ring

This section shows that contiguous shards waste the causal mask's savings, and how two other layouts fix it.

Each ring step waits for its slowest rank. The time of the whole ring is the sum over steps of the *busiest* rank's work, called the *critical path*, and the ideal is the mean rank's total work. The ratio of the two is the efficiency.

With contiguous shards, rank 3 holds the last 16 queries and every earlier block is fully visible to it. It does a full block at every step. In the 4-rank example, the rank totals are 136, 392, 648, and 904 pairs, and the critical path is $136 + 3 \times 256 = 904$. The ideal is $2080 / 4 = 520$. The efficiency is $520 / 904 = 0.575$. ==The causal mask halves the total work and saves almost none of the time.==

*Striped attention* gives rank $r$ the tokens $r, r+P, r+2P, \dots$. Every block then contains an even sample of positions, so each block is about half visible: a full triangle when the key block is earlier in the rank order, and a strict triangle (diagonal excluded) when it is later. *Zigzag* assignment, which Llama 3 uses for training, cuts the sequence into $2P$ chunks and gives rank $r$ chunks $r$ and $2P-1-r$.

```viz
19b-ring-work
```

For 64 tokens and 4 ranks, computed by the lab:

| Assignment | Pairs per rank | Busiest over mean | Critical path | Efficiency |
|---|---|---|---|---|
| Contiguous | 136, 392, 648, 904 | 1.7385 | 904 | 0.5752 |
| Striped | 496, 512, 528, 544 | 1.0462 | 544 | 0.9559 |
| Zigzag | 520, 520, 520, 520 | 1.0000 | 520 | 1.0000 |

The striped layout is not exactly balanced, because the strict and full triangles differ by 16 pairs per block. That gap is one token's worth per block, so it shrinks as shards grow. At 64 tokens, striped efficiency falls from 0.985 at 2 ranks to 0.903 at 8 ranks, while contiguous falls from 0.670 to 0.537. Zigzag is exact here. Striped Attention reports up to 1.45 times the training throughput of Ring Attention on A100s at 256k tokens. That is that paper's measurement and not one from this machine.

> [!NOTE] Llama 3 gathers instead of rotating
> The Llama 3 report uses the zigzag split, and it all-gathers K and V before computing attention for the local queries, rather than overlapping a ring. The report gives two reasons: the all-gather supports other masks, such as a document mask, more easily, and with grouped-query attention the K and V tensors are much smaller than Q, so the exposed all-gather latency is small. A ring and an all-gather move the same K and V bytes. They differ in overlap and in how easily a mask changes.

> [!WARNING] This model's 48 linear-attention layers do not ring
> The chapter's ring covers the 16 full-attention layers. The 48 gated-delta layers carry a fixed 147.8 MiB recurrent state from each token to the next ([chapter 6](/c/06-gated-delta-rule)), so a token shard needs the final state of the shard before it. Handing that state over costs little in bytes, but it serializes the ranks. The lab does not model that, and this chapter does not derive a fix. Treat the context-parallel numbers below as the full-attention layers' share.

## DeepSpeed Ulysses: shard the heads for the attention step

Ulysses splits the sequence like context parallelism does, and then changes what it shards. Before attention, an all-to-all converts each rank's sequence shard of all heads into all tokens of $1/P$ of the heads. Each rank runs ordinary attention, with no ring and no mask surgery, on its heads. A second all-to-all converts the output back to a sequence shard. The authors' analysis says the communication volume stays constant when the sequence length and the device count grow together.

The limit is the head count. Ulysses needs at least one head per rank, and with grouped-query attention the binding count is the KV heads. This model has 4, so Ulysses stops at 4 ranks unless you replicate KV heads. Ring attention has no such limit.

The bytes also differ for this model. Per full-attention layer on 4 ranks at 1,048,576 tokens, Ulysses sends $(P-1)/P$ of each of four local tensors: Q and the output at 3.22 GB each, and K and V together at 1.07 GB. That is 5.64 GB per layer and 90.19 GB over 16 layers. The ring sends the K and V shard $P - 1$ times: 51.54 GB. With 24 query heads against 4 KV heads, moving K and V is cheaper than moving Q and the output. Computed from the geometry, not measured.

## One long request, four strategies

This section compares tensor, pipeline, context, and data parallelism for a single 1,048,576-token request on four A100s.

The decode column is a bandwidth floor at batch 1: bytes each GPU reads per step at 1275 GB/s, plus the collectives. The prefill column is the bytes one rank sends. The decode formulas follow chapter 19's alpha-beta model with NVLink at 300 GB/s. Prefill compute at the 312 TFLOP/s peak is $2.73 \times 10^{17}$ FLOPs, $5.64 \times 10^{16}$ dense and the rest causal attention in 16 layers.

| Strategy | Weights per GPU | Cache per GPU | Fits | Decode floor | Prefill sent per rank | Prefill compute floor |
|---|---|---|---|---|---|---|
| Data (one replica) | 50.11 GiB | 64.00 GiB | No | 96.09 ms | 0 | 873.7 s on one GPU |
| Tensor | 12.53 GiB | 16.00 GiB | Yes | 27.10 ms | 2,061.58 GB | 218.4 s |
| Pipeline | 12.53 GiB | 16.00 GiB | Yes | 96.11 ms | 10.74 GB | 259.4 s |
| Context | 50.11 GiB | 16.00 GiB | Yes | 55.87 ms | 51.54 GB | 218.4 s |

"Fits" compares weights plus cache with chapter 2's budget: the weights plus about 19 GiB, which is 69.11 GiB. Reading the table by column:

- **Tensor** has the lowest decode floor. Each GPU reads a quarter of the weights and a quarter of the cache (24.02 ms), and 128 all-reduces add 3.079 ms. Its prefill traffic is the highest: 2.06 TB per rank, which is 6.9 s at NVLink speed and 64.4 s at PCIe speed, against a 218 s compute floor. The degree is capped at 4 by the KV heads.
- **Pipeline** divides the memory and sends almost nothing, 10.7 GB. Decode stays at the single-GPU floor of 96.1 ms, and prefill pays the 15.8% bubble.
- **Context** keeps the whole 50.11 GiB of weights on every GPU, so decode reads $53.8$ GB plus a quarter of the cache: 55.9 ms. The 16 per-layer merges, each an all-gather of float32 partial outputs and log-sum-exps, add 0.196 ms. Its prefill compute floor equals tensor's, with 51.5 GB of K and V traffic that hides behind the compute above the 347-token shard bound.
- **Data** cannot serve this request. A replica holds 64 GiB of cache against a 19 GiB budget, and a single request does not split across replicas.

> [!KEY] Pick by what limits the request
> Tensor parallelism wins decode latency and needs the fastest link. Context parallelism wins when the cache, not the weights, is the problem and you cannot raise the tensor degree: it has no head-count limit and sends far less than tensor parallelism in prefill. Pipeline parallelism wins on link traffic and loses on latency. Combine them as [chapter 19](/c/19-tensor-parallelism) says: tensor inside a node, and the others across nodes.

For the performance model that turns these floors into goodput, see [chapter 17a](/c/17a-performance-model).

## What the real collective adds

The lab's last check starts two CPU processes with PyTorch's gloo backend, runs `all_reduce` on seeded tensors, and compares the result with the simulated ring from lab 19. It checks the *semantics*: the real collective and the simulation agree. Gloo on a CPU says nothing about latency or bandwidth, and it does not exercise NCCL.

NCCL needs GPUs that the lab dispatcher does not provide, so no NCCL output is in this chapter. To measure $\alpha$ and $\beta$ for your own links, run NCCL's all-reduce benchmark on a multi-GPU machine and fit the two terms from chapter 19's model to its message-size sweep. The 4 µs and the link rates used here are the course's assumptions.

If the check cannot start gloo in your environment, for example because the two processes time out, it reports a skip with the reason and passes. A wrong all-reduce still fails it.

## What goes wrong

- **A block-local causal mask.** Building the mask from indices inside a block is right for the diagonal block and wrong for every other. The error against full attention is 2.5 in the lab's mutant run.
- **Contiguous shards for a causal ring.** The output is exact and the schedule is slow. The lab checks the work balance, so a "striped" function that returns contiguous runs fails.
- **Forgetting to restore token order.** Striped and zigzag shards come back permuted, and the output must be scattered to the owners' positions.
- **Averaging blocks instead of merging.** The partial outputs need their $e^{L_i - L}$ weights. Chapter 14a shows the failure.
- **Counting the bubble as $(p-1)/m$.** The denominator is the number of slots, $m + p - 1$. At $p = 4$ and $m = 16$, $3/16$ is 18.75% against the true 15.79%.
- **Splitting a decode batch to fill the pipeline.** Each micro-batch re-reads the weights.
- **Expecting pipeline parallelism to lower decode latency.** It does not. A sequence still crosses every stage.
- **Reading a floor as a measurement.** Every time in this chapter is bandwidth or peak-FLOP arithmetic. A real run adds kernel inefficiency and overlap losses.

> [!RECAP]
> - A pipeline's Gantt table has $m + p - 1$ slots, and its bubble is $(p-1)/(m+p-1)$: 42.9% at $p = 4$, $m = 4$ and 15.8% at $m = 16$.
> - Decode through a pipeline has utilization $\min(m, p)/p$ and an unchanged 42.2 ms inter-token latency. Dividing a fixed batch makes the step slower.
> - Ring attention is exact. Each rank merges the K and V blocks it receives with the log-sum-exp merge, and masks by global position.
> - A ring step waits for its busiest rank. Contiguous causal shards give efficiency 0.575 on 4 ranks, striped 0.956, and zigzag 1.000.
> - The ring's peak memory is $2/P$ of the K and V, and its transfer hides behind compute above 347 tokens per shard on NVLink.
> - For one 1,048,576-token request on four A100s, tensor decodes fastest (27.1 ms floor), context shards the cache and keeps the weights whole (55.9 ms), and pipeline keeps the 96.1 ms single-GPU latency.

## Check your understanding

> [!QUESTION] A pipeline has 8 stages. How many micro-batches keep the bubble under 10%?
> You need $7/(m+7) < 0.1$, so $m + 7 > 70$ and $m \ge 64$. At $m = 63$ the bubble is exactly 10%, so 64 is the first count below it. Deeper pipelines need many more micro-batches.

> [!QUESTION] Why does adding a second GPU to a pipeline not cut a sequence's inter-token latency?
> The token must pass through every stage in order. Each stage reads half the weights, so a stage takes half as long, but there are twice as many stages. The product is the single-GPU weight read, 42.2 ms, plus the hop time.

> [!QUESTION] Ring attention on 2 ranks has a peak memory ratio of 1.0. What is the benefit?
> The resident cache is half the K and V, so a decode step reads half the cache. The peak equals the full size only while the next block is in flight. With a double-buffered ring, savings in the peak start at 3 ranks.

> [!QUESTION] Striped assignment on 4 ranks has an imbalance of 1.046, not 1.000. Why, and what removes it?
> A striped block is a full triangle when the key rank is lower than the query rank and a strict triangle otherwise, and they differ by one diagonal per block. Zigzag pairs an early chunk with a late one for each rank, so the totals match exactly: 520 pairs each.

> [!QUESTION] Why can Ulysses not run on 8 ranks for this model without replicating KV heads?
> It shards by head, and the model has 4 KV heads. Eight ranks would leave four of them with no KV head. Ring attention shards tokens, so the head count does not limit it.

## Lab

> [!TRY]
> Build the ring: `block_attention`, the three shard layouts, `ring_attention`, and the work and memory accounting. Then build the pipeline Gantt table and decode model, and the one-request comparison. You pass when ring attention matches full attention for every layout, contiguous work is imbalanced and striped and zigzag are not, and the Gantt table's bubble equals $(p-1)/(m+p-1)$.

The harness merges blocks with `merge_partials` from chapter 14a and checks a real two-process gloo `all_reduce` against `ring_all_reduce` from lab 19. Both are given. Implement the following:

- `block_attention(q, k, v, q_pos, k_pos, causal)`, returning the normalized output and the log-sum-exp, with zeros and $-\infty$ for a row that sees no key.
- `shard_indices(total, world, kind)` for `contiguous`, `striped`, and `zigzag`.
- `ring_attention(q, k, v, world, kind, causal)`, returning the output in token order and the work matrix, one row per ring step.
- `work_balance(work)`, with the per-rank totals, the imbalance, the critical path, and the efficiency.
- `memory_per_rank(tokens, world)` and `min_shard_tokens(link_bps)`.
- `pipeline_schedule(stages, microbatches)`, `bubble_fraction(schedule)`, and `bubble_formula(stages, microbatches, chunks)`.
- `decode_pipeline(stages, in_flight, stage_time, hop_time)`.
- `prefill_flops(tokens)` and `compare_strategies(tokens, world, chunks)`. The starter's docstring gives each strategy's formula.

The checks run in about two seconds on a CPU. The starter fails the first nine, and the tenth passes because both its inputs are given.

## Further reading

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889). Overlap the K and V transfer with blockwise attention, so the supported sequence grows with the device count.
- [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431). The causal imbalance and the striped layout. The 1.45 times throughput at 256k tokens is that paper's A100 and TPU measurement.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783), section 3.3.2, "Parallelism for Model Scaling". Context parallelism with $2 \times$ CP chunks per rank and an all-gather of K and V.
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509). All-to-all between sequence and head sharding, with constant communication volume when sequence and devices grow together.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965). Micro-batches and the fill-and-drain schedule.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). The interleaved pipeline schedule.
- [Megatron-LM: training multi-billion parameter language models using model parallelism](https://arxiv.org/abs/1909.08053). The tensor-parallel split from chapter 19.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102). An analytical model for choosing a partitioning layout at inference time.
- Stanford CS336, Spring 2026 schedule: Lecture 7, "Parallelism", and Lecture 8, "Parallelism" ([course page](https://cs336.stanford.edu/)). Titles as listed on the course schedule.
