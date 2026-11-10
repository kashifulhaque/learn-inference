---
title: Mixture-of-experts serving and expert parallelism
slug: 19a-moe-and-expert-parallelism
part: "Part 6 — Scaling"
summary: Route tokens to top-k experts, count how many experts a decode batch reads, and price the all-to-all and the load imbalance that expert parallelism adds.
minutes: 110
gpu: false
objectives:
  - Implement top-k softmax routing with renormalization and capacity-factor token dropping.
  - Dispatch tokens sorted by expert, combine the results, and match a per-token reference.
  - Compute the expected number of distinct experts a batch touches, and the decode bytes and intensity that follow.
  - Compare a grouped matmul over expert segments with a loop over experts.
  - Derive the expert-parallel all-to-all volume and the cost of load imbalance, and plan expert replication.
lab: 19a-moe
---

# Mixture-of-experts serving and expert parallelism

> [!TLDR]
> - A mixture-of-experts (MoE) layer replaces one MLP with $E$ expert MLPs and a router. Each token runs through its top $k$ experts, weighted by renormalized router probabilities. Qwen3-30B-A3B has $E = 128$ and $k = 8$, so it stores 30.5 billion parameters and uses 3.35 billion per token.
> - Decode reads every expert that at least one token in the batch picks. A batch of $B$ tokens touches $E\,(1 - (1 - k/E)^B)$ experts: 8 at $B = 1$, 111.8 at $B = 32$, and 128 from $B = 128$ on. At $B = 32$ the step reads 53.7 GB, as many bytes as the course's dense 27B model.
> - The expert MLPs' arithmetic intensity is $Bk/D$, where $D$ is the number of experts read. It reaches the A100's ridge of 161.2 FLOPs per byte at batch 2,580, against about 162 for a dense MLP. That 16-fold gap is $E/k$.
> - Expert parallelism puts experts on different GPUs. Each token's $k$ copies travel in an all-to-all: 3.5 MiB per rank per direction at 128 tokens per rank on 8 ranks, and 7 of every 8 copies leave the rank.
> - A step waits for the busiest rank. Replicating hot experts and repacking them, as DeepSeek's EPLB does, cuts the synthetic skewed example's slowdown from 1.76 to 1.00.

[Chapter 19](/c/19-tensor-parallelism) splits every matrix of a dense model across GPUs and pays 128 all-reduces per step. A mixture-of-experts model changes three things that chapter takes for granted. The MLP holds 95% of this chapter's example model's parameters, but each token uses a small share of it. Which weights a step reads depends on which tokens are in the batch. And the natural way to split the model is by expert, not by matrix.

Many large open models are mixtures of experts, including DeepSeek-V3, Qwen3-30B-A3B, and Mixtral. This chapter builds the layer, counts what a decode step reads, and prices the communication, so that you can predict why an MoE model is cheap at batch 1 and ordinary at batch 32.

The course model is dense in its MLP, so the worked numbers in this chapter use a published MoE model: Qwen3-30B-A3B. Its numbers come from its `config.json`. The chapter also quotes DeepSeek-V3 and Mixtral where they sharpen a point. Nothing here is an A100 measurement. The routing statistics come from seeded simulations with synthetic routers, and the lab runs on the CPU.

## Before you start

**The decode roofline, from [chapter 10](/c/10-roofline).** An operation that does $F$ FLOPs and moves $B$ bytes takes at least the longer of $F/C$ and $B/\beta$. On the A100, $C$ is 312 TFLOP/s and the rated $\beta$ is 1935 GB/s, so the ridge is 161.2 FLOPs per byte. Decode sits far below it, so its time is its bytes.

**The step model, from [chapter 17a](/c/17a-performance-model).** A decode step's floor is the weight and cache bytes over the bandwidth. This chapter reads weights at the measured copy rate of 1275 GB/s, as chapter 19 does, and uses the rated 1935 GB/s only to place the ridge.

**Ranks and links, from [chapter 19](/c/19-tensor-parallelism).** A *rank* is one GPU's process. One A100 sends at 300 GB/s over NVLink and at 32 GB/s over PCIe, each in one direction. The *alpha-beta model* prices a message as a fixed latency $\alpha$ plus its bytes over the bandwidth $\beta$. Chapter 19 assumes $\alpha = 4\,\mu\text{s}$ per hop, and this chapter reuses that assumption.

**The geometry of Qwen3-30B-A3B.** The following numbers come from the model's [`config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json):

- Hidden size $d = 2048$ and 48 layers, each with an attention block and an MoE block.
- $E = 128$ routed experts per layer, $k = 8$ chosen per token, and an expert width of $f = 768$.
- Each expert is a SwiGLU block with a gate, an up, and a down matrix, so it holds $3 \times 2048 \times 768 = 4{,}718{,}592$ parameters, or 9.0 MiB in bfloat16.
- 32 query heads, 4 KV heads, head dimension 128, vocabulary 151,936, untied embeddings, and `norm_topk_prob` set to true.

Adding these up gives 30,532,122,624 parameters: 28,991,029,248 in experts and 1,541,093,376 elsewhere. The weights take 61.06 GB, which is 56.87 GiB. On an 80 GiB A100 that leaves 23.13 GiB, room for about 252,600 tokens of KV cache at 96 KiB per token. Every expert stays resident, even though a token uses 8 of 128.

## What an expert layer computes

An MoE layer sends each token through $k$ of its $E$ expert MLPs and adds their outputs with weights from a router. Picture a hospital front desk. Every patient (token) is seen by the desk (router), which names the $k$ best specialists, and the patient's results are a weighted blend of what those specialists say.

The router is one linear layer, $W_r \in \mathbb{R}^{E \times d}$. It scores every expert, a softmax turns the scores into probabilities, and the token keeps the $k$ largest:

$$
p = \operatorname{softmax}(x W_r^{\top}), \qquad S = \operatorname{top\text{-}k}(p)
$$

Then it divides by the kept mass, so the $k$ weights sum to 1:

$$
\boxed{y = \sum_{e \in S} \hla{g_e}\, f_e(x), \qquad \hla{g_e} = \frac{p_e}{\sum_{j \in S} p_j}}
$$

Here $f_e$ is expert $e$'s SwiGLU block. Qwen3 renormalizes, which its config calls `norm_topk_prob`. The renormalized weights equal a softmax over only the $k$ chosen scores, so the two ways of writing the router agree.

> [!EXAMPLE] Four experts, top 2
> A token's router logits are $2, 1, 0, -1$. The softmax gives $0.6439, 0.2369, 0.0871, 0.0321$. The top two are experts 0 and 1, with a mass of $0.8808$. Dividing by it gives $g = (0.7311, 0.2689)$. Without the division, the layer would output 88% of the intended magnitude, and the shortfall would change with every token.

The following code routes a batch and returns the weights and expert ids, both of shape $(T, k)$:

```python
def route(logits, k):
    probs = torch.softmax(logits.float(), dim=-1)
    top, experts = torch.topk(probs, k, dim=-1)
    return top / top.sum(dim=-1, keepdim=True), experts
```

A real model also adds an auxiliary loss during training so that the router spreads tokens across experts. DeepSeek-V3 instead balances load without an auxiliary loss, and its config sets `topk_method` to `noaux_tc` with sigmoid scores. Serving doesn't train the router, but the balance the training achieved decides how even your load is. The later sections measure what happens when it isn't.

## Capacity: when an expert is oversubscribed

A router doesn't spread tokens evenly, so some experts receive more than their share. A *capacity* caps the slots per expert, and tokens that overflow are dropped from that expert. The Switch Transformer paper defines the capacity as the tokens per batch over the number of experts, times a *capacity factor*. It describes overflowing tokens as not processed by the layer, and a larger factor as fewer overflows at the price of padded, wasted slots. With $k$ experts per token, the same idea is the following:

$$
C = \left\lceil c \cdot \frac{T\,k}{E} \right\rceil
$$

Here $T$ is the number of tokens, $c$ is the capacity factor, and $T k / E$ is the mean load per expert. A dropped assignment contributes nothing to the sum, so the token's output is built from fewer than $k$ experts. The residual connection around the MoE block carries the token through.

The rule for who loses is arbitrary and the lab fixes it: walk the assignments in row-major order, token first and slot second. Each expert keeps the first $C$ it sees. Early tokens win, so the result depends on the order of the batch.

How much does that cost? The following figures come from the lab's functions on seeded uniform routing with $E = 128$ and $k = 8$. A real router is less even than this, so treat them as a floor:

| Tokens | Mean load | Busiest expert | Dropped at $c = 1.0$ | Dropped at $c = 1.25$ | Dropped at $c = 1.5$ |
|---|---|---|---|---|---|
| 256 | 16 | 29 | 8.84% | 1.66% | 0.34% |
| 1,024 | 64 | 81 | 4.36% | 0.01% | 0.00% |
| 4,096 | 256 | 291 | 2.18% | 0.00% | 0.00% |

At $T = 1024$, a factor of 1.25 means 80 slots per expert, just under the busiest expert's 81. Larger batches average out, so they need less slack. Dropping tokens changes the model's output, so most inference engines serve without a capacity and handle imbalance by other means. The [MegaBlocks paper](https://arxiv.org/abs/2211.15841) states its goal as never dropping a token, and builds block-sparse kernels to do so without padding.

## Dispatch, compute, and combine

A layer has three steps: gather the tokens each expert needs, run each expert on its tokens, and scatter the weighted results back. Sorting the assignments by expert makes the gather contiguous, so each expert's tokens form one block of rows.

The *dispatch* flattens the $(T, k)$ assignments that survived the capacity into a list of (token, slot) pairs, sorts the pairs by expert with a stable sort, and gathers the matching rows of $x$. The *combine* multiplies each expert output by its routing weight and adds it into its token's row.

```python
def dispatch(x, experts, keep, n_experts):
    T, k = experts.shape
    flat_keep = keep.reshape(-1)
    tok = torch.arange(T).repeat_interleave(k)[flat_keep]
    slot = torch.arange(k).repeat(T)[flat_keep]
    e = experts.reshape(-1)[flat_keep]
    order = torch.argsort(e, stable=True)
    counts = torch.bincount(e, minlength=n_experts)
    return x[tok[order]], tok[order], slot[order], counts

def combine(y_sorted, token_idx, slot_idx, weights, n_tokens):
    out = torch.zeros(n_tokens, y_sorted.shape[1], dtype=y_sorted.dtype)
    scale = weights[token_idx, slot_idx].unsqueeze(1)
    return out.index_add(0, token_idx, y_sorted * scale)
```

This must equal the obvious per-token loop that runs each token through its own experts. The lab checks that, with and without dropping. Gather and scatter are a permutation, so they add no arithmetic.

> [!KEY] Without a capacity, the layer treats tokens independently
> A token's output depends only on that token and the weights. So permuting the batch permutes the output the same way, and a test of that property catches an index mixed up between dispatch and combine. With a capacity, the property fails on purpose: which tokens are dropped depends on their order.

## Grouped matmul against a loop over experts

Dispatch leaves one block of rows per expert, and each block multiplies its own weight matrix. A loop over the 128 experts launches 128 small matmuls per projection, three projections per layer. With 8 of 128 experts active, most blocks hold a handful of rows, which is the skinny shape that [chapter 13a](/c/13a-skinny-decode-gemm) shows leaves a GPU idle. Python-level loop overhead also grows with $E$.

A *grouped matmul* does all of those products in one kernel. Production engines write it as a fused kernel. vLLM's `fused_moe` defines a Triton kernel, `fused_moe_kernel`, and sorts and pads the token assignments by expert with `moe_align_block_size`. Its modular kernel design permutes the tokens by expert, runs the first matmul, applies the activation and the multiply, runs the second matmul, and unpermutes. This course doesn't run it, because the lab is CPU only. The lab's version, a padded batched matmul, shows the idea and its cost:

1. Pad every expert's block with zero rows to the same height $M$, a multiple of a block size.
2. Stack the blocks into a tensor of shape $(E, M, d)$.
3. Run one `torch.bmm`, then slice each expert's real rows back out.

The cost of the padding is the rows it adds. With block 8 and per-expert counts $5, 0, 17, 2, 0, 9, 1, 4$, the largest block is 17, so $M = 24$, and the bmm computes $8 \times 24 = 192$ rows for 38 real ones, a factor of 5.05. A kernel that tiles each expert's rows separately pads only to the next multiple of the block per expert. MegaBlocks goes further. It treats the whole computation as block-sparse, so that it doesn't have to drop tokens or pad.

The lab checks that the padded product equals the loop on the same tokens, including experts with no rows.

## How many experts does a decode step read?

A decode step reads every expert that at least one token in the batch chose, so the bytes grow with the batch until the experts run out. This is the reason an MoE model behaves differently from a dense one.

Suppose each token picks its $k$ experts uniformly at random, and tokens are independent. By symmetry, a given expert is in one token's top $k$ with probability $k/E$. All $B$ tokens miss it with probability $(1 - k/E)^B$. Expectation is linear, so you add the expert indicators without any independence between them:

$$
\boxed{\mathbb{E}[D(B)] = E\left(1 - \left(1 - \frac{k}{E}\right)^{B}\right)}
$$

The two ends are easy to check. At $B = 1$ the formula gives $E \cdot k/E = k$, exactly 8 experts. For large $B$ it approaches $E$.

A tempting wrong formula treats the $Bk$ slots as independent draws among $E$ experts, which gives $E(1 - (1 - 1/E)^{Bk})$. A token's $k$ experts are distinct, so that formula undercounts: at $B = 1$ it gives 7.78 instead of 8. The lab's tests reject it.

For Qwen3-30B-A3B, the formula gives the following:

| Batch $B$ | Experts read $D$ | Share of 128 | Weights read per step | Floor at 1275 GB/s |
|---|---|---|---|---|
| 1 | 8.00 | 6% | 6.71 GB | 5.26 ms |
| 8 | 51.62 | 40% | 26.47 GB | 20.76 ms |
| 32 | 111.77 | 87% | 53.71 GB | 42.13 ms |
| 64 | 125.94 | 98% | 60.13 GB | 47.16 ms |
| 128 | 127.97 | 100% | 61.05 GB | 47.88 ms |

The bytes are the 3.08 GB of non-expert weights plus $48 \times D \times 9{,}437{,}184$ bytes of experts. At $B = 1$ that is 6,706,065,408 bytes, 11% of the model. At $B = 32$ the step reads 8.01 times as much as at $B = 1$, and 53.71 GB is nearly the 53.79 GB that the course's dense Qwen3.8-27B reads every step. Past $B = 128$ the read stops growing, because every expert is already in.

A real router is not uniform. Tokens in a batch often share a topic, and some experts are popular. Skew makes the same batch touch fewer experts. With a Zipf-shaped popularity of exponent 0.8, a synthetic assumption that has no measured router behind it, $B = 32$ reads 88.6 experts, not 111.8. The closed form still holds when you replace $k/E$ with each expert's own probability $p_e$ and sum $1 - (1 - p_e)^B$ over experts, and the lab does that. Skew helps at small batches and doesn't change where the reads saturate.

```viz
19a-distinct-experts
```

### Intensity and the ridge

Decode is memory bound while the intensity is under the ridge. Each token does $2 \cdot 3df$ FLOPs in each of its $k$ experts, and each expert read costs $3df \cdot 2$ bytes in bfloat16, so the factors of $d$ and $f$ cancel:

$$
\boxed{I(B) = \frac{2\,B\,k\,(3df)}{D(B)\,(3df)\cdot 2} = \frac{B\,k}{D(B)}}
$$

Read it as tokens times experts per token, per expert read. A dense MLP reads each weight once for all $B$ tokens, so its intensity is $B$. An MoE layer's is $B$ times $k/D$, and $D$ starts at $k$ and ends at $E$. At $B = 1$, $I = 1$ for both. At $B = 64$ the dense MLP is at 64, and the MoE layer's 512 expert evaluations over 125.9 experts read give 4.07.

Once $D \to E$, the intensity is $Bk/E$, so it reaches the ridge of 161.24 at $B \approx 161.24 \times E/k$. The lab's search gives 2,580, against 162 for dense, a ratio of 15.9, which is $E/k = 16$. Skew leaves it at 2,580, because every expert is read by then.

> [!KEY] MoE multiplies the ridge batch by $E/k$
> A model that activates 1 expert in 16 needs 16 times the batch to use its arithmetic units. DeepSeek-V3 has 256 routed experts and picks 8 (from its `config.json`), a ratio of 32, so its expert MLPs need a batch of 5,160 by the same arithmetic. Mixtral picks 2 of 8, a ratio of 4, and reaches the ridge at 645.

That is why MoE decode stays memory bound at batch sizes where a dense model has crossed the ridge. It is also why one A100 can't reach the ridge here. At batch 2,580 and a context of 1,024 tokens, the cache alone is $2580 \times 1024 \times 96$ KiB, about 242 GiB, against the 23 GiB left after the weights.

## Expert parallelism

*Expert parallelism* (EP) places different experts on different ranks, so each rank stores $E/W$ of the experts and runs only those. With 8 ranks, each holds 16 of Qwen3-30B-A3B's 128 experts per layer, 7.25 GB of expert weights, plus whatever you replicate. Tokens come to the experts, instead of the experts' matrices being cut, which makes this a different trade from the tensor parallelism of chapter 19.

Each rank starts a layer with its own tokens. Every token chose $k$ experts, and each expert lives on some rank. The layer needs two communication steps:

1. **Dispatch.** Each rank sends a copy of each token to each rank that holds one of its experts.
2. **Combine.** Each rank sends the expert outputs back to the token's home rank, where they are weighted and added.

Each is an *all-to-all*: every rank sends a different message to every other rank. The lab counts it from the assignments. Its matrix has entry $[s, d]$ equal to the number of (token, slot) copies from tokens on rank $s$ to experts on rank $d$, with expert $e$ on rank $\lfloor e/(E/W) \rfloor$. The diagonal is local and free. Everything else crosses a link.

If routing is uniform, a copy is local with probability $1/W$. A rank with $T$ tokens, $k$ copies each, hidden size $h$, and $b$ bytes per element then sends the following in each direction:

$$
\boxed{V = T\,k\,h\,b\,\frac{W - 1}{W}}
$$

That's the same $(W-1)/W$ factor that the ring all-reduce has in chapter 19. The combine sends the same volume back.

> [!EXAMPLE] 128 decode tokens per rank on 8 ranks
> $V = 128 \times 8 \times 2048 \times 2 \times 7/8 = 3{,}670{,}016$ bytes, 3.5 MiB. Seven of each token's eight copies leave the rank. A lab check compares this against the average off-rank bytes of 40 seeded uniform routings, and they agree to within 1%.

At 300 GB/s one way, 3.5 MiB takes 12.2 µs. The alpha-beta model adds the latency of $W - 1 = 7$ messages. With chapter 19's assumed $\alpha = 4\,\mu\text{s}$, that is 28 µs, so one all-to-all costs about 40 µs. A step has two per layer and 48 layers, so the communication costs the following:

$$
t_{\text{comm}} \approx 2 \times 48 \times 40.2\,\mu\text{s} = 3.86\ \text{ms}
$$

The latency term is 70% of each all-to-all. The table shows how the balance shifts with the batch:

| Tokens per rank | Bytes per rank, one direction | Bandwidth term | Latency term | Per step (96 all-to-alls) |
|---|---|---|---|---|
| 1 | 28,672 | 0.10 µs | 28 µs | 2.70 ms |
| 16 | 458,752 | 1.53 µs | 28 µs | 2.84 ms |
| 128 | 3,670,016 | 12.23 µs | 28 µs | 3.86 ms |
| 1,024 | 29,360,128 | 97.87 µs | 28 µs | 12.08 ms |
| 4,096 | 117,440,512 | 391.47 µs | 28 µs | 40.27 ms |

This is a model with an assumed latency, not a measurement. Real systems implement the all-to-all with specialized kernels. The DeepEP library, from the DeepSeek team, provides expert-parallel dispatch and combine kernels, with an FP8 option for the dispatch and both high-throughput and low-latency modes. DeepSeek-V3's config also sets `n_group` to 8 and `topk_group` to 4. That is group-limited routing, which the EPLB README ties to less inter-node traffic.

Expert parallelism pays off by pooling the batch. On 8 ranks with 128 tokens each, 1,024 tokens share each layer, and a given expert sees $1024 \times 8 / 128 = 64$ of them on average, so its intensity is 64, not the 1 you get at batch 1. A rank also reads only its own experts' 7.25 GB. That is the useful trade: more communication in exchange for each expert seeing more tokens.

## Load imbalance

A step ends when its slowest rank ends, so what matters is the maximum load, not the average. Take the number of (token, slot) assignments each rank receives. The step takes time proportional to the largest, and the following measures how much of the average rank's time is wasted:

$$
\text{factor} = \frac{\max_r L_r}{\overline{L}}, \qquad \text{efficiency} = \frac{1}{\text{factor}}
$$

Loads of $10, 10, 10, 2$ have a mean of 8 and a maximum of 10, so the factor is 1.25 and the efficiency is 0.8. The step is 25% longer than a perfect split.

The following numbers use 1,024 tokens on 8 ranks, with 16 contiguous experts per rank, from seeded simulations. The skewed cases give the experts a Zipf-shaped popularity, a synthetic assumption, with random expert ids:

| Routing | Busiest rank over mean | Efficiency |
|---|---|---|
| Uniform | 1.049 | 0.953 |
| Zipf exponent 0.5 | 1.467 | 0.682 |
| Zipf exponent 0.8 | 1.758 | 0.569 |

Even uniform routing wastes 5%, from sampling noise. The skewed cases are an assumption, and real routers vary by model and by workload, but they show the shape of the problem: one hot expert makes a whole rank the bottleneck, and every other rank waits.

### Replicate the hot experts

DeepSeek's open-source *EPLB*, the expert-parallelism load balancer, does two things. It gives heavily loaded experts extra copies, and it packs all the copies onto ranks so that each rank's load is even. It takes estimated expert loads, such as a moving average of earlier traffic, and returns a placement. The repository describes a hierarchical policy that keeps expert groups on the same node, for prefill, and a global policy that ignores groups, for decode.

The lab implements a small version with a deterministic rule:

1. Start with one replica per expert. For each spare slot, give a replica to the expert with the largest load per replica, up to $W$ replicas, with ties going to the lowest expert id.
2. Each replica carries its expert's load divided by its replica count. Place replicas in order of decreasing load, each on the least-loaded rank that has a free slot and doesn't already hold that expert.

> [!EXAMPLE] Four experts on two ranks
> Expert loads are $9, 4, 2, 1$ and there are 6 slots, 3 per rank. Placing experts 0 and 1 on rank 0 and 2 and 3 on rank 1 gives loads $13$ and $3$, a factor of $13/8 = 1.625$. EPLB's plan gives experts 0 and 1 two replicas each (loads 4.5 and 2 per replica) and places them on ranks $[0, 1, 2]$ and $[0, 1, 3]$. The rank loads are $8.5$ and $7.5$, a factor of $1.0625$.

On the skewed simulation with exponent 0.8, packing alone, with no spare slots, moves the factor from 1.758 to 1.172. Eight spare slots, one extra replica per rank, bring it to 1.001. Replicas cost memory, here 9 MiB per expert per layer, and the router must split an expert's tokens among its copies. The lab's plan assumes an even split. The loads in a real system drift, so the placement is recomputed, which is why the input to EPLB is a moving average.

## Why decode batching helps MoE less

Batching a dense model amortizes one weight read over all $B$ tokens, so the cost per token falls as $1/B$. An MoE model's read grows while experts are still being added, and the cost per token falls more slowly. The following compares weight bytes per generated token for Qwen3-30B-A3B and the course's dense 27B model. The two are different models, so this compares scale, not quality:

| Batch | MoE bytes per token | Dense bytes per token |
|---|---|---|
| 1 | 6.71 GB | 53.79 GB |
| 8 | 3.31 GB | 6.72 GB |
| 32 | 1.68 GB | 1.68 GB |
| 128 | 0.48 GB | 0.42 GB |

At batch 1 the MoE model reads 8 times fewer bytes, which is the whole appeal: about 190 tokens per second against 24 at the bandwidth floor. By batch 32 that advantage is gone, because the batch has touched 87% of the experts. After that the MoE model reads its full 61 GB, more than the dense model, and its advantage is only that it does less arithmetic, which doesn't help while the step is memory bound.

Three consequences follow for serving:

- **A big batch is what makes an MoE worthwhile.** The ridge batch is $E/k$ times higher, so you need many more concurrent sequences than you would for a dense model, and the KV cache must hold them. Expert parallelism over many GPUs pools their batches.
- **Memory is the price.** All 61 GB stay resident, even though a token uses 3.35 billion parameters, 11% of them. A smaller active size doesn't make the model fit a smaller GPU.
- **Balance is a serving problem.** Training balances the load on average. At inference, a batch from one user or one topic can be as skewed as the table above, and the placement must follow the traffic.

## What goes wrong

These are the mistakes this chapter's code is most likely to meet:

**Forgetting to renormalize the top-$k$ weights.** The weights sum to less than 1, the layer's output shrinks, and nothing raises an error. Check that each row sums to 1.

**Softmaxing after the top-$k$ on the wrong axis.** The weights must be per token, across that token's $k$ experts. A softmax across the batch mixes tokens, and the permutation test fails.

**Counting experts from slots instead of tokens.** $E(1 - (1 - 1/E)^{Bk})$ ignores that a token's experts are distinct. It is wrong at $B = 1$, where the answer is exactly $k$.

**Dropping tokens in the wrong order.** If the capacity walk is expert-major instead of token-major, a different set of tokens survives. The result is plausible and slightly wrong. Fix the order and test it on a hand example.

**Padding empty experts.** A padded matmul that gives an empty expert a full block of rows wastes compute and masks a bug where a count is off by one. Skip empty experts, or test them explicitly.

**Taking all-to-all bytes as $T k h b$.** The $(W-1)/W$ share of copies that stay on the rank isn't sent. At $W = 8$, forgetting it overstates the traffic by 14%.

**Assuming the batch spreads evenly over ranks.** The busiest rank sets the time. Measure the maximum, not the mean.

> [!RECAP]
> - An MoE layer routes each token to its top $k$ of $E$ experts and adds their outputs with renormalized softmax weights. A capacity factor caps each expert's slots and drops the overflow, in token order.
> - Dispatch sorts the (token, slot) pairs by expert, and combine scatters weighted results back. Without a capacity, the layer equals the per-token loop, and permuting tokens permutes the output.
> - A grouped matmul runs all expert blocks in one kernel. The lab's padded bmm matches a loop over experts and shows the padding cost, 192 rows for 38 real ones in the example.
> - A batch of $B$ tokens reads $E(1 - (1 - k/E)^B)$ experts. For Qwen3-30B-A3B that is 8 at $B = 1$ and 111.8 at $B = 32$. The expert MLPs' intensity is $Bk/D$, so the ridge batch is $E/k$ times the dense one: 2,580 against 162.
> - Expert parallelism sends $Tkhb(W-1)/W$ bytes per rank in each of dispatch and combine. At 128 tokens per rank on 8 ranks that is 3.5 MiB, and latency is 70% of the time.
> - The step waits on the busiest rank. Replicating hot experts and repacking them brings a synthetic factor of 1.76 down to 1.00.

## Check your understanding

> [!QUESTION] Why does an MoE model read 8 times fewer weight bytes than a dense model at batch 1 and about the same at batch 32?
> At batch 1 the token touches exactly $k = 8$ of 128 experts per layer, so the step reads 6.71 GB. A batch of 32 tokens picks $32 \times 8 = 256$ assignments, and most land on different experts. The expected number of distinct experts is $128(1 - (15/16)^{32}) = 111.8$, so the step reads 53.71 GB. The dense model reads 53.79 GB at every batch size, so the two meet near $B = 32$.

> [!QUESTION] The expert MLPs' intensity is $Bk/D$. Why does that put the ridge at $E/k$ times the dense batch?
> A dense MLP reads each weight once for the whole batch, so its intensity is $B$ and it reaches the ridge at $B \approx 161$. In an MoE layer, once the batch is large enough that every expert is read, $D = E$, so the intensity is $Bk/E$, and it reaches 161.24 at $B = 161.24 \times E/k$. For $E/k = 16$ that is 2,580. Skew doesn't change it, because by that batch every expert is read anyway.

> [!QUESTION] With 8 ranks and top-8 routing, what share of a token's expert copies leave its rank, and what does that make the dispatch volume per rank?
> If routing is uniform and the experts are spread evenly, each copy stays on the rank with probability $1/8$, so $7/8$ of the 8 copies, 7 per token, leave it. The volume is $T \times 8 \times 2048 \times 2 \times 7/8$ bytes, which is 28,672 bytes per token, and 3,670,016 bytes at 128 tokens. The combine moves the same amount back.

> [!QUESTION] Loads of 10, 10, 10, and 2 sit on four ranks. How much slower is the step than a perfect split, and what does adding a replica do?
> The step takes time proportional to the maximum, 10, against a mean of 8, so it is 25% slower and the efficiency is 0.8. A replica of a hot expert moves part of its load to another rank. That lowers the maximum toward the mean, as in the lab's example, where replicating two experts takes the factor from 1.625 to 1.0625. The cost is the replica's memory and a router that splits the expert's tokens.

> [!QUESTION] Why can't a capacity-limited MoE layer promise that permuting the batch permutes its output?
> A capacity keeps the first $C$ assignments each expert sees, in token order. Reorder the tokens and a different set is dropped, so the tokens' outputs change. Without a capacity, each token's output depends only on that token, and the property holds.

## Lab

> [!TRY]
> Build routing, capacity, dispatch and combine, a grouped matmul, the expected-experts formula, the all-to-all counts, and an EPLB plan. You pass when the layer matches a per-token loop, the closed forms match seeded simulations, and the exact-number examples come out as this chapter derives them.

The lab runs on the CPU with no weights and finishes in a second or two. Implement `route` first. It is the smallest piece, and the first check needs only that. Then add `expert_capacity` and `apply_capacity`, `dispatch` and `combine`, and `moe_layer`, which ties them together with a loop-over-experts matmul. Next add `grouped_matmul_loop`, `grouped_matmul_padded`, and `padded_rows`.

The statistics come next: `expected_distinct_experts`, which takes an optional list of per-expert probabilities, `moe_decode_intensity`, and `batch_to_ridge`. Last come the expert-parallel pieces: `all_to_all_matrix`, `all_to_all_bytes`, `all_to_all_volume`, `expert_loads`, `rank_loads`, `imbalance_cost`, and `eplb_plan`.

The harness checks the following:

- The top-2 weights for logits $2, 1, 0, -1$ are $(0.7311, 0.2689)$, and every row sums to 1.
- Capacities of 2, 10, and 3 for the examples, and the exact keep mask for a hand-built assignment.
- Dispatch order, slots, and counts for a small example, and a combine that restores each token.
- The layer equals a per-token loop to $10^{-4}$, with and without dropping, with at least one dropped assignment.
- Permuting the tokens permutes the output.
- The padded matmul equals the loop, including empty experts, and computes 192 rows for the 38-row example.
- The closed form matches 8 at $B = 1$, and agrees with a seeded Monte Carlo within five standard errors, for uniform and skewed routing.
- At $B = 1$ the intensity is exactly 1, and the batch to the ridge brackets 161.24 between two integers.
- A hand-built assignment gives the exact all-to-all matrix and per-rank bytes, and the closed form agrees with a seeded simulation to 1%.
- The imbalance factor for $10, 10, 10, 2$ is 1.25, and the EPLB plan on $9, 4, 2, 1$ matches the replicas, ranks, and loads in the example above.

## Further reading

- [Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity](https://arxiv.org/abs/2101.03961). The expert capacity and capacity factor, and the overflow of tokens, are in Section 2 and Figure 3.
- [Mixtral of experts](https://arxiv.org/abs/2401.04088). Eight experts per layer, two per token, 47 billion parameters in total and 13 billion active.
- [DeepSeekMoE: towards ultimate expert specialization in mixture-of-experts language models](https://arxiv.org/abs/2401.06066). Fine-grained experts and shared experts.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437). 671 billion parameters with 37 billion active, and the bias-based load balancing that avoids an auxiliary loss. The [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) lists 256 routed experts, top 8, and 8 groups with 4 chosen.
- [Qwen3-30B-A3B `config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json). The geometry this chapter uses.
- [MegaBlocks: efficient sparse training with mixture-of-experts](https://arxiv.org/abs/2211.15841). Block-sparse kernels that never drop tokens.
- [DeepEP](https://github.com/deepseek-ai/DeepEP). Expert-parallel dispatch and combine kernels, with FP8 dispatch.
- [EPLB](https://github.com/deepseek-ai/EPLB). Expert replication and packing, with a hierarchical and a global policy.
- [vLLM, Fused MoE modular kernel](https://docs.vllm.ai/en/latest/design/fused_moe_modular_kernel/). Prepare and dispatch, the experts stage with permute and unpermute, then combine.
- [Stanford CS336, Language Modeling from Scratch, Spring 2025](http://cs336.stanford.edu/spring2025/). Lecture 4, Mixture of experts, 10 April. The syllabus lists titles and dates; this chapter doesn't claim what the lecture covers beyond its title.
