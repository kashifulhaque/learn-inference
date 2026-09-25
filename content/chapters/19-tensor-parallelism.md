---
title: Tensor parallelism
slug: 19-tensor-parallelism
part: "Part 6 — Scaling"
summary: Deriving column and row sharding as block-partitioned matrix products, and finding where the interconnect stops paying.
minutes: 100
gpu: true
objectives:
  - Derive column-parallel and row-parallel linear layers as block-partitioned matrix multiplies.
  - Explain why pairing them gives exactly one all-reduce per sub-block rather than two.
  - Derive why attention shards by head and the MLP by intermediate dimension.
  - Compute communication volume per token per layer at degree P and compare it against NVLink and PCIe.
  - Describe how the hybrid model's 48 linear-attention layers and their recurrent state shard.
  - Contrast tensor parallelism with pipeline and data parallelism.
lab: 19-tensor-parallel
---

# Tensor parallelism

> [!TLDR]
> - A column-parallel layer splits the weight's output dimension and needs no
>   communication. A row-parallel layer splits the input dimension and ends in
>   one all-reduce of partial sums.
> - Pair them, column then row, and each sub-block costs one all-reduce: 128 per
>   forward pass through 64 layers.
> - Attention shards by head, the MLP by intermediate channel, and the
>   linear-attention layers by value head, recurrent state included. The head
>   counts cap this model at $P \mid 4$.
> - Per-rank traffic never exceeds 40 KiB per token per layer, but prefill
>   needs NVLink: communication is 2.5% of compute there and 24% over PCIe.
> - In decode, 128 all-reduces cost about 1.02 ms of fixed latency, so 2-way
>   decode is 1.91x faster, not 2x.

One A100 80GB holds this model in bfloat16 with about 19 GiB left for cache.
That's enough to serve, and not enough to serve well: at 64 KiB per token, 19
GiB is about 311,000 tokens of KV cache, which at a batch of 32 is under 10,000
tokens of context each.

Two GPUs change both numbers. Each holds 26.9 GB of weights instead of 53.8,
which frees another 25 GiB per card for cache. And decode, which is memory
bound, gets faster in proportion: each GPU reads half the weights, so the
weight-read floor drops from 42.2 ms to 21.1 ms per token.

Tensor parallelism is how you get that. It splits individual matrices across
devices, so every GPU runs every layer on a slice of the work. This chapter
derives the sharding from block matrix multiplication, counts the communication
it costs, and finds the point where the interconnect stops paying.

## Before you start

**Block matrix multiplication.** The whole chapter rests on one identity. If you
partition $W$ into column blocks, the product partitions the same way. If you
partition $W$ into row blocks and $X$ into matching column blocks, the product
becomes a sum. The next two sections derive both.

**The PyTorch weight convention, which is the opposite of the maths.** An
`nn.Linear` with `in_features` inputs and `out_features` outputs stores a weight
of shape `(out_features, in_features)` and computes `x @ weight.T`. The maths in
this chapter uses $Y = XW$ with
$W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, the code uses the
PyTorch layout, and each listing says which.

> [!WARNING] Column parallel is `dim=0` in PyTorch
> Because PyTorch stores $W$ transposed, a *column*-parallel split, column in
> the mathematical sense, is `weight.chunk(P, dim=0)`. This is the most common
> source of confusion in this chapter.

**Four collective operations.** Every rank holds a tensor of the same shape, and
each collective leaves it with something different:

| Collective | What each rank ends up with | Where it appears in this chapter |
|---|---|---|
| All-reduce | The elementwise sum over all ranks, replicated | Closing every row-parallel layer |
| All-gather | The concatenation of all ranks' tensors | Rebuilding a column-parallel output, which you avoid |
| Reduce-scatter | Its own slice of the elementwise sum | The first phase of a ring all-reduce |
| Broadcast | A copy of one designated rank's tensor | Sending the sampled token id |

A ring all-reduce is a reduce-scatter followed by an all-gather. Each phase
moves $\hlc{\tfrac{P-1}{P}}$ of the tensor per rank, so one all-reduce moves
the following number of bytes per rank:

$$
2\,\hlc{\frac{P-1}{P}}\,\hld{S}
$$

Here $\hld{S}$ is the tensor size in bytes and $P$ is the tensor-parallel
degree. That ring factor appears in every communication estimate in this
chapter.

**The geometry from chapter 2.** Hidden 5120, intermediate 17408, 24 query
heads, 4 KV heads, head dimension 256, and 64 layers, of which 16 are full
attention and 48 are gated delta linear attention with 48 value heads and 16
key heads.

## Column-parallel linear layers

A column-parallel layer splits the output and needs no communication. Write a
linear layer as $Y = XW$ with $X \in \mathbb{R}^{T \times d_{\text{in}}}$ for
$T$ tokens and $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$.

Partition $W$ into $P$ column blocks, each
$d_{\text{in}} \times (d_{\text{out}}/P)$:

$$
W = \begin{bmatrix} \hla{W_1} & \hla{W_2} & \cdots & \hla{W_P} \end{bmatrix}
$$

The definition of matrix multiplication gives the product block by block:

$$
\boxed{XW = \begin{bmatrix} X\hla{W_1} & X\hla{W_2} & \cdots & X\hla{W_P} \end{bmatrix}}
$$

Rank $p$ holds the shard $\hla{W_p}$, receives the full $X$, and computes
$Y_p = X\hla{W_p}$ of shape $(T, d_{\text{out}}/P)$.

**Input replicated, output sharded, no communication.** Each output column
depends on the full input but only on its own column of $W$, so nothing crosses
the network. To reconstruct the full $Y$ you'd all-gather, but as the next two
sections show, you usually don't want to.

```python
def split_column_parallel(weight, world_size):
    # weight is (out_features, in_features): PyTorch stores W transposed,
    # so the mathematical column split is dim 0 here.
    return list(weight.chunk(world_size, dim=0))

def column_parallel_forward(x, shards):
    # x: (T, in_features) replicated. Returns world_size tensors of
    # (T, out_features // world_size).
    return [x @ shard.T for shard in shards]
```

## Row-parallel linear layers

A row-parallel layer splits the input and ends in one all-reduce. Partition $W$
into $P$ row blocks, each $(d_{\text{in}}/P) \times d_{\text{out}}$, and $X$
into matching column blocks:

$$
W = \begin{bmatrix} \hla{W_1} \\ \hla{W_2} \\ \vdots \\ \hla{W_P} \end{bmatrix},
\qquad
X = \begin{bmatrix} \hlb{X_1} & \hlb{X_2} & \cdots & \hlb{X_P} \end{bmatrix}
$$

Expand the product and group the sum over $d_{\text{in}}$ by block:

$$
Y_{tj} = \sum_{k=1}^{d_{\text{in}}} X_{tk} W_{kj}
= \sum_{p=1}^{P} \sum_{k \in \text{block } p} X_{tk} W_{kj}
$$

The inner sum is one block's product, so the whole thing is a sum of block
products:

$$
\boxed{XW = \sum_{p=1}^{P} \hlb{X_p}\,\hla{W_p}}
$$

Rank $p$ holds $\hla{W_p}$, receives only the input shard $\hlb{X_p}$, and
computes a *partial sum* $Y^{(p)} = \hlb{X_p}\hla{W_p}$ of the full output shape
$(T, d_{\text{out}})$.

**Input sharded, output a partial sum, one all-reduce.** Every rank's result has
the right shape and the wrong value. Summing them across ranks is an all-reduce.
The partial sums ==must be added, never concatenated==. Concatenating gives a
tensor $P$ times too wide, which at least raises a shape error somewhere
downstream.

```python
def split_row_parallel(weight, world_size):
    # weight is (out_features, in_features): the mathematical row split
    # is the input dimension, which is dim 1 here.
    return list(weight.chunk(world_size, dim=1))

def row_parallel_forward(x_shards, shards):
    partials = [x @ shard.T for x, shard in zip(x_shards, shards)]
    return sum(partials)          # this sum is the all-reduce
```

## The pairing, and why one all-reduce is enough

Column parallel followed by row parallel needs only one collective. A
column-parallel layer produces an output sharded along its *output* dimension,
a row-parallel layer wants an input sharded along its *input* dimension, and in
a two-layer stack ==those are the same axis==.

$$
X \xrightarrow{\text{column parallel}} \hlb{H_p}
\xrightarrow{\text{row parallel}} Y^{(p)}
\xrightarrow{\text{all-reduce}} Y
$$

The sharded intermediate $\hlb{H_p}$ is never gathered. It goes straight into
the next layer in the form that layer already wants.

Compare that against the naive scheme, which all-gathers after the
column-parallel layer to rebuild the full intermediate, then all-reduces after
the row-parallel layer:

$$
\underbrace{\hlc{\frac{P-1}{P}}S_H}_{\text{all-gather}}
+ \underbrace{2\,\hlc{\frac{P-1}{P}}S_Y}_{\text{all-reduce}}
$$

The pairing costs only the second term, $2\hlc{\frac{P-1}{P}}S_Y$. In the MLP
the intermediate is 17408 wide against a hidden size of 5120, so
$S_H = 3.4\,S_Y$, and the naive scheme moves 2.7 times the bytes. It also costs
a second collective launch per sub-block, which during decode is pure latency
and matters more than the bytes.

> [!KEY] One all-reduce per sub-block
> That's two per transformer layer and 128 per forward pass through 64 layers.
> It's the whole communication cost of tensor parallelism, and it's why the
> technique scales well inside a node.

## The MLP shards by intermediate dimension

The MLP shards cleanly because its activation acts on each intermediate channel
independently. The SwiGLU block is the following:

$$
Y = \big(\operatorname{silu}(XW_g) \odot XW_u\big) W_d
$$

Here $W_g, W_u \in \mathbb{R}^{5120 \times 17408}$ and
$W_d \in \mathbb{R}^{17408 \times 5120}$.

Shard $W_g$ and $W_u$ column-wise into $P$ blocks of width $17408/P$. Rank $p$
computes its slice of the intermediate:

$$
\hlb{H_p} = \operatorname{silu}(X\hla{W_{g,p}}) \odot X\hla{W_{u,p}}
$$

The elementwise step makes this legal. Both $\operatorname{silu}$ and $\odot$
act independently on each intermediate channel, so taking a block commutes with
the activation:

$$
\big(\operatorname{silu}(XW_g) \odot XW_u\big)_{\text{block } p}
= \operatorname{silu}(X\hla{W_{g,p}}) \odot X\hla{W_{u,p}}
$$

So $\hlb{H_p}$ is exactly the $p$-th column block of the full intermediate $H$,
==computed without ever forming $H$==. Had the activation mixed intermediate
channels, as a softmax or a norm over that axis would, the partition would need
a collective in the middle, and the scheme would be far less attractive.

Now shard $W_d$ row-wise into blocks of height $17408/P$. By the row-parallel
identity, the output is a sum of per-rank products:

$$
\boxed{H W_d = \sum_{p=1}^{P} \hlb{H_p}\,\hla{W_{d,p}}}
$$

$\hlb{H_p}$ is already sitting on rank $p$, so the block ends in one all-reduce
of a $(T, 5120)$ tensor.

The following table shows the shapes at $P = 2$, for $T$ tokens:

| Tensor | Unsharded | Per rank |
|---|---|---|
| $W_g$, $W_u$ | `(17408, 5120)` | `(8704, 5120)` |
| $W_d$ | `(5120, 17408)` | `(5120, 8704)` |
| $XW_g$, $XW_u$ | `(T, 17408)` | `(T, 8704)` |
| $H$ | `(T, 17408)` | `(T, 8704)` |
| Output | `(T, 5120)` | `(T, 5120)` partial sum |

```python
def parallel_mlp(x, gate_w, up_w, down_w, world_size):
    gate_shards = gate_w.chunk(world_size, dim=0)   # column parallel
    up_shards = up_w.chunk(world_size, dim=0)       # column parallel
    down_shards = down_w.chunk(world_size, dim=1)   # row parallel
    out = None
    for g, u, d in zip(gate_shards, up_shards, down_shards):
        h = F.silu(x @ g.T) * (x @ u.T)             # (T, 17408 // P)
        partial = h @ d.T                           # (T, 5120)
        out = partial if out is None else out + partial
    return out                                      # the all-reduce
```

## Attention shards by head

Attention isn't one matmul, so the argument is different, and stronger. For
query head $h$ with its KV group $g(h) = \lfloor h/6 \rfloor$, since there are
24 query heads over 4 KV heads, the head's output is the following:

$$
O_h = \operatorname{softmax}\!\left(
\frac{Q_h K_{g(h)}^{\top}}{\sqrt{256}} + M \right) V_{g(h)}
$$

> [!INTUITION] Every head is its own problem
> No term on the right couples head $h$ to any other head. The MLP's partition
> commutes with one nonlinearity; attention goes further and decomposes into
> independent per-head problems.

So partition by head:

- $W_Q \in \mathbb{R}^{5120 \times 6144}$, where $6144 = 24 \times 256$, is
  column parallel with the split falling on head boundaries.
- $W_K, W_V \in \mathbb{R}^{5120 \times 1024}$, where $1024 = 4 \times 256$, are
  column parallel with the split on KV head boundaries.
- $W_O \in \mathbb{R}^{6144 \times 5120}$ takes the concatenated heads, so its
  rows partition along the same head boundaries. It's row parallel.

Rank $p$ owns whole heads, computes their attention with no communication at
all, and owns their KV cache outright. One all-reduce follows $W_O$.

**Why the boundary must fall on a head.** The softmax reduces over the key axis
*within* a head. Splitting one head across two ranks would require exchanging
the running maximum and the running sum of exponentials: the online softmax from
[chapter 14](/c/14-flash-attention), but over a network instead of shared
memory. That's two extra collectives per head per layer, and it's never worth
it.

### The head-divisibility constraint

Rank $p$ must own every query head that belongs to any KV head it owns, or it
would fetch that KV head's keys and values from another rank on every step. So
the shard boundary must fall on a KV head boundary:

$$
P \mid \text{num\_kv\_heads} = 4
$$

> [!EXAMPLE] Splitting 24 query heads and 4 KV heads
> | Degree $P$ | Query heads per rank | KV heads per rank |
> |---|---|---|
> | 2 | 12 | 2 |
> | 4 | 6 | 1 |
> | 8 | — | — |
>
> At $P = 8$ there's no valid split: 4 KV heads can't divide among 8 ranks.

The usual workaround at $P = 8$ is to replicate KV heads, so two ranks hold a
copy of the same KV head. That doubles the aggregate KV cache from 64 KiB to 128
KiB per token and ==gives back exactly the memory tensor parallelism was
supposed to buy==. The lab raises `ValueError` rather than allowing it.

## The hybrid model's linear-attention layers

Only 16 of the 64 layers are full attention. The other 48 run the gated delta
rule from [chapter 6](/c/06-gated-delta-rule), and they have their own head
structure: 16 key heads and 48 value heads, with key and value head dimension
128.

The delta rule is independent per value head. Each value head $h$ carries its
own state $S^{(h)} \in \mathbb{R}^{128 \times 128}$, its own forget gate, and its
own step size, and its update touches no other value head's state.

> [!DEEPDIVE] The per-head delta-rule update
> Head $h$ has forget gate $\alpha^{(h)}_t \in (0,1)$ and step size
> $\beta^{(h)}_t$, chapter 6's $a_t$ and $b_t$. The 48 value heads share 16 key
> heads in groups of three; write $k^{(g)}_t$ for the key of the key head $g$
> that serves this value head. The update is as follows:
>
> $$
> S^{(h)}_t = \alpha^{(h)}_t S^{(h)}_{t-1}
> + \beta^{(h)}_t\, k^{(g)}_t \big(v^{(h)}_t
> - \alpha^{(h)}_t\, {S^{(h)}_{t-1}}^{\!\top} k^{(g)}_t\big)^{\!\top}
> $$
>
> Every term belongs to head $h$ or to its own key head.

So these layers shard by value head exactly as attention shards by query head.
Rank $p$ owns $48/P$ value heads, the projections that produce their $q$, $k$,
and $v$ are column parallel, `out_proj` is row parallel, and one all-reduce
closes the sub-block.

### The recurrent state shards with the heads

This part has no analogue in a pure transformer. The state isn't a cache you
can recompute; it's the layer's entire memory of the sequence, 147.8 MiB per
sequence across all 48 layers. Because it's indexed by value head, it
partitions with the heads and ==never has to move==:

| Degree $P$ | Value heads per rank | State per sequence per rank |
|---|---|---|
| 1 | 48 | 147.8 MiB |
| 2 | 24 | 73.9 MiB |
| 4 | 12 | 36.9 MiB |

No collective touches the state at any point. Only that rank's heads read and
write a rank's state.

### The binding constraint is the greatest common divisor

Every head count in the model must divide evenly: 24 query heads, 4 KV heads, 48
linear value heads, and 16 linear key heads. So the degree must divide their
greatest common divisor:

$$
\boxed{P \mid \gcd(24,\, 4,\, 48,\, 16) = 4}
$$

> [!KEY] This model supports degrees 1, 2, and 4, and nothing above
> The 4 KV heads of the full-attention layers are the binding term; everything
> else would allow 8 or 16. That's a concrete consequence of grouped-query
> attention with a small group count, and it's worth checking on any model
> before you plan a deployment.

## Communication volume

Per-rank traffic grows with $P$ but levels off at a ceiling. Per token, per
all-reduce, the tensor is the hidden state: 5120 elements at 2 bytes, so
$\hld{S} = 10240$ bytes = 10 KiB. With two all-reduces per layer, the per-rank
traffic per token per layer is the following:

$$
\boxed{B(P) = 2 \times 2\,\hlc{\frac{P-1}{P}} \times \hld{10240}
= 40960\,\hlc{\frac{P-1}{P}} \text{ bytes}}
$$

| $P$ | Ring factor $\hlc{(P-1)/P}$ | $B(P)$ per token per layer |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 0.5 | 20.0 KiB |
| 4 | 0.75 | 30.0 KiB |
| 8 | 0.875 | 35.0 KiB |
| $\infty$ | 1 | 40.0 KiB |

No matter how many ranks you add, ==per-rank traffic never exceeds 40 KiB per
token per layer==. Tensor parallelism doesn't have a communication blow-up; it
has a latency problem, which is a different thing.

For a whole forward pass over 64 layers at $T$ tokens, the total is
$B_{\text{total}} = 64\,T\,B(P)$.

> [!EXAMPLE] A 4096-token prefill at $P = 2$
> $$
> 64 \times 4096 \times 20480 = 5.37 \times 10^{9} \text{ bytes} = 5.37 \text{ GB}
> $$

### Where it stops paying, in prefill

Prefill tolerates NVLink and suffers on PCIe. NVLink between two A100s runs at
600 GB/s, and PCIe Gen4 x16 runs at about 64 GB/s, so moving 5.37 GB takes the
following:

$$
t_{\text{NVLink}} = \frac{5.37}{600} = 8.95 \text{ ms},
\qquad
t_{\text{PCIe}} = \frac{5.37}{64} = 83.9 \text{ ms}
$$

Now the compute it overlaps with. A prefill of 4096 tokens does roughly
$2 \times 26.9 \times 10^{9} \times 4096 = 2.20 \times 10^{14}$ FLOPs, split
across two GPUs at 312 TFLOP/s each:

$$
t_{\text{compute}} = \frac{2.20 \times 10^{14}}{2 \times 312 \times 10^{12}}
= 353 \text{ ms}
$$

So communication takes 2.5% as long as the compute over NVLink, and 24% as long
over PCIe. That's the whole argument for tensor parallelism as a within-node
technique: the same sharding that costs nothing over NVLink ==costs a quarter
of your prefill over PCIe==.

### Where it stops paying, in decode

At decode, fixed latency dominates, not bandwidth. The tensors are tiny, one
token of 10 KiB, and each all-reduce costs 5 to 10 microseconds of latency
regardless of size. There are 128 of them per step:

$$
t_{\text{comm}} \approx 128 \times 8\,\mu\text{s} = 1.02 \text{ ms}
$$

That figure barely changes with $P$. The weight read does:

$$
t_{\text{weights}}(P) = \frac{53.8 \times 10^{9}}{P \times 1275 \times 10^{9}}
= \frac{42.2}{P} \text{ ms}
$$

Adding the two gives the step time:

| $P$ | Weight read | Comm | Step | Speedup | Efficiency |
|---|---|---|---|---|---|
| 1 | 42.2 ms | 0 | 42.2 ms | 1.00 | 100% |
| 2 | 21.1 ms | 1.02 ms | 22.1 ms | 1.91 | 96% |
| 4 | 10.6 ms | 1.02 ms | 11.6 ms | 3.64 | 91% |
| 8 | 5.3 ms | 1.02 ms | 6.3 ms | 6.70 | 84% |
| 16 | 2.6 ms | 1.02 ms | 3.6 ms | 11.7 | 73% |

Decode latency improves sublinearly, and the loss compounds. The fixed 1.02 ms
is 2.4% of the single-GPU weight read but 28% of the step at $P = 16$. The two
terms are equal at $42.2 / P = 1.02$, or $P = 41$, but efficiency is already
down to 73% long before that. On this model the question
is academic, because $P \le 4$ is the architectural limit anyway.

> [!NOTE] CUDA graphs shrink the fixed term
> CUDA graphs collapse the launch overhead of a whole decode step, collectives
> included, into one replay. That's a large part of why production engines
> capture decode. This course's engine, so far, doesn't.

## What stays replicated

Not everything shards, and the reasons differ for each piece.

**Norms are replicated, and must be.** RMSNorm reduces over the *entire* hidden
dimension:

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{k=1}^{d} x_k^2 + \epsilon}
$$

A sharded $x$ would need a scalar all-reduce per token to compute the
denominator. The all-reduce at the end of the previous sub-block already leaves
$x$ replicated, so keeping norms replicated saves that collective and costs
almost no memory. A norm weight is $5120 \times 2 = 10$ KiB, so every norm in
the model together is a couple of megabytes.

**The input embedding can shard by vocabulary.** Replicated, it costs the
following on every rank:

$$
248320 \times 5120 \times 2 = 2.54 \times 10^{9} \text{ bytes} = 2.54 \text{ GB}
$$

Sharded, rank $p$ holds the rows for vocabulary ids in $[pV/P, (p+1)V/P)$, looks
up the ids that fall in its range, and writes zeros for the rest, and then the
ranks all-reduce. That's one extra all-reduce per forward pass, one more out of
129, in exchange for $2.54(1 - 1/P)$ GB per rank. Replication is simpler, and
the lab doesn't require either.

**The output projection shards by vocabulary and isn't gathered.** It maps
$5120 \to 248320$, so it's naturally column parallel. But gathering its output
costs $248320 \times 2 = 497$ KB per token, 48 times the hidden-state
all-reduce, which would dwarf everything else in this chapter.

The fix is to keep the logits sharded and sample vocab-parallel. The softmax
needs only two reductions over the vocabulary, a maximum and a sum of
exponentials. Each rank computes those locally, the ranks all-reduce two scalars
per sequence, and the chosen token id is broadcast. Communication drops from 497
KB per token to a handful of bytes.

## Tensor, pipeline, and data parallelism

The three ways to use more than one GPU answer different questions:

| Strategy | Splits | Communication at inference | What it improves |
|---|---|---|---|
| Data parallelism | The batch; the model is replicated | None | Throughput only |
| Pipeline parallelism | The layers | One point-to-point send per stage boundary | Memory per GPU, not latency |
| Tensor parallelism | Every matrix | 128 all-reduces per step | Memory per GPU and decode latency |

**Data parallelism** needs no communication at inference, because there are no
gradients. But every GPU needs all 53.8 GB, so it does nothing for a model that
barely fits or for latency: each GPU still reads 53.8 GB per decode step.

**Pipeline parallelism** puts layers 0 to 31 on GPU 0 and 32 to 63 on GPU 1. It
sends one hidden state, $T \times 5120 \times 2$ bytes, at each stage boundary,
$P-1$ times per forward pass rather than $2L = 128$ times. At $T = 4096$ and
$P = 2$, that's 42 MB sent once, against 5.37 GB for tensor parallelism: 128
times less traffic, and point-to-point rather than a collective. Pipeline
parallelism therefore tolerates PCIe, or even Ethernet between nodes.

Its cost is the bubble. During decode there's one token in flight, so GPU 1
idles while GPU 0 works and vice versa: latency is unchanged and utilization is
$1/P$. You fill the bubble with micro-batches, which requires many concurrent
requests, which is exactly what you don't have at low load.

**Tensor parallelism** is the only one of the three that reduces single-request
decode latency, because it's the only one that reduces the bytes each GPU reads
per token. It pays for that with 128 collectives per step and a hard
requirement on the interconnect.

The standard arrangement follows: ==tensor parallelism within a node over
NVLink==, pipeline parallelism across nodes, and data parallelism across
replicas.

## Test without two GPUs

You don't need two GPUs to check the partitioning. Simulate every rank on one
device: split the weights, run each shard in turn, and combine, concatenating
for a column-parallel output and summing for a row-parallel one. The result must
match the unsharded layer, up to floating-point reduction order.

That test catches every partitioning bug: wrong split axis, wrong concatenation
order, a sum where a concatenation belongs, and a missing all-reduce. It doesn't
catch NCCL configuration, device placement, or the real latency, all of which
need real devices.

> [!TIP] Compare with a tolerance, not bitwise
> Summing $P$ partial products in a different order from the unsharded matmul
> changes the floating-point result. Demand agreement to about $10^{-3}$ in
> bfloat16 or $10^{-4}$ in float32. The lab uses $10^{-3}$ for the MLP and
> $10^{-4}$ for attention.

## What goes wrong

**Concatenating row-parallel partial sums instead of adding them.** The output is
$P$ times too wide. This one at least raises a shape error downstream.

**Concatenating column-parallel outputs in the wrong rank order.** No shape
error, no exception, silently wrong results. Always concatenate in rank order.

**Splitting a projection by raw output feature instead of by head.** With 24
heads of 256 and $P = 2$, the feature split at 3072 happens to land on head 12,
so it works by luck. At $P = 16$ each rank would get 384 features, one and a
half heads, and the split cuts a head in two, which produces wrong attention
with no error anywhere. Compute the split in heads, then multiply by `head_dim`.

**Adding a row-parallel layer's bias on every rank.** After the all-reduce, the
bias has been added $P$ times. Add it after the reduction, or on one rank only.

**Sharding a norm.** The reduction is over the full hidden dimension, so a
sharded norm silently normalizes by the wrong denominator. The output is scaled
by roughly $\sqrt{P}$, and the model produces fluent nonsense.

**Sampling with a different RNG state on each rank.** Every rank picks a
different next token, and the sequences diverge immediately. Seed identically,
or sample on one rank and broadcast.

> [!RECAP]
> - Column parallel splits the output dimension (`dim=0` in PyTorch) with no
>   communication; row parallel splits the input dimension (`dim=1`) and ends in
>   an all-reduce that adds partial sums.
> - Column then row costs one all-reduce per sub-block, 128 per forward pass.
> - Attention shards by head, the MLP by intermediate channel, and the
>   linear-attention layers by value head, with their recurrent state staying
>   put. This model needs $P \mid \gcd(24, 4, 48, 16) = 4$.
> - Per-rank traffic is $40960\,\hlc{\tfrac{P-1}{P}}$ bytes per token per layer,
>   capped at 40 KiB. Prefill needs NVLink, not PCIe.
> - Decode pays about 1.02 ms of fixed all-reduce latency per step, so speedup
>   is sublinear: 1.91x at $P = 2$.

## Check your understanding

> [!QUESTION] Why does pairing a column-parallel layer with a row-parallel layer need one all-reduce rather than two?
> Because the column-parallel layer's output is already sharded along the axis
> the row-parallel layer wants its input sharded along. The intermediate never
> needs to be reconstructed. Reversing the pairing, row parallel then column
> parallel, would need an all-reduce to complete the first layer's partial sums
> and then a broadcast or all-gather to feed the second, so the order isn't
> arbitrary.

> [!QUESTION] 8-way tensor parallelism would give each rank 6.7 GB of weights. Why does this model refuse it?
> The full-attention layers have 4 KV heads, and 4 doesn't divide by 8. Every
> query head must sit on the same rank as its KV head, so there's no valid split.
> Replicating KV heads works mechanically but doubles the aggregate KV cache from
> 64 KiB to 128 KiB per token, which cancels the memory saving. Across the whole
> model the constraint is $P \mid \gcd(24, 4, 48, 16) = 4$.

> [!QUESTION] Two-way tensor parallelism halves the weight bytes each GPU reads. Why is decode only 1.91 times faster, not 2?
> The 128 all-reduces per step each cost 5 to 10 microseconds of launch latency
> regardless of how little data they carry, so roughly 1.02 ms per step is fixed
> overhead that doesn't shrink with $P$. The weight read falls from 42.2 ms to
> 21.1 ms, but the step is $21.1 + 1.02 = 22.1$ ms. At higher degrees the fixed
> term is a larger fraction and efficiency keeps falling: 91% at $P = 4$, 73% at
> $P = 16$.

> [!QUESTION] Where does the recurrent state go when you shard the linear-attention layers?
> It shards with the value heads and never moves. Each of the 48 value heads owns
> a $128 \times 128$ state matrix that only its own head reads and writes, so at
> $P = 2$ each rank holds 24 heads' worth, 73.9 MiB of the 147.8 MiB per
> sequence, and no collective ever touches it. That's a genuine advantage over a
> KV cache, which at least has to be sized and paged.

## Lab

> [!TRY]
> Build the sharding primitives, a tensor-parallel SwiGLU block, a head splitter
> that refuses invalid degrees, and an all-reduce byte counter. You pass when
> every sharded result matches its unsharded reference and the head and byte
> counts come out as this chapter derives.

Implement `split_column_parallel` and `split_row_parallel` on PyTorch-layout
weights, `column_parallel_forward` returning one output shard per rank, and
`row_parallel_forward` summing the partial sums. Then build `parallel_mlp` as a
tensor-parallel SwiGLU block with gate and up column parallel, down row
parallel, and the activation running independently on each shard.

Add `split_attention_heads`, which returns query heads and KV heads per rank and
raises `ValueError` when the KV heads don't divide evenly, and
`all_reduce_bytes`, which applies the $2(P-1)/P$ ring factor and returns 0 at
$P = 1$.

The harness checks that the shards reassemble into the original weight, that
the column-parallel outputs concatenate to the unsharded result and the
row-parallel partial sums add to it, and that the parallel MLP matches the
unsharded block to $10^{-3}$ at 1-way, 2-way, and 4-way. It also checks that
splitting attention by head reproduces the unsharded output to $10^{-4}$, that
24 and 4 heads give $(12, 2)$ at $P = 2$ and $(6, 1)$ at $P = 4$ while $P = 8$
raises, and that 2-way all-reduce moves exactly one tensor's worth per rank
while 4-way moves more but stays under twice the tensor size.

## Further reading

- [Megatron-LM: training multi-billion parameter language models using model parallelism](https://arxiv.org/abs/1909.08053)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102)
- [Reducing activation recomputation in large transformer models](https://arxiv.org/abs/2205.05198) — sequence parallelism, which shards the norms tensor parallelism leaves replicated.
- [NCCL: collective communication primitives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
