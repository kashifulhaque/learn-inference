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

One A100 80GB holds this model in bfloat16 with about 19 GiB left for cache.
That is enough to serve, and not enough to serve well: at 64 KiB per token, 19
GiB is about 311,000 tokens of KV cache, which at a batch of 32 is under 10,000
tokens of context each.

Two GPUs change both numbers. Each holds 26.9 GB of weights instead of 53.8,
which frees another 25 GiB per card for cache. And decode, which is memory bound,
gets faster in proportion: each GPU reads half the weights, so the weight-read
floor drops from 42.2 ms to 21.1 ms per token.

Tensor parallelism is how you get that. It splits individual matrices across
devices, so every GPU runs every layer on a slice of the work. This chapter
derives the sharding from block matrix multiplication, counts the
communication it costs, and finds the point where the interconnect stops paying.

## Before you start

**Block matrix multiplication.** The whole chapter rests on one identity. If you
partition $W$ into column blocks, the product partitions the same way; if you
partition $W$ into row blocks and $X$ into matching column blocks, the product
becomes a sum. Both are derived below, and nothing else is needed.

**The PyTorch weight convention, which is the opposite of the maths.** An
`nn.Linear` with `in_features` inputs and `out_features` outputs stores a weight
of shape `(out_features, in_features)` and computes `x @ weight.T`. So the
"output dimension" is dimension 0 of the stored tensor, and a *column*-parallel
split — column in the mathematical sense — is `weight.chunk(P, dim=0)`. This is
the single most common source of confusion in this chapter. Throughout, the
maths uses $Y = XW$ with $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$
and the code uses the PyTorch layout, and each listing says which.

**Four collective operations.** Every rank holds a tensor of the same shape.

| Collective | What each rank ends up with |
|---|---|
| All-reduce | The elementwise sum over all ranks, replicated |
| All-gather | The concatenation of all ranks' tensors |
| Reduce-scatter | Its own slice of the elementwise sum |
| Broadcast | A copy of one designated rank's tensor |

A ring all-reduce is a reduce-scatter followed by an all-gather. Each phase
moves $(P-1)/P$ of the tensor per rank, so one all-reduce moves

$$
2\,\frac{P-1}{P}\,S
$$

bytes per rank, where $S$ is the tensor size in bytes and $P$ is the
tensor-parallel degree. That factor appears in every communication estimate
below.

**The geometry from chapter 2.** Hidden 5120, intermediate 17408, 24 query
heads, 4 KV heads, head dimension 256, 64 layers of which 16 are full attention
and 48 are gated delta linear attention with 48 value heads and 16 key heads.

## Column-parallel linear layers

Write a linear layer as $Y = XW$ with $X \in \mathbb{R}^{T \times d_{\text{in}}}$
for $T$ tokens and $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$.

Partition $W$ into $P$ column blocks, each $d_{\text{in}} \times
(d_{\text{out}}/P)$:

$$
W = \begin{bmatrix} W_1 & W_2 & \cdots & W_P \end{bmatrix}
$$

The definition of matrix multiplication gives, immediately,

$$
XW = \begin{bmatrix} XW_1 & XW_2 & \cdots & XW_P \end{bmatrix}
$$

Rank $p$ holds $W_p$, receives the full $X$, and computes $Y_p = XW_p$ of shape
$(T, d_{\text{out}}/P)$.

**Input replicated, output sharded, no communication.** Each output column
depends on the full input but only on its own column of $W$, so nothing needs to
cross the network. To reconstruct the full $Y$ you would all-gather, but as the
next two sections show, you usually do not want to.

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

Now partition $W$ into $P$ row blocks, each $(d_{\text{in}}/P) \times
d_{\text{out}}$, and $X$ into matching column blocks:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_P \end{bmatrix},
\qquad
X = \begin{bmatrix} X_1 & X_2 & \cdots & X_P \end{bmatrix}
$$

Expanding the product and grouping the sum over $d_{\text{in}}$ by block:

$$
Y_{tj} = \sum_{k=1}^{d_{\text{in}}} X_{tk} W_{kj}
= \sum_{p=1}^{P} \sum_{k \in \text{block } p} X_{tk} W_{kj}
$$

which is exactly

$$
XW = \sum_{p=1}^{P} X_p W_p
$$

Rank $p$ holds $W_p$, receives only $X_p$, and computes a *partial sum*
$Y^{(p)} = X_p W_p$ of the full output shape $(T, d_{\text{out}})$.

**Input sharded, output a partial sum, one all-reduce.** Every rank's result has
the right shape and the wrong value. Summing them across ranks is an all-reduce.

The failure mode here is worth naming now: the partial sums must be **added**,
never concatenated. Concatenating gives a tensor $P$ times too wide, which at
least raises a shape error somewhere downstream.

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

Put the two back to back. A column-parallel layer produces an output sharded
along its *output* dimension. A row-parallel layer wants an input sharded along
its *input* dimension. In a two-layer stack those are the same axis.

$$
X \xrightarrow{\text{column parallel}} H_p
\xrightarrow{\text{row parallel}} Y^{(p)}
\xrightarrow{\text{all-reduce}} Y
$$

The sharded intermediate never has to be gathered. It goes straight into the
next layer in the form the next layer already wants.

Compare against the naive scheme, which all-gathers after the column-parallel
layer to rebuild the full intermediate, then all-reduces after the row-parallel
layer. That costs

$$
\underbrace{\frac{P-1}{P}S_H}_{\text{all-gather}}
+ \underbrace{2\,\frac{P-1}{P}S_Y}_{\text{all-reduce}}
$$

against $2\frac{P-1}{P}S_Y$ for the pairing. With $S_H$ larger than $S_Y$ — in
the MLP the intermediate is 17408 wide against a hidden size of 5120, so
$S_H = 3.4\,S_Y$ — the naive scheme moves 2.7 times the bytes. It also costs a
second collective launch per sub-block, which during decode is pure latency and
matters more than the bytes.

So: **one all-reduce per sub-block**, two per transformer layer, 128 per forward
pass through 64 layers. That is the whole communication cost of tensor
parallelism, and it is why the technique scales well inside a node.

## The MLP shards by intermediate dimension

The SwiGLU block is

$$
Y = \big(\operatorname{silu}(XW_g) \odot XW_u\big) W_d
$$

with $W_g, W_u \in \mathbb{R}^{5120 \times 17408}$ and
$W_d \in \mathbb{R}^{17408 \times 5120}$.

Shard $W_g$ and $W_u$ column-wise into $P$ blocks of width $17408/P$. Rank $p$
computes

$$
H_p = \operatorname{silu}(XW_{g,p}) \odot XW_{u,p}
$$

The step that makes this legal is the elementwise one. Both $\operatorname{silu}$
and $\odot$ act independently on each intermediate channel, so

$$
\big(\operatorname{silu}(XW_g) \odot XW_u\big)_{\text{block } p}
= \operatorname{silu}(XW_{g,p}) \odot XW_{u,p}
$$

That is, $H_p$ is exactly the $p$-th column block of the full intermediate $H$,
computed without ever forming $H$. Had the activation mixed intermediate
channels — a softmax or a norm over that axis — the partition would need a
collective in the middle, and the whole scheme would be far less attractive.

Now shard $W_d$ row-wise into blocks of height $17408/P$. By the row-parallel
identity,

$$
H W_d = \sum_{p=1}^{P} H_p W_{d,p}
$$

and $H_p$ is already sitting on rank $p$. One all-reduce of a $(T, 5120)$
tensor.

Shapes at $P = 2$, for $T$ tokens:

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

Attention is not one matmul, so the argument is different — and stronger.

For query head $h$ with its KV group $g(h) = \lfloor h/6 \rfloor$, since there
are 24 query heads over 4 KV heads:

$$
O_h = \operatorname{softmax}\!\left(
\frac{Q_h K_{g(h)}^{\top}}{\sqrt{256}} + M \right) V_{g(h)}
$$

**No term on the right couples head $h$ to any other head.** The function is
block diagonal across heads. That is a stronger property than the MLP's
elementwise activation: not only does the partition commute with the nonlinearity,
the entire attention computation decomposes into independent per-head problems.

So partition by head:

- $W_Q \in \mathbb{R}^{5120 \times 6144}$, where $6144 = 24 \times 256$, is
  column parallel with the split falling on head boundaries.
- $W_K, W_V \in \mathbb{R}^{5120 \times 1024}$, where $1024 = 4 \times 256$, are
  column parallel with the split on KV head boundaries.
- $W_O \in \mathbb{R}^{6144 \times 5120}$ takes the concatenated heads, so its
  rows partition along exactly the same head boundaries. Row parallel.

Rank $p$ owns whole heads, computes their attention with no communication at
all, and owns their KV cache outright. One all-reduce, after $W_O$.

**Why the boundary must fall on a head.** The softmax reduces over the key axis
*within* a head. Splitting one head across two ranks would require exchanging
the running maximum and the running sum of exponentials — precisely the online
softmax from chapter 14, but over a network instead of over shared memory. That
is two extra collectives per head per layer, and it is never worth it.

### The head-divisibility constraint

Rank $p$ must own every query head belonging to any KV head it owns, or it would
have to fetch that KV head's keys and values from another rank on every step. So
the shard boundary must fall on a KV head boundary, and

$$
P \mid \text{num\_kv\_heads} = 4
$$

At $P = 2$ each rank takes 12 query heads and 2 KV heads. At $P = 4$, 6 query
heads and 1 KV head. At $P = 8$, there is no valid split: 4 KV heads cannot be
divided among 8 ranks. The usual workaround is to replicate KV heads, so two
ranks hold a copy of the same KV head. That doubles the aggregate KV cache from
64 KiB to 128 KiB per token and gives back exactly the memory tensor parallelism
was supposed to buy. The lab raises `ValueError` rather than allowing it.

## The hybrid model complicates this

Only 16 of the 64 layers are full attention. The other 48 run the gated delta
rule from chapter 6, and they have their own head structure: 16 key heads and 48
value heads, with key and value head dimension 128.

The delta rule is independent per value head. Each value head $h$ carries its
own state $S^{(h)} \in \mathbb{R}^{128 \times 128}$, its own forget gate
$\alpha^{(h)}_t \in (0,1)$, and its own step size $\beta^{(h)}_t$ — chapter 6's
$a_t$ and $b_t$. Writing $k^{(g)}_t$ for the key of the key head $g$ that serves
this value head, since 48 value heads share 16 key heads in groups of three, the
update is

$$
S^{(h)}_t = \alpha^{(h)}_t S^{(h)}_{t-1}
+ \beta^{(h)}_t\, k^{(g)}_t \big(v^{(h)}_t
- \alpha^{(h)}_t\, {S^{(h)}_{t-1}}^{\!\top} k^{(g)}_t\big)^{\!\top}
$$

and it touches no other value head's state. So these layers shard by value head
exactly as attention
shards by query head: rank $p$ owns $48/P$ value heads, the projections that
produce their $q$, $k$, $v$ are column parallel, `out_proj` is row parallel, and
one all-reduce closes the sub-block.

**The recurrent state shards with the heads.** This is the part that has no
analogue in a pure transformer. The state is not a cache you can recompute — it
is the layer's entire memory of the sequence, 147.8 MiB per sequence across all
48 layers. Because it is indexed by value head, it partitions with the heads and
never has to move:

| Degree $P$ | Value heads per rank | State per sequence per rank |
|---|---|---|
| 1 | 48 | 147.8 MiB |
| 2 | 24 | 73.9 MiB |
| 4 | 12 | 36.9 MiB |

No collective touches the state at any point. A rank's state is read and written
only by that rank's heads.

**The binding constraint is the greatest common divisor.** Every head count in
the model must divide evenly: 24 query heads, 4 KV heads, 48 linear value heads,
16 linear key heads. So

$$
P \mid \gcd(24,\, 4,\, 48,\, 16) = 4
$$

This model supports tensor-parallel degrees 1, 2, and 4 with no replication, and
nothing above that. The 4 KV heads of the full-attention layers are the binding
term — everything else would allow 8 or 16. That is a concrete architectural
consequence of grouped-query attention with a small group count, and it is worth
checking on any model before planning a deployment.

## Communication volume

Per token, per all-reduce, the tensor is the hidden state: $5120$ elements at 2
bytes, so $S = 10240$ bytes = 10 KiB. With two all-reduces per layer, the
per-rank traffic per token per layer is

$$
B(P) = 2 \times 2\,\frac{P-1}{P} \times 10240
= 40960\,\frac{P-1}{P} \text{ bytes}
$$

| $P$ | $B(P)$ per token per layer |
|---|---|
| 1 | 0 |
| 2 | 20.0 KiB |
| 4 | 30.0 KiB |
| 8 | 35.0 KiB |
| $\infty$ | 40.0 KiB |

The ceiling matters: no matter how many ranks you add, per-rank traffic never
exceeds 40 KiB per token per layer. Tensor parallelism does not have a
communication blow-up; it has a latency problem, which is a different thing.

For a whole forward pass over 64 layers at $T$ tokens:

$$
B_{\text{total}} = 64\,T\,B(P)
$$

At $T = 4096$ and $P = 2$:

$$
64 \times 4096 \times 20480 = 5.37 \times 10^{9} \text{ bytes} = 5.37 \text{ GB}
$$

### Where it stops paying, in prefill

Compare against the interconnect. NVLink between two A100s runs at 600 GB/s;
PCIe Gen4 x16 runs at about 64 GB/s.

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

So communication is 2.5% of the step over NVLink and 24% over PCIe. That is the
whole argument for why tensor parallelism is a within-node technique: the same
sharding that costs nothing over NVLink costs a quarter of your prefill over
PCIe.

### Where it stops paying, in decode

At decode the tensors are tiny — one token, 10 KiB — so bandwidth is irrelevant
and launch latency dominates. Each all-reduce costs 5 to 10 microseconds of
latency regardless of size, and there are 128 of them per step:

$$
t_{\text{comm}} \approx 128 \times 8\,\mu\text{s} = 1.02 \text{ ms}
$$

That figure barely changes with $P$. The weight read does:

$$
t_{\text{weights}}(P) = \frac{53.8 \times 10^{9}}{P \times 1275 \times 10^{9}}
= \frac{42.2}{P} \text{ ms}
$$

Add them:

| $P$ | Weight read | Comm | Step | Speedup | Efficiency |
|---|---|---|---|---|---|
| 1 | 42.2 ms | 0 | 42.2 ms | 1.00 | 100% |
| 2 | 21.1 ms | 1.02 ms | 22.1 ms | 1.91 | 96% |
| 4 | 10.6 ms | 1.02 ms | 11.6 ms | 3.64 | 91% |
| 8 | 5.3 ms | 1.02 ms | 6.3 ms | 6.70 | 84% |
| 16 | 2.6 ms | 1.02 ms | 3.6 ms | 11.7 | 73% |

Decode latency improves sublinearly, and the loss compounds: the fixed 1.02 ms
is 2.4% of the step at $P = 1$ and 28% at $P = 16$. The two terms are equal at

$$
\frac{42.2}{P} = 1.02 \quad\Longrightarrow\quad P = 41
$$

but efficiency is already down to 73% long before that. On this model the
question is academic, because $P \le 4$ is the architectural limit anyway.

CUDA graphs help here, because they collapse the launch overhead of a whole
decode step — collectives included — into one replay. That is a large part of
why production engines capture decode and this course's engine, so far, does
not.

## What stays replicated

Not everything shards, and the reasons differ.

**Norms are replicated, and must be.** RMSNorm computes

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{k=1}^{d} x_k^2 + \epsilon}
$$

which is a reduction over the *entire* hidden dimension. A sharded $x$ would
need a scalar all-reduce per token just to compute the denominator. Since the
all-reduce at the end of the previous sub-block already leaves $x$ replicated,
keeping norms replicated costs one extra collective fewer and almost no memory:
a norm weight is $5120 \times 2 = 10$ KiB, so every norm in the model together
is a couple of megabytes.

**The input embedding can shard by vocabulary.** Replicated, it costs

$$
248320 \times 5120 \times 2 = 2.54 \times 10^{9} \text{ bytes} = 2.54 \text{ GB}
$$

on every rank. Sharded, rank $p$ holds the rows for vocabulary ids in
$[pV/P, (p+1)V/P)$, looks up the ids that fall in its range, writes zeros for
the rest, and the ranks all-reduce. That is one extra all-reduce per forward
pass — one more out of 129 — in exchange for $2.54(1 - 1/P)$ GB per rank.
Replication is simpler and the lab does not require either.

**The output projection should shard by vocabulary and should not be gathered.**
It maps $5120 \to 248320$, so it is naturally column parallel. But gathering its
output costs $248320 \times 2 = 497$ KB per token, 48 times the hidden-state
all-reduce, which would dwarf everything else in this chapter.

The fix is to keep the logits sharded and do the sampling vocab-parallel. The
softmax needs only two reductions over the vocabulary — a maximum and a sum of
exponentials — so each rank computes those locally, the ranks all-reduce two
scalars per sequence, and the chosen token id is broadcast. Communication drops
from 497 KB per token to a handful of bytes.

## Tensor, pipeline, and data parallelism

Three ways to use more than one GPU, and they answer different questions.

**Data parallelism** replicates the whole model on each GPU and splits the batch.
At inference there are no gradients, so there is no communication at all. But
every GPU needs all 53.8 GB, so it does nothing for a model that barely fits,
and it does nothing for latency — each GPU still reads 53.8 GB per decode step.
It multiplies throughput, and only throughput.

**Pipeline parallelism** splits by layer: GPU 0 holds layers 0 to 31, GPU 1
holds 32 to 63. Communication is one point-to-point hidden-state send per stage
boundary, $T \times 5120 \times 2$ bytes, $P-1$ times per forward pass rather
than $2L = 128$ times. At $T = 4096$ and $P = 2$ that is 42 MB sent once, against
5.37 GB for tensor parallelism — 128 times less traffic, and point-to-point
rather than a collective. Pipeline parallelism therefore tolerates PCIe, or even
Ethernet between nodes.

Its cost is the bubble. During decode there is one token in flight, so GPU 1
idles while GPU 0 works and vice versa: latency is unchanged and utilization is
$1/P$. You fill the bubble with micro-batches, which requires many concurrent
requests, which is exactly what you do not have at low load.

**Tensor parallelism** is the only one of the three that reduces single-request
decode latency, because it is the only one that reduces the bytes each GPU must
read per token. It pays for that with 128 collectives per step and a hard
requirement on the interconnect.

The standard arrangement follows: tensor parallelism within a node over NVLink,
pipeline parallelism across nodes, data parallelism across replicas.

## Testing without two GPUs

You do not need two GPUs to check the partitioning. Simulate every rank on one
device: split the weights, run each shard in turn, and combine — concatenate for
a column-parallel output, sum for a row-parallel one. The result must match the
unsharded layer, up to floating-point reduction order.

That test catches every partitioning bug: wrong split axis, wrong concatenation
order, a sum where a concatenation belongs, a missing all-reduce. What it does
not catch is NCCL configuration, device placement, and the real latency, all of
which need real devices.

Tolerance matters. Summing $P$ partial products in a different order from the
unsharded matmul changes the floating-point result, so demand agreement to about
$10^{-3}$ in bfloat16 or $10^{-4}$ in float32, not bitwise equality. The lab
uses $10^{-3}$ for the MLP and $10^{-4}$ for attention.

## What goes wrong

**Concatenating row-parallel partial sums instead of adding them.** The output is
$P$ times too wide. This one at least raises a shape error downstream.

**Concatenating column-parallel outputs in the wrong rank order.** No shape
error, no exception, silently wrong results. Always concatenate in rank order.

**Splitting a projection by raw output feature instead of by head.** With 24
heads of 256 and $P = 2$, the feature split at 3072 happens to land on head 12,
so it works by luck. At $P = 16$ each rank would get 384 features — one and a
half heads — and the split cuts a head in two, which produces wrong attention
with no error anywhere. Compute the split in heads, then multiply by `head_dim`.

**Adding a row-parallel layer's bias on every rank.** After the all-reduce the
bias has been added $P$ times. Add it after the reduction, or on one rank only.

**Sharding a norm.** The reduction is over the full hidden dimension, so a
sharded norm silently normalizes by the wrong denominator: the output is scaled
by roughly $\sqrt{P}$ and the model produces fluent nonsense.

**Sampling with a different RNG state on each rank.** Every rank picks a
different next token and the sequences diverge immediately. Seed identically, or
sample on one rank and broadcast.

## Check your understanding

**Why does pairing a column-parallel layer with a row-parallel layer need one
all-reduce rather than two?**

Because the column-parallel layer's output is already sharded along the axis the
row-parallel layer wants its input sharded along. The intermediate never needs to
be reconstructed. Reversing the pairing — row parallel then column parallel —
would need an all-reduce to complete the first layer's partial sums and then a
broadcast or all-gather to feed the second, so the order is not arbitrary.

**8-way tensor parallelism would give each rank 6.7 GB of weights. Why does this
model refuse it?**

The full-attention layers have 4 KV heads, and 4 does not divide by 8. Every
query head must sit on the same rank as its KV head, so there is no valid split.
Replicating KV heads works mechanically but doubles the aggregate KV cache from
64 KiB to 128 KiB per token, which cancels the memory saving. Across the whole
model the constraint is $P \mid \gcd(24, 4, 48, 16) = 4$.

**Two-way tensor parallelism halves the weight bytes each GPU reads. Why is
decode only 1.91 times faster, not 2?**

The 128 all-reduces per step each cost 5 to 10 microseconds of launch latency
regardless of how little data they carry, so roughly 1.02 ms per step is fixed
overhead that does not shrink with $P$. The weight read falls from 42.2 ms to
21.1 ms, but the step is $21.1 + 1.02 = 22.1$ ms. At higher degrees the fixed
term is a larger fraction and efficiency keeps falling: 91% at $P = 4$, 73% at
$P = 16$.

**Where does the recurrent state go when you shard the linear-attention layers?**

It shards with the value heads and never moves. Each of the 48 value heads owns
a $128 \times 128$ state matrix that only its own head reads and writes, so at
$P = 2$ each rank holds 24 heads' worth — 73.9 MiB of the 147.8 MiB per
sequence — and no collective ever touches it. That is a genuine advantage over a
KV cache, which at least has to be sized and paged.

## Lab

Implement `split_column_parallel` and `split_row_parallel` on PyTorch-layout
weights, `column_parallel_forward` returning one output shard per rank, and
`row_parallel_forward` summing the partial sums. Then build `parallel_mlp` as a
tensor-parallel SwiGLU block with gate and up column parallel, down row
parallel, and the activation running independently on each shard.

Add `split_attention_heads`, which returns query heads and KV heads per rank and
raises `ValueError` when the KV heads do not divide evenly, and
`all_reduce_bytes`, which applies the $2(P-1)/P$ ring factor and returns 0 at
$P = 1$.

The harness checks that the shards reassemble into the original weight, that the
column-parallel outputs concatenate to the unsharded result and the row-parallel
partial sums add to it, that the parallel MLP matches the unsharded block to
$10^{-3}$ at 1-way, 2-way, and 4-way, that splitting attention by head
reproduces the unsharded output to $10^{-4}$, that 24 and 4 heads give
$(12, 2)$ at $P = 2$ and $(6, 1)$ at $P = 4$ while $P = 8$ raises, and that
2-way all-reduce moves exactly one tensor's worth per rank while 4-way moves
more but stays under twice the tensor size.

## Further reading

- [Megatron-LM: training multi-billion parameter language models using model parallelism](https://arxiv.org/abs/1909.08053)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102)
- [Reducing activation recomputation in large transformer models](https://arxiv.org/abs/2205.05198) — sequence parallelism, which shards the norms tensor parallelism leaves replicated.
- [NCCL: collective communication primitives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
