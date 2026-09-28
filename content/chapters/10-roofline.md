---
title: The roofline
slug: 10-roofline
part: "Part 3 — Making it fast"
summary: Deciding whether an operation is limited by arithmetic or by memory, from a picture you can sketch on a napkin to every number that decides it.
minutes: 110
gpu: true
objectives:
  - Sketch the roofline and place an operation on it.
  - Derive the ridge point from a GPU's two peak rates.
  - Derive arithmetic intensity for the MLP, attention, RMSNorm, and the embedding.
  - Say where the "intensity is roughly T" approximation holds and where it fails.
  - Predict whether prefill or decode limits a given workload.
  - Use measured bandwidth to tell a slow kernel from a memory-bound one.
  - Quantify what kernel launch overhead and idle SMs cost a decode step.
lab: 10-roofline
---

# The roofline

> [!TLDR]
> - Every operation waits on one of two things: arithmetic or memory traffic.
>   Whichever takes longer sets a floor under its runtime, and one ratio,
>   FLOPs per byte moved, tells you which it is.
> - The MLP gets more work out of each byte as more tokens share one read of its
>   weights. Generating one token at a time leaves the GPU's arithmetic more than
>   99% idle; a long prompt keeps it busy.
> - Attention during decode waits on memory at every context length and every
>   batch size. RMSNorm and the embedding lookup do even less arithmetic per
>   byte. For all three, the only fix is to move fewer bytes.
> - Judge a memory-bound kernel against the speed a plain copy really reaches on
>   the card, 1275 GB/s, not the 1935 GB/s on the spec sheet.
> - At batch 1, two costs the roofline doesn't model dominate decode: starting
>   hundreds of small kernels, and leaving most of the GPU's processors idle.

Before you optimize anything, decide what limits it. The roofline model answers
that with one number per operation, and it's usually right.

The model is worth taking seriously because of what it rules out. If an
operation is waiting on memory by a factor of ten, ==no amount of better
arithmetic helps==: not a faster algorithm, not tensor cores, not a
lower-precision matmul. The only moves that work are moving fewer bytes, or
moving them from a closer place. Knowing which case you're in saves weeks.

This chapter starts with the picture, then derives where each operation in the
model sits on it, and ends with two costs the picture leaves out.

## Before you start

- **GPU vocabulary from
  [chapter 0a](/c/00a-notation-and-prerequisites).** *HBM* is the GPU's main memory,
  where the weights and the KV cache live. An *SM*, or streaming multiprocessor,
  is one of the GPU's 108 independent processors. The *L2 cache* is a 40 MB
  buffer between the SMs and HBM. A *warp* is 32 threads that execute together,
  and *coalescing* is what happens when a warp's 32 reads fall on neighboring
  addresses and merge into a few wide transfers.
- **FLOP counting for a matmul.** A *FLOP* is one floating-point operation, an
  add or a multiply, and *FLOP/s* counts them per second. An $(m, k)$ by
  $(k, n)$ product computes $mn$ outputs, each a sum of $k$ products. That's
  $mkn$ multiplies and about $mkn$ adds, so $2mkn$ FLOPs. That factor of 2 is
  the only convention here, and every number in this chapter uses it.
- **The geometry and memory arithmetic from
  [chapter 2](/c/02-memory-arithmetic).**
- **The idea of a lower bound.** Nothing here predicts a runtime. Everything here
  produces a floor that no implementation can go below.

## Picture the roofline

This section gives you the picture that the rest of the chapter puts numbers on.
Once you can sketch it, you can see why decode and prefill behave so differently.

Think of a kitchen with fast cooks and one narrow delivery door. Every dish
needs ingredients carried through the door, the bytes, and chopping, the FLOPs:

- **A crate of ingredients for a little chopping.** The cooks stand at the door
  waiting. Faster cooks change nothing; only a wider door, or fewer
  ingredients, helps.
- **Hours of chopping for a handful of ingredients.** The door sits idle. Now
  faster cooks help, and a wider door doesn't.

What separates the two is one ratio: work per ingredient. On a GPU, that ratio
is FLOPs per byte moved, and it's called *arithmetic intensity*.

### The plot

Put arithmetic intensity on the horizontal axis and the throughput an operation
can reach, in FLOP/s, on the vertical axis. Both axes are logarithmic. The
following sketch isn't to scale, but the shape is exact:

```text
 attainable FLOP/s (log scale)

                          ridge point: 161 FLOPs per byte
                          |
312 TFLOP/s - - - - - - - o==================o====   flat roof: peak arithmetic
                         /                     prefill MLP, 1899
                       /
                     /
                   /   <- slanted roof: bandwidth x intensity
                 /
               o   decode attention, 6
             /
           /
         o   RMSNorm and decode MLP, 1
       /
      +-------------------------------------------->  arithmetic intensity
                                                      (FLOPs per byte, log scale)
```

Read it from left to right:

1. **The slanted part.** The cooks wait at the door. Each byte that arrives
   unlocks a fixed amount of work, so throughput is bandwidth times intensity:
   double the FLOPs per byte and you double the throughput.
2. **The bend.** The bytes arrive exactly as fast as the arithmetic can use
   them. That's the *ridge point*, 161 FLOPs per byte on this course's A100.
3. **The flat part.** The arithmetic units are the limit. More FLOPs per byte
   buys nothing, because throughput is already at the GPU's peak.

Seen from the side, the two lines look like a roof, which is where the name
comes from. An operation left of the ridge is *memory bound*; right of it,
*compute bound*.

The sketch places the model's operations, and the chapter derives each one:

| Operation | FLOPs per byte | Side of the ridge |
|---|---|---|
| Embedding lookup | 0 | Memory |
| RMSNorm | 1 | Memory |
| MLP, one token (decode at batch 1) | 1.0 | Memory |
| Attention during decode | 6 | Memory |
| MLP, 2048 tokens (prefill) | 1899 | Compute |

The embedding, at 0, falls off the left edge of a log axis: it has no arithmetic
at all. Nearly everything else sits on the slanted part. Next, you turn the sketch into two
formulas.

## The two rates and the ridge point

This section derives the ridge point, the intensity where an operation stops
waiting on memory. Two rates set it: arithmetic, $\hla{C}$ FLOPs per second, and
memory bandwidth, $\hlb{\beta}$ bytes per second.

The A100 80GB PCIe card has the following figures. *Tensor cores* are the
units inside each SM that do matrix multiplies, and they're where the peak
arithmetic rate comes from:

| Quantity | Value | Source |
|---|---|---|
| bfloat16 tensor core throughput | 312 TFLOP/s | Rated |
| HBM2e bandwidth | 1935 GB/s | Rated |
| HBM2e bandwidth | 1275 GB/s | Measured by a plain copy |
| Ridge point | 161 FLOPs/byte | Derived in this section |
| Streaming multiprocessors | 108 | Hardware |
| L2 cache | 40 MB | Hardware |

> [!WARNING] Check which card you have
> The 80GB A100 ships in two forms: the SXM4 module, rated at 2039 GB/s, and the
> PCIe card, rated at 1935. The GPU provider hands you whichever is free, so two
> runs of the same lab can land on different silicon. Be suspicious of a 5%
> difference between runs. The lab in chapter 0 prints the card's name.

### Two floors, one ceiling

Start with a tiny case. Suppose an operation reads 1 GB and does 1 GFLOP of
arithmetic:

- **Memory time.** $10^{9}$ bytes at $1.935 \times 10^{12}$ bytes per second is
  0.517 ms.
- **Compute time.** $10^{9}$ FLOPs at $312 \times 10^{12}$ FLOP/s is 3.2 µs.

The operation can't finish before its bytes arrive, so it takes at least
0.517 ms, with the arithmetic idle over 99% of that time. Balancing the two
would take 161 times as many FLOPs for the same bytes.

Now the general case. An operation does $F$ FLOPs and moves $B$ bytes, and each
rate sets its own floor:

$$
t_{\text{compute}} = \frac{F}{\hla{C}}, \qquad t_{\text{memory}} = \frac{B}{\hlb{\beta}}
$$

The hardware can overlap the two but can't beat either, so the runtime floor is
the larger one:

$$
t \;\ge\; \max\!\left(\frac{F}{\hla{C}}, \frac{B}{\hlb{\beta}}\right)
$$

Define the *arithmetic intensity* $\hlc{I} = F / B$, in FLOPs per byte. Divide
$F$ by the floor to get the throughput an operation achieves at best:

$$
P = \frac{F}{t} = \min\!\left(\hla{C},\; \hlb{\beta}\,\hlc{I}\right)
$$

That's the plot as a formula: a flat ceiling at $\hla{C}$, and a slanted
ceiling $\hlb{\beta}\hlc{I}$ that rises with intensity. The ridge point is
where they meet, $\hlb{\beta}\hlc{I} = \hla{C}$:

$$
\boxed{I^{\ast} = \frac{\hla{C}}{\hlb{\beta}} = \frac{312 \times 10^{12}}{1.935 \times 10^{12}} = 161.2\ \text{FLOPs/byte}}
$$

> [!INTUITION]
> Below the ridge point, the operation finishes its arithmetic before the next
> bytes arrive, so adding compute does nothing. Above it, the tensor cores can in
> principle stay saturated.

### The ridge point moves with the rates

The ridge point belongs to the pair of rates, not to the hardware alone.
Change either rate and it moves:

| Compute | Bandwidth | Ridge point |
|---|---|---|
| 312 TFLOP/s | 1935 GB/s (PCIe, rated) | 161.2 |
| 312 TFLOP/s | 2039 GB/s (SXM4, rated) | 153.0 |
| 312 TFLOP/s | 1275 GB/s (measured copy) | 244.7 |

The measured row is the honest one for judging a real kernel, and it makes the
problem worse. Against achievable bandwidth, ==an operation needs 245 FLOPs per
byte== before compute is the limit. So even more of the model sits on the
memory side than the headline 161 suggests.

The following function computes the floor in milliseconds:

```python
def roofline_ms(flops, bytes_moved, spec):
    compute_ms = flops / (spec["bf16_tflops"] * 1e12) * 1e3
    memory_ms  = bytes_moved / (spec["hbm_bandwidth_gbs"] * 1e9) * 1e3
    return max(compute_ms, memory_ms)
```

Real kernels reach 70 to 90% of the floor at best. The number's value isn't the
prediction; it's the direction it points and the moment it tells you to stop.

## Why measured bandwidth is 66% of rated

Which bandwidth you divide by decides whether a kernel looks finished or
broken, so this section explains the gap. Even a perfect copy reaches only two thirds of the rating. A device-to-device
copy is the simplest bandwidth-bound kernel there is: read a word, write it, no
arithmetic, perfectly coalesced, no reuse. On this card it measures 1275 GB/s
against a rating of 1935, which is 66%. The first reason is pure accounting.

> [!WARNING] Count both the reads and the writes
> Copying $N$ bytes moves $2N$ bytes across the memory bus: $N$ read from the
> source and $N$ written to the destination. The 1275 GB/s figure is $2N / t$. If
> you report $N / t$ instead ("I copied 4 GB in 6.3 ms, so 635 GB/s"), you halve
> your own result and conclude the GPU is broken.

This rule applies to every roofline in this chapter, and getting it wrong is the
most common way to misjudge a kernel. The repository's hand-written CUDA vector
add measures 1304 GB/s on the same card, which looks identical to the copy.

It's only identical because both numbers count total traffic. The vector add
reads two arrays and writes one, so its byte count is $3N$, not $N$. Under the
same accounting the two kernels agree, which is the correct conclusion: a vector
add has nothing for a hand-written kernel to beat.

The second reason is that **the rating is a pin rate**: memory clock times bus
width, assuming the bus never idles. A real access stream pays for several
things the rating ignores:

- DRAM refresh.
- Row activation and precharge, whenever an access misses the open row.
- Turning the bus around between reads and writes.

A copy alternates reads and writes continuously, close to the worst case for
turnaround, with no arithmetic to hide any of it. So 1275 GB/s is the ceiling a
bandwidth-bound kernel can reach on this card, and it's the number to measure
against:

```python
achieved_gbs = bytes_moved / (elapsed_s * 1e9)
efficiency = achieved_gbs / 1275   # against the measured copy, not the rating
```

> [!KEY] Efficiency against the copy ceiling tells you when to stop
> A memory-bound kernel at 85% of the copy ceiling is finished. Rewriting it is
> wasted effort; the only remaining move is to touch less data, usually by fusing
> it with a neighbor. A memory-bound kernel at 30% has a real problem:
> uncoalesced access, too few blocks to fill the SMs, or launch overhead
> dominating a tiny kernel.

For reference, the fused RMSNorm in [chapter 13](/c/13-fusion-in-triton) reaches
945 GB/s on 4096 rows of 5120 columns. That's 74% of the copy ceiling and 49% of
the rating. Quoting the second number makes a good kernel look broken.

## The MLP

This section shows why the MLP's intensity tracks the number of tokens in the
pass, which is the single fact that explains batching.

The MLP here is a *SwiGLU* block: two matmuls side by side, *gate* and *up*,
combined elementwise, then a third, *down*. Take one block with the following
sizes:

- Hidden size $h = 5120$, the width of each token's vector.
- Intermediate size $i = 17408$, the width inside the block.
- $T$ tokens processed together in one forward pass.
- $e = 2$ bytes per element, for bfloat16, the 16-bit floating-point format
  the weights are stored in.

In *prefill*, the engine processes a whole prompt in one pass, so $T$ is the
prompt length. In *decode*, it generates one token per sequence per pass, so at
batch 1, $T = 1$.

### FLOPs and bytes

The block runs three matmuls. Gate and up are each $(T, h) \times (h, i)$, and
down is $(T, i) \times (i, h)$. Add their FLOPs:

$$
F = \underbrace{2Thi}_{\text{gate}} + \underbrace{2Thi}_{\text{up}} + \underbrace{2Tih}_{\text{down}} = 6Thi
$$

The SiLU and the elementwise product add $O(Ti)$ FLOPs, three orders of
magnitude below $6Thi$ at $h = 5120$. Drop them.

The block moves two kinds of bytes:

- **Weights.** The three weight matrices are read ==once per forward pass, no
  matter how many tokens are in it==. That property is what makes batching
  work: $3hi\,e = 3 \cdot 5120 \cdot 17408 \cdot 2 = 534.8$ MB.
- **Activations.** The block reads its input and writes its output, both
  $(T, h)$, for $2The$ bytes.

The intermediate $(T, i)$ tensors are transient. Ignore them for now; the end of
this section says when you can't. The total and the intensity are as follows:

$$
B = 3hi\,e + 2The, \qquad \hlc{I}(T) = \frac{6Thi}{3hi\,e + 2The}
$$

Divide through by the weight bytes and set $e = 2$:

$$
\boxed{\hlc{I}(T) = \frac{T}{1 + T / \hld{26112}}}
$$

> [!DEEPDIVE] The simplification, step by step
> Divide numerator and denominator by $3hi\,e$:
>
> $$
> I(T) = \frac{6Thi / (3hie)}{1 + 2The/(3hie)} = \frac{2T/e}{1 + 2T/(3i)}
> $$
>
> With $e = 2$, the numerator is $T$:
>
> $$
> I(T) = \frac{T}{1 + \dfrac{2T}{3i}} = \frac{T}{1 + \dfrac{T}{26112}}
> $$
>
> since $3i/2 = 3 \cdot 17408 / 2 = 26112$.

> [!INTUITION]
> Each token does 2 FLOPs per weight for each of the three matrices, and every
> token in the pass shares one read of those weights. So intensity is
> approximately $T$, the number of tokens in the pass, until the activations
> grow to rival the weights at $T \approx \hld{26112}$.

The approximation is good exactly when $T \ll \hld{26112}$. Its relative error
is $T / 26112$:

| Tokens $T$ | Relative error |
|---|---|
| 1 | 0.004% |
| 16 | 0.06% |
| 2048 | 7.8% |

### Where the approximation fails

The approximation fails in two places, and both matter.

**The asymptote.** Taking $T \to \infty$ in the exact expression gives the
following limit:

$$
\lim_{T \to \infty} \hlc{I}(T) = \frac{3i}{2} = \hld{26{,}112}
$$

Intensity doesn't rise without limit. Past about $T = 26{,}112$, the input and
output activations move more bytes than the weights do, and adding tokens stops
buying reuse. You never run a forward pass that wide, so in practice the bound is
academic. It's why the correct statement is "intensity approaches $T$ from
below" rather than "intensity is $T$."

**The intermediate tensor.** The derivation assumes the $(T, i)$ gate and up
outputs never reach HBM. At small $T$ that's nearly true: they fit in the 40 MB
L2, and a fused kernel keeps them in registers or shared memory, the on-chip
storage inside each SM.

At $T = 4096$ each one is $4096 \times 17408 \times 2 = 143$ MB, 3.6 times the
L2 on its own, and SwiGLU holds two at once, so they spill. Counting them
honestly adds $4Tie$ bytes, two written and two read:

$$
\hlc{I}_{\text{spilled}}(T) = \frac{6Thi}{3hi\,e + 2The + 4Tie}
$$

At $T = 4096$ that's 1842 instead of 3540, a little over half. The operation is
still compute bound, so the verdict doesn't change, but the *floor* does.
Chapter 13 measures this effect: fusing SwiGLU gives a 1.52x speedup, and the
byte count predicts it only once the intermediate stops fitting in L2.

### The table that explains the engine

The following table uses the exact $\hlc{I}(T)$ and a ridge point of 161.2:

| Tokens in pass | Intensity | Ratio to ridge | Bound by |
|---|---|---|---|
| 1 (decode, batch 1) | 1.0 | 0.006 | Memory, by 161x |
| 16 | 16.0 | 0.099 | Memory, by 10x |
| 161 | 160.0 | 0.993 | At the ridge point |
| 512 | 502 | 3.1 | Compute |
| 2048 (prefill) | 1899 | 11.8 | Compute |

> [!KEY] Decode at batch 1 wastes 99.4% of the tensor cores
> At one token, the MLP reaches $1/161.2 = 0.62\%$ of the A100's arithmetic peak.
> No kernel fixes that, because the limit is that you read 535 MB of weights to
> do 535 MFLOPs of work.

==Getting 161 tokens into a single forward pass is what makes the GPU work.==
Continuous batching ([chapter 16](/c/16-continuous-batching)) and chunked
prefill exist to do exactly that, and this table is why.

```viz
10-roofline-explorer
```

The MLP gains from batching because the batch shares its weights. Next, you see
an operation where nothing is shared.

## Attention during decode

This section shows that decode attention's intensity is a constant that neither
context length nor batching can move.

Each new token's query attends to the keys and values of every earlier token,
which the engine stores in the *KV cache*. The MLP reads weights that the whole
batch shares; attention reads a cache that is *per sequence*, so batching
doesn't raise its intensity at all.

The model uses *grouped-query attention* (GQA), where several query heads share
one key-value head. Take one decode step, one full-attention layer, and one
sequence at context length $L$:

- Query heads $H = 24$.
- KV heads $H_{kv} = 4$, so each KV head serves a group of $24 / 4 = 6$ query
  heads.
- Head dimension $d = 256$, the width of one head's query, key, or value
  vector.

**FLOPs.** The query is a single token. Per query head, the scores are a
$(1, d)$ by $(d, L)$ matmul, and applying the weights to the values is a
$(1, L)$ by $(L, d)$ matmul of the same size:

$$
F = \underbrace{2HLd}_{\text{scores}} + \underbrace{2HLd}_{\text{values}} = 4HLd
$$

The softmax and the $\sqrt{d}$ scaling are each $O(HL)$, smaller by a factor of
$d = 256$. Drop them.

**Bytes.** The step reads the whole K and V prefix for the $H_{kv}$ heads only.
Any kernel worth using indexes the shared heads rather than duplicating them:

$$
B = 2 \cdot H_{kv} \cdot L \cdot d \cdot e
$$

The query and the output are $Hd\,e$ bytes each. Both are independent of $L$ and
vanish next to the cache read for any $L$ beyond a handful of tokens.

**Intensity.** Divide, and both $L$ and $d$ cancel:

$$
\hlc{I} = \frac{4H \cancel{L} \cancel{d}}{2H_{kv}\cancel{L}\cancel{d}\,e} = \frac{2H}{e\,H_{kv}} \overset{e=2}{=} \frac{H}{H_{kv}} = \frac{24}{4} = \boxed{6}
$$

That's exactly the GQA group size. Three consequences follow from the
cancellation:

- **Context length doesn't matter.** Intensity at 1k context and at 256k context
  is the same 6. Longer context makes attention slower in proportion, never less
  efficient, and it never becomes compute bound at any length.
- **Batch size doesn't matter.** Each sequence brings its own cache, so both $F$
  and $B$ scale with the batch and the ratio doesn't move. This is the sharpest
  contrast with the MLP, whose whole improvement with batch size came from
  sharing one weight read.
- **The dtype, the cache's number format, matters, and it's the only lever in
  the formula.** Quantizing the
  cache to int8 sets $e = 1$ and doubles intensity to 12. Raising $H_{kv}$ to
  24, which is plain multi-head attention, drops it to 1, six times worse. That's
  the whole argument for grouped queries.

At 6 against a ridge point of 161.2, decode attention is memory bound by a
factor of 27. It's ==permanently memory bound==, which is why every fix reads
fewer bytes: grouped queries, paged storage, and quantized caches.

> [!EXAMPLE] One sequence at 32k context
> Take $L = 32768$ across all 16 full-attention layers. The bytes are as
> follows:
>
> $$
> B = 16 \cdot 2 \cdot 4 \cdot 32768 \cdot 256 \cdot 2 = 2{,}147{,}483{,}648\ \text{bytes} = 2\ \text{GiB}
> $$
>
> That's chapter 2's 64 KiB per token times 32,768 tokens, as it must be. The
> FLOPs are as follows:
>
> $$
> F = 16 \cdot 4 \cdot 24 \cdot 32768 \cdot 256 = 1.29 \times 10^{10}\ \text{FLOPs}
> $$
>
> The memory floor is $2.147 \times 10^{9} / 1.935 \times 10^{12} = 1.11$ ms,
> and the compute floor is $1.29 \times 10^{10} / 312 \times 10^{12} = 41.3\ \mu$s.
> Their ratio is 26.9, which is $161.2 / 6$, as the intensity predicts.

## RMSNorm

This section shows that RMSNorm sits at an intensity of 1 at every size, so its
cost is pure memory traffic.

*RMSNorm* rescales each token's vector so that its root-mean-square is 1, then
multiplies by a learned gain $g$. Every layer runs two of these, and the final
norm makes 129 in the model. They look free and aren't:

$$
y = \frac{x}{\sqrt{\frac{1}{h}\sum_{j=1}^{h} x_j^{2} + \epsilon}} \odot g
$$

Count the work and the traffic for one row, one token's $h$ values:

- **FLOPs per row.** The sum of squares is $h$ multiplies and $h$ adds. Then
  one reciprocal square root, $h$ multiplies to scale, and $h$ multiplies by the
  gain $g$. That's $4h + O(1)$ FLOPs per row, so $4Th$ for $T$ rows. Running
  the reduction in float32 while $x$ is bfloat16, which
  [chapter 4](/c/04-rmsnorm-and-residuals) requires for accuracy, changes none of
  this: the same operations happen in a wider accumulator.
- **Bytes per row.** Read $x$, $he$ bytes, and write $y$, $he$ bytes. The gain
  $g$ is $he$ bytes read once for the whole launch, amortized to nothing over $T$
  rows. That's $2The$, or $4Th$ in bfloat16.

Divide the two, and both $T$ and $h$ cancel:

$$
\hlc{I} = \frac{4Th}{2The} = \frac{2}{e} = 1\ \text{FLOP/byte}
$$

RMSNorm sits 161 times below the ridge point at every size, which is as memory
bound as an operation with any arithmetic in it can be.

The following table gives the numbers at $T = 4096$, $h = 5120$, the size the
repository measures:

| Quantity | Value | How |
|---|---|---|
| Bytes moved | 83.9 MB | $2 \cdot 4096 \cdot 5120 \cdot 2$ |
| FLOPs | 83.9 MFLOP | $4 \cdot 4096 \cdot 5120$ |
| Memory floor | 43.4 µs | At 1935 GB/s |
| Memory floor | 65.8 µs | At 1275 GB/s |
| Compute floor | 0.27 µs | At 312 TFLOP/s |
| Measured, Triton | 88.8 µs | At 945 GB/s |

The compute floor is 160 times below the memory floor, so nothing about the
arithmetic is worth touching. The only lever is the 83.9 MB, and the only way to
shrink it is to ==stop writing the output to HBM at all==.

Fusing the norm with the residual add that precedes it, and with the next
matmul, does that.
Chapter 13 does it and measures 1.10x past L2 and 0.92x within it.

## The embedding lookup

This section covers the first operation in the forward pass, a gather: the
degenerate case of an operation with no arithmetic at all.

- **FLOPs: zero.** An embedding lookup is `table[input_ids]`. It computes
  nothing.
- **Bytes.** For $T$ tokens, read $T$ rows of the table ($The$ bytes), write the
  output ($The$ bytes), and read the indices (8 bytes each as int64), for
  $B = 2The + 8T$.
- **Intensity: zero.** $\hlc{I} = 0 / B = 0$, infinitely far below any ridge
  point. The floor is pure bandwidth, with no compute term to compare against.

> [!EXAMPLE] A 2048-token prefill
> $$
> B = 2 \cdot 2048 \cdot 5120 \cdot 2 + 8 \cdot 2048 = 41.96\ \text{MB}
> $$
>
> That's 21.7 µs at the rating and 32.9 µs at measured bandwidth, against a
> prefill that spends hundreds of milliseconds in the layers.

The embedding is noise, but the reasons it's noise tell you when it stops being
noise:

- **The gather coalesces well here.** Each row of the table is
  $5120 \times 2 = 10{,}240$ contiguous bytes, 80 whole 128-byte cache lines. A
  warp reading one row reads sequentially and wastes nothing, however scattered
  the token IDs are: the gather is random at row granularity and sequential
  within a row. At a hidden size of 128, each row would be 256 bytes, and the
  random component would start to dominate.
- **There's no reuse to exploit.** The table is
  $248{,}320 \times 5120 \times 2 = 2.54$ GB, sixty-three times the 40 MB L2, so
  nothing stays resident between forward passes. The one exception is
  repetition within a batch: if a token ID appears twice, the second read hits
  L2. Common tokens make this happen more than you'd guess, which is why measured
  embedding time is often below the floor computed from distinct rows.

## Kernel launch overhead

The roofline assumes the GPU is busy, and at batch 1 it often isn't. This
section and the next cover the two ways it goes idle.

A *kernel* is one function the GPU runs, and *launching* it means the CPU
handing it to the GPU to start. Each launch costs 5 to 10 microseconds end to
end. A forward pass is a single stream of data-dependent kernels, so those costs
serialize: kernel $j+1$ can't start before kernel $j$ finishes and the launch for
$j+1$ is processed.

Put numbers on a decode step at batch 1, with $N = 26.9$ billion parameters and
2 FLOPs each. The two floors are as follows:

$$
\begin{aligned}
t_{\text{compute}} &= \frac{2N}{\hla{C}} = \frac{53.8 \times 10^{9}}{312 \times 10^{12}} = 0.172\ \text{ms} \\[4pt]
t_{\text{memory}} &= \frac{53.8\ \text{GB}}{1935\ \text{GB/s}} = 27.8\ \text{ms}
\end{aligned}
$$

A decode step runs several hundred kernels; take 400. The following table prices
their launches:

| Measure | At 5 µs per launch | At 10 µs per launch |
|---|---|---|
| Total for 400 launches | 2.0 ms | 4.0 ms |
| As a multiple of the arithmetic floor | 12x | 23x |
| As a fraction of the memory floor | 7% | 14% |

> [!KEY] Starting the kernels costs more than their arithmetic
> The launch overhead is ==an order of magnitude larger than all the arithmetic
> the kernels do==. Launch overhead equals the entire arithmetic floor after
> $0.172\ \text{ms} / 5\ \mu\text{s} = 34$ kernels, and a 64-layer model passes 34
> before it finishes its second layer.

The small kernels are where it bites hardest. A per-token RMSNorm at batch 1
moves $2 \cdot 5120 \cdot 2 = 20{,}480$ bytes, which at 1935 GB/s is 10.6
nanoseconds of work. A 5 µs launch is 470 times the work it launches. Across the
model's 129 norms, that's 0.65 ms of launching to do 1.4 µs of normalizing.

> [!TIP] Capture decode in a CUDA graph
> A CUDA graph captures the whole step, every kernel, dependency, and argument,
> and replays it as one launch, which removes almost all of this overhead. Every
> production engine uses graphs for decode. None of them use graphs for prefill,
> where shapes change every step and the arithmetic is large enough not to care.

## Idle SMs and occupancy

This section asks whether a kernel has enough parallel work to fill 108 SMs,
because a kernel that leaves most of them idle runs far below its roofline.

A kernel's threads come in *blocks*, and the GPU assigns each block to one SM.
*Occupancy*, strictly, is how much of each SM's room for resident threads a
kernel fills. The coarsest form of the question comes first: does every SM get
a block at all?

Picture 108 checkout lanes. A kernel that launches $b$ blocks runs them in
waves of at most 108, and while a partly full last wave finishes, the other
lanes stand empty. The fraction of the machine kept busy is the following:

$$
u(b) = \frac{b}{108 \cdot \lceil b / 108 \rceil}
$$

| Blocks | Waves | Utilization |
|---|---|---|
| 8 | 1 | 7.4% |
| 108 | 1 | 100% |
| 109 | 2 | 50.5% |
| 216 | 2 | 100% |
| 4096 | 38 | 99.8% |

The table carries two lessons:

- **Enough blocks means at least 108**, so that every SM has something to do.
- **Enough waves means the partial last one doesn't matter.** One extra block
  over a full wave halves your utilization, while one extra block over
  thirty-seven full waves costs nothing. This effect is called *wave
  quantization*, and it's why a kernel can get slower when the problem grows by
  one.

```viz
10-wave-quantization
```

Decode at batch 1 loses badly here. Take RMSNorm with one block per row:

- **A 4096-token prefill** launches 4096 blocks: 38 waves at 99.8% utilization.
- **Decode at batch 1** launches one block, $1/108 = 0.93\%$ of the GPU, and the
  other 107 SMs sit idle.

You need batch 108 before every SM has a row, and batch 216 before the waves
come out even. No kernel-level fix exists for that. ==The fix is more work==,
which is what batching is for. This is the second half of the argument that the
MLP table started.

## What goes wrong

- **Counting only reads.** Report $N / t$ for a copy and you halve the measured
  bandwidth. Count every byte read and every byte written.
- **Dropping the factor of 2.** A matmul is $2mkn$ FLOPs, a multiply and an add
  per product. Count $mkn$ and every intensity comes out half size.
- **Charging the weights per token.** The MLP reads its weights once per forward
  pass, whatever $T$ is. Multiply them by $T$ and every pass looks memory bound.
- **Reading all 24 heads of KV.** The cache holds $H_{kv} = 4$ heads. Counting $H$
  instead gives an intensity of 1, not 6.

> [!RECAP]
> - The floor is $\max(F/\hla{C}, B/\hlb{\beta})$, and the ridge point
>   $\hla{C}/\hlb{\beta}$ is 161.2 FLOPs per byte at the rated bandwidth and
>   244.7 at the measured 1275 GB/s.
> - MLP intensity is $T / (1 + T/26112)$, roughly the tokens in the pass: memory
>   bound by 161x at one token, compute bound at 2048.
> - Decode attention's intensity is $2H/(e\,H_{kv}) = 6$, independent of context
>   length and batch size. Only the dtype and the KV head count move it.
> - RMSNorm's intensity is 1 and the embedding's is 0; the only lever for either
>   is fewer bytes.
> - At batch 1, 400 launches cost 12x to 23x the arithmetic floor, and one block
>   per row fills 1 of 108 SMs. Batching and CUDA graphs are the fixes.

## Check your understanding

> [!QUESTION] An operation has intensity 300 on the A100. Is it compute bound?
> Against the rated ridge point of 161, yes. Against the ridge point computed
> from achievable bandwidth, 245, it's still compute bound but only by 22%, so a
> real kernel could land on either side. Intensities within a factor of two of
> the ridge point deserve a measurement rather than a verdict.

> [!QUESTION] You fuse RMSNorm with the residual add, and the byte count says you save 20%. You measure 0.92x, a slowdown. What happened?
> The traffic you eliminated was never going to HBM. At that size the
> intermediate fit in the 40 MB L2, so the unfused version already read it from
> cache at several times HBM bandwidth, and the fused version paid extra register
> pressure for nothing. The roofline counts HBM bytes; when the working set fits
> in cache, its byte count is the wrong model. The repository measures exactly
> this: 1.10x past L2 and 0.92x within it.

> [!QUESTION] Decode attention has intensity 6, and you serve batch 64. What's the intensity now?
> Still 6. Both the FLOPs and the bytes scale with the batch, so the ratio
> doesn't move. Batching raises intensity only for operations that share
> something across sequences, and the KV cache is shared with nothing.

> [!QUESTION] Your matmul kernel moves 2 GB and runs in 2.1 ms. Is it good?
> $2 \times 10^{9} / (2.1 \times 10^{-3}) = 952$ GB/s, which is 75% of the 1275
> GB/s copy ceiling and 49% of the rating. Quote the first: 75% of achievable is
> a finished kernel, and the remaining work is to move less data, not to move it
> faster.

## Lab

> [!TRY]
> Write the roofline calculator that the rest of the course uses: FLOP and byte
> counts for a matmul, an MLP block, and decode attention, plus the ridge point
> and a floor that says which resource binds. You pass when the harness confirms
> the counts, the GQA intensity of 6, and the memory-bound and compute-bound
> verdicts at 1 and 2048 tokens.

Implement the following functions:

- `matmul_flops(m, k, n)` returns $2mkn$.
- `mlp_cost(tokens, hidden, intermediate)` returns `(flops, bytes)` for one
  SwiGLU block: three matmuls' worth of FLOPs, and weights read once plus the
  input read and output written. The harness checks that one token costs
  $3 \cdot 2 \cdot h \cdot i$ FLOPs and that the weight read is 535 MB, that
  intensity at one token lands between 0.5 and 4, and that intensity at 2048
  tokens exceeds five times the ridge point.
- `decode_attention_cost(context_len, num_heads, num_kv_heads, head_dim)`
  returns `(flops, bytes)` for one decode step's attention in one layer. The
  harness checks that intensity at 1k context equals intensity at 64k, that it
  equals the GQA group size of 6, that the operation is memory bound, and that a
  model with 24 KV heads instead of 4 has exactly one sixth the intensity.
- `arithmetic_intensity(flops, bytes)` returns FLOPs per byte. The harness uses
  it for every intensity check.
- `ridge_point()` must come out near 161.2.
- `roofline(flops, bytes)` must return `compute_ms`, `memory_ms`, `floor_ms`,
  `intensity`, and `bound_by`, with the floor equal to the larger of the two
  times, the MLP at one token memory bound, and the MLP at 2048 tokens compute
  bound.

## Further reading

- [Roofline: an insightful visual performance model for multicore architectures](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102) — the same analysis applied to a serving system end to end.
- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) — where the two peak rates come from.
