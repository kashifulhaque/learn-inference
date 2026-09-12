---
title: The roofline
slug: 10-roofline
part: "Part 3 — Making it fast"
summary: Deciding whether an operation is limited by arithmetic or by memory, and deriving every number that decides it.
minutes: 90
gpu: true
objectives:
  - Derive the ridge point from a GPU's two peak rates.
  - Derive arithmetic intensity for the MLP, attention, RMSNorm, and the embedding.
  - Say where the "intensity is roughly T" approximation holds and where it fails.
  - Predict whether prefill or decode limits a given workload.
  - Use measured bandwidth to tell a slow kernel from a memory-bound one.
  - Quantify what kernel launch overhead and low occupancy cost a decode step.
lab: 10-roofline
---

# The roofline

Before optimizing anything, decide what is limiting it. The roofline model
answers that with one number per operation, and it is usually right.

The model is worth taking seriously because of what it rules out. If an
operation is memory bound by a factor of ten, no amount of better arithmetic
will help: not a faster algorithm, not tensor cores, not a lower-precision
matmul. The only moves that work are moving fewer bytes, or moving them from a
closer place. Knowing which of those you are in saves weeks.

## Before you start

This chapter assumes:

- **Chapter 0a's GPU vocabulary**: SM, warp, HBM, L2, coalescing, occupancy.
- **FLOP counting for a matmul.** An $(m, k)$ by $(k, n)$ product computes
  $mn$ outputs, each a sum of $k$ products, so it does $mk n$ multiplies and
  about $mkn$ adds — $2mkn$ FLOPs. That factor of 2 is the only convention here,
  and every number in this chapter uses it.
- **Chapter 2's geometry and memory arithmetic.**
- **The idea of a lower bound.** Nothing here predicts a runtime. Everything
  here produces a floor that no implementation can go below.

## The model

A GPU has two peak rates: arithmetic and memory bandwidth. Call them $C$ FLOPs
per second and $\beta$ bytes per second. For the A100 80GB PCIe card:

| Quantity | Value |
|---|---|
| bfloat16 tensor core throughput | 312 TFLOP/s |
| HBM2e bandwidth, rated | 1935 GB/s |
| HBM2e bandwidth, measured by a plain copy | 1275 GB/s |
| Ridge point | 161 FLOPs/byte |
| Streaming multiprocessors | 108 |
| L2 cache | 40 MB |

Check which card you have before trusting any of these. The 80GB A100 ships in
two forms — the SXM4 module rated at 2039 GB/s and the PCIe card at 1935 — and
the GPU provider hands you whichever is free. Two runs of the same lab can land
on different silicon, which is one reason to be suspicious of a 5% difference
between them. Chapter 0's lab prints the name.

### Deriving the ridge point

An operation does $F$ FLOPs and moves $B$ bytes. Two independent floors:

$$
t_{\text{compute}} = \frac{F}{C}, \qquad t_{\text{memory}} = \frac{B}{\beta}
$$

The hardware can overlap them but cannot beat either, so the runtime floor is
the larger:

$$
t \;\ge\; \max\!\left(\frac{F}{C}, \frac{B}{\beta}\right)
$$

Define the *arithmetic intensity* $I = F / B$, in FLOPs per byte. Then the
achieved throughput at the floor is

$$
P = \frac{F}{t} = \min\!\left(C,\; \beta I\right) .
$$

That is the roofline: a flat ceiling at $C$, and a slanted ceiling of slope
$\beta$ that the operation climbs as its intensity rises. The two meet where
$\beta I = C$, which gives the ridge point

$$
I^{\ast} = \frac{C}{\beta} = \frac{312 \times 10^{12}}{1.935 \times 10^{12}}
= 161.2\ \text{FLOPs/byte} .
$$

Above $I^{\ast}$ an operation can in principle saturate the tensor cores. Below
it, it cannot — it finishes its arithmetic before the next bytes arrive, and
adding compute does nothing.

The ridge point is a property of the pair of rates, not of the hardware alone.
Change either rate and it moves:

| Rates used | Ridge point |
|---|---|
| 312 TFLOP/s over 1935 GB/s (PCIe, rated) | 161.2 |
| 312 TFLOP/s over 2039 GB/s (SXM4, rated) | 153.0 |
| 312 TFLOP/s over 1275 GB/s (measured copy) | 244.7 |

The last row is the honest one for judging a real kernel, and it makes the
problem worse: against achievable bandwidth, an operation needs 245 FLOPs per
byte before compute is the limit, so even more of the model sits on the memory
side than the headline 161 suggests.

In code:

```python
def roofline_ms(flops, bytes_moved, spec):
    compute_ms = flops / (spec["bf16_tflops"] * 1e12) * 1e3
    memory_ms  = bytes_moved / (spec["hbm_bandwidth_gbs"] * 1e9) * 1e3
    return max(compute_ms, memory_ms)
```

Real kernels reach 70 to 90% of the floor at best. The value of the number is
not the prediction; it is the direction it points and the moment it tells you to
stop.

## Why measured bandwidth is 66% of rated

A device-to-device copy is the simplest bandwidth-bound kernel there is: read a
word, write it, no arithmetic, perfectly coalesced, no reuse. On this card it
measures 1275 GB/s against a rating of 1935 — 66%.

Two things are going on, and the first is pure accounting.

**Count reads and writes.** Copying $N$ bytes moves $2N$ bytes across the memory
bus: $N$ read from the source and $N$ written to the destination. The 1275 GB/s
figure is $2N / t$. If you instead reported $N / t$ — "I copied 4 GB in 6.3 ms,
so 637 GB/s" — you would be halving your own result and concluding the GPU was
broken.

This accounting rule applies to every roofline in this chapter, and getting it
wrong is the most common way to misjudge a kernel. The repository's hand-written
CUDA vector add measures 1304 GB/s on the same card, which looks identical to
the copy. It is only identical because both numbers count total traffic: the
vector add reads two arrays and writes one, so its byte count is $3N$, not $N$.
Under the same accounting the two kernels agree, which is the correct conclusion
— there is nothing in a vector add for a hand-written kernel to beat.

**The rating is a pin-rate.** 1935 GB/s is memory clock times bus width. It
assumes the bus never idles. A real access stream pays for DRAM refresh, for
row activation and precharge whenever an access misses the open row, and for
turning the bus around between reads and writes. A copy alternates reads and
writes continuously, which is close to the worst case for bus turnaround, and it
has zero arithmetic to hide any of it behind.

So 1275 GB/s is the ceiling a bandwidth-bound kernel can actually reach on this
card, and it is the number to measure against:

```python
achieved_gbs = bytes_moved / (elapsed_s * 1e9)
efficiency = achieved_gbs / 1275   # against the measured copy, not the rating
```

A memory-bound kernel at 85% of the copy ceiling is finished. Rewriting it is
wasted effort; the only remaining move is to make it touch less data, usually by
fusing it with a neighbor. A memory-bound kernel at 30% has a real problem —
uncoalesced access, too few blocks to fill the SMs, or launch overhead
dominating a tiny kernel.

For reference, the fused RMSNorm in chapter 13 reaches 945 GB/s on 4096 rows of
5120 columns, which is 74% of the copy ceiling and 49% of the rating. Quoting
the second number would make a good kernel look broken.

## The MLP

Take one SwiGLU block: hidden $h = 5120$, intermediate $i = 17408$, processing
$T$ tokens in one forward pass. Write $e = 2$ for bytes per element in bfloat16.

### FLOPs

Three matmuls. Gate and up are each $(T, h) \times (h, i)$; down is
$(T, i) \times (i, h)$:

$$
F = \underbrace{2Thi}_{\text{gate}} + \underbrace{2Thi}_{\text{up}} + \underbrace{2Tih}_{\text{down}} = 6Thi
$$

The SiLU and the elementwise product add $O(Ti)$ FLOPs, three orders of
magnitude below $6Thi$ at $h = 5120$. Drop them.

### Bytes

The three weight matrices are read once per forward pass, no matter how many
tokens are in it — that is the property that makes batching work:

$$
B_{\text{weights}} = 3hi\,e = 3 \cdot 5120 \cdot 17408 \cdot 2 = 534.8\ \text{MB}
$$

The block reads its input and writes its output, both $(T, h)$:

$$
B_{\text{act}} = 2The
$$

The intermediate $(T, i)$ tensors are transient. Ignore them for now; the last
part of this section says when you cannot.

$$
B = 3hi\,e + 2The
$$

### Intensity

$$
I(T) = \frac{6Thi}{3hi\,e + 2The}
$$

Divide numerator and denominator by $3hi\,e$:

$$
I(T) = \frac{6Thi / (3hie)}{1 + 2The/(3hie)} = \frac{2T/e}{1 + 2T/(3i)}
$$

With $e = 2$ this simplifies to

$$
I(T) = \frac{T}{1 + \dfrac{2T}{3i}} = \frac{T}{1 + \dfrac{T}{26112}}
$$

since $3i/2 = 26112$.

So intensity is **approximately $T$**, the number of tokens in the forward pass,
and the approximation is good exactly when $T \ll 3i/2 = 26112$. The relative
error is $T / 26112$: 0.004% at $T = 1$, 0.6% at $T = 16$, 7.8% at $T = 2048$.

### Where the approximation fails

Two places, and both matter.

**The asymptote.** Taking $T \to \infty$ in the exact expression gives

$$
\lim_{T \to \infty} I(T) = \frac{3i}{2} = 26{,}112 .
$$

Intensity does not rise without limit. Past about $T = 26{,}112$ the input and
output activations move more bytes than the weights do, and adding tokens stops
buying reuse. You will never run a forward pass that wide, so in practice this
bound is academic — but it is why the correct statement is "intensity approaches
$T$ from below" rather than "intensity is $T$".

**The intermediate tensor.** The derivation assumed the $(T, i)$ gate and up
outputs never reach HBM. At small $T$ that is nearly true: they fit in the 40 MB
L2 and a fused kernel keeps them in registers or shared memory. At $T = 4096$
each is $4096 \times 17408 \times 2 = 143$ MB — 3.6 times the L2 on its own — and
SwiGLU holds two at once, so they spill. Counting them honestly adds $4Tie$
bytes, two written and two read, and the intensity becomes

$$
I_{\text{spilled}}(T) = \frac{6Thi}{3hi\,e + 2The + 4Tie}
$$

At $T = 4096$ that is 1842 instead of 3540, just over half. The operation is
still compute bound, so the conclusion does not change, but the *floor* does.
This is exactly the effect chapter 13 measures: fusing SwiGLU gives a 1.52x
speedup, and the byte count only predicts it once the intermediate stops fitting
in L2.

### The table that explains the engine

Using the exact $I(T)$ and a ridge point of 161.2:

| Tokens in pass | Intensity | Ratio to ridge | Bound by |
|---|---|---|---|
| 1 (decode, batch 1) | 1.0 | 0.006 | Memory, by 161x |
| 16 | 16.0 | 0.099 | Memory, by 10x |
| 161 | 160.0 | 0.993 | At the ridge point |
| 512 | 502 | 3.1 | Compute |
| 2048 (prefill) | 1899 | 11.8 | Compute |

Decode at batch 1 reaches $1/161.2 = 0.62\%$ of the A100's arithmetic peak. It
wastes 99.4% of the tensor cores, and there is no kernel that fixes that,
because the limit is that you read 535 MB of weights to do 535 MFLOPs of work.

Getting 161 tokens into a single forward pass is what makes the GPU work.
Continuous batching (chapter 16) and chunked prefill exist to do exactly that,
and this table is why.

## Attention during decode

The MLP reads weights that are shared across the batch. Attention reads a KV
cache that is *per sequence*, so batching does not raise its intensity at all.

Take one decode step, one full-attention layer, one sequence, at context length
$L$. Query heads $H = 24$, KV heads $H_{kv} = 4$, head dimension $d = 256$.

### FLOPs

The query is a single token, so per query head the score computation is a
$(1, d)$ by $(d, L)$ matmul:

$$
F_{\text{scores}} = H \cdot 2 \cdot 1 \cdot d \cdot L = 2HLd
$$

Applying the weights to the values is a $(1, L)$ by $(L, d)$ matmul, the same
size:

$$
F_{\text{values}} = H \cdot 2 \cdot 1 \cdot L \cdot d = 2HLd
$$

The softmax itself is $O(HL)$ and the $\sqrt{d}$ scaling is $O(HL)$; both are
smaller by a factor of $d = 256$. Total:

$$
F = 4HLd
$$

### Bytes

The step reads the whole K and V prefix, for $H_{kv}$ heads only — the shared
heads are indexed, not duplicated, in any kernel worth using:

$$
B = 2 \cdot H_{kv} \cdot L \cdot d \cdot e
$$

The query itself is $Hd\,e$ bytes and the output is $Hd\,e$; both are independent
of $L$ and vanish next to the cache read for any $L$ beyond a handful of tokens.

### Intensity

$$
I = \frac{4HLd}{2H_{kv}Ld\,e} = \frac{4H \cancel{L} \cancel{d}}{2H_{kv}\cancel{L}\cancel{d}\,e} = \frac{2H}{e\,H_{kv}}
$$

Both $L$ and $d$ cancel. With $e = 2$:

$$
I = \frac{H}{H_{kv}} = \frac{24}{4} = 6
$$

Exactly the GQA group size. Three consequences follow directly from the
cancellation.

**Context length does not matter.** Intensity at 1k context and at 256k context
is the same 6. Longer context makes attention slower in proportion, never less
efficient. It also never becomes compute bound, at any length.

**Batch size does not matter.** Each sequence brings its own cache, so both $F$
and $B$ scale with the batch and the ratio is unchanged. This is the sharpest
contrast with the MLP, whose entire improvement with batch size came from
sharing one weight read.

**The dtype matters, and it is the only lever in the formula.** The general
result is $I = 2H/(e H_{kv})$. Quantizing the cache to int8 sets $e = 1$ and
doubles intensity to 12. Raising $H_{kv}$ to 24 — plain multi-head attention —
drops it to 1, six times worse, which is the whole argument for grouped queries.

At 6 against a ridge point of 161.2, decode attention is memory bound by a factor
of 27. It is permanently memory bound, which is why every fix is about reading
fewer bytes: grouped queries, paged storage, quantized caches.

### The numbers at 32k

One sequence at $L = 32768$, across all 16 full-attention layers:

$$
B = 16 \cdot 2 \cdot 4 \cdot 32768 \cdot 256 \cdot 2 = 2{,}147{,}483{,}648\ \text{bytes} = 2\ \text{GiB}
$$

which is chapter 2's 64 KiB per token times 32,768 tokens, as it must be.

$$
F = 16 \cdot 4 \cdot 24 \cdot 32768 \cdot 256 = 1.29 \times 10^{10}\ \text{FLOPs}
$$

Floors: $2.147 \times 10^{9} / 1.935 \times 10^{12} = 1.11$ ms of memory, against
$1.29 \times 10^{10} / 312 \times 10^{12} = 41.3\ \mu$s of compute. A ratio of
26.9, which is $161.2 / 6$ as the intensity predicted.

## RMSNorm

Every layer runs two of these, and the final norm makes 129 in the model. They
look free and are not.

$$
y = \frac{x}{\sqrt{\frac{1}{h}\sum_{j=1}^{h} x_j^{2} + \epsilon}} \odot g
$$

**FLOPs per row.** The sum of squares is $h$ multiplies and $h$ adds. One
reciprocal square root. Then $h$ multiplies to scale and $h$ multiplies by the
weight $g$. That is $4h + O(1)$ FLOPs per row, so $4Th$ for $T$ rows.

Running the reduction in float32 while $x$ is bfloat16, which chapter 4 requires
for accuracy, changes none of this: the same operations happen in a wider
accumulator.

**Bytes per row.** Read $x$: $he$ bytes. Write $y$: $he$ bytes. The weight $g$ is
$he$ bytes read once for the whole launch, amortized to nothing over $T$ rows.
So $2The$, or $4Th$ in bfloat16.

**Intensity.**

$$
I = \frac{4Th}{2The} = \frac{2}{e} = 1\ \text{FLOP/byte}
$$

$T$ and $h$ both cancel. RMSNorm sits 161 times below the ridge point at every
size, which is as memory bound as an operation with any arithmetic in it can be.

At $T = 4096$, $h = 5120$ — the size the repository measures:

| | Value |
|---|---|
| Bytes moved | $2 \cdot 4096 \cdot 5120 \cdot 2 = 83.9$ MB |
| FLOPs | $4 \cdot 4096 \cdot 5120 = 83.9$ MFLOP |
| Memory floor at 1935 GB/s | 43.4 µs |
| Memory floor at 1275 GB/s | 65.8 µs |
| Compute floor at 312 TFLOP/s | 0.27 µs |
| Measured, Triton, at 945 GB/s | 88.8 µs |

The compute floor is 160 times below the memory floor. Nothing about the
arithmetic is worth touching. The only lever is the 83.9 MB, and the only way to
shrink it is to stop writing $y$ to HBM at all — which is what fusing the norm
with the residual add and with the next matmul achieves. Chapter 13 does that
and measures 1.10x past L2 and 0.92x within it.

## The embedding lookup

The first operation in the forward pass is a gather, and its roofline is the
degenerate case.

**FLOPs: zero.** An embedding lookup is `table[input_ids]`. It computes nothing.

**Bytes.** For $T$ tokens: read $T$ rows of the table, $The$ bytes; write the
output, $The$ bytes; read the indices, 8 bytes each as int64.

$$
B = 2The + 8T
$$

**Intensity: zero.** $I = 0 / B = 0$, infinitely far below any ridge point. The
floor is pure bandwidth and there is no compute term to compare against.

At $T = 2048$:

$$
B = 2 \cdot 2048 \cdot 5120 \cdot 2 + 8 \cdot 2048 = 41.96\ \text{MB}
$$

which is 21.7 µs at the rating and 32.9 µs at measured bandwidth. Against a
prefill of 2048 tokens that spends hundreds of milliseconds in the layers, the
embedding is noise — but the reason it is noise is worth knowing, because the
same analysis explains when it stops being noise.

**Why the gather coalesces well here.** Each row of the table is
$5120 \times 2 = 10{,}240$ contiguous bytes, 80 whole cache lines. A warp reading
one row reads sequentially and wastes nothing, regardless of how scattered the
token ids are. The gather is random at row granularity and perfectly sequential
within a row. Shrink the hidden size to 128 and each row would be 256 bytes, at
which point the random component starts to dominate.

**Why there is no reuse to exploit.** The table is
$248{,}320 \times 5120 \times 2 = 2.54$ GB, sixty-three times the 40 MB L2.
Nothing stays resident between forward passes. The one exception is repetition
within a batch: if the same token id appears twice, the second read hits L2.
Common tokens make this happen more than you would guess, which is why measured
embedding time is often below the floor computed from distinct rows.

## Two things the roofline does not cover

The roofline assumes the GPU is busy. Two ways it is not.

### Kernel launch overhead

Launching a kernel costs 5 to 10 microseconds end to end. In a single stream of
data-dependent kernels — which is what a forward pass is — those costs
serialize, because kernel $j+1$ cannot start before kernel $j$ finishes and the
launch for $j+1$ has been processed.

Put numbers on a decode step at batch 1. The floors:

$$
t_{\text{compute}} = \frac{2N}{C} = \frac{53.8 \times 10^{9}}{312 \times 10^{12}} = 0.172\ \text{ms}
$$

$$
t_{\text{memory}} = \frac{53.8\ \text{GB}}{1935\ \text{GB/s}} = 27.8\ \text{ms}
$$

using $N = 26.9$ billion parameters and 2 FLOPs each. A decode step runs several
hundred kernels; take 400. Then:

| | Cost |
|---|---|
| 400 launches at 5 µs | 2.0 ms |
| 400 launches at 10 µs | 4.0 ms |
| As a multiple of the arithmetic floor | 12x to 23x |
| As a fraction of the memory floor | 7% to 14% |

The second row is the one that should be alarming. The overhead of *starting*
the kernels is an order of magnitude larger than all the arithmetic they do. Put
differently, the number of kernels at which launch overhead equals the entire
arithmetic floor is

$$
\frac{0.172\ \text{ms}}{5\ \mu\text{s}} = 34\ \text{kernels} ,
$$

and a 64-layer model passes 34 before it finishes its second layer.

The small kernels are where it bites hardest. A per-token RMSNorm at batch 1
moves $2 \cdot 5120 \cdot 2 = 20{,}480$ bytes, which at 1935 GB/s is 10.6
nanoseconds of work. A 5 µs launch is 470 times the work it launches. Across
the model's 129 norms that is 0.65 ms of launch to do 1.4 µs of normalizing.

CUDA graphs capture the whole step — every kernel, every dependency, every
argument — and replay it as one launch, which removes almost all of this. Every
production engine uses them for decode, and none of them use them for prefill,
where shapes change every step and the arithmetic is large enough not to care.

### Occupancy

Occupancy is whether you have enough parallel work to fill 108 SMs. A kernel
launching $b$ blocks runs them in waves of at most 108, so the fraction of the
machine it keeps busy is

$$
u(b) = \frac{b}{108 \cdot \lceil b / 108 \rceil} .
$$

| Blocks | Waves | Utilization |
|---|---|---|
| 8 | 1 | 7.4% |
| 108 | 1 | 100% |
| 109 | 2 | 50.5% |
| 216 | 2 | 100% |
| 4096 | 38 | 99.8% |

Two lessons. "Enough blocks" means at least 108, so that every SM has something
to do. And it means enough *waves* that the partial last one does not matter:
one extra block over a full wave halves your utilization, while one extra block
over thirty-seven full waves costs nothing. That effect has a name, *wave
quantization*, and it is why a kernel can get slower when you increase the
problem size by one.

Decode at batch 1 loses here badly. Take RMSNorm with one block per row. Prefill
of 4096 tokens launches 4096 blocks: 38 waves, 99.8% utilization, the bottom row
of the table. Decode at batch 1 launches one block: $1/108 = 0.93\%$ of the GPU,
and the other 107 SMs sit idle for the duration. You need batch 108 before every
SM has a row, and batch 216 before the waves come out even.

There is no kernel-level fix for that. The fix is to have more work, which is
what batching is for, and this is the second half of the argument the MLP table
started.

## Check your understanding

**An operation has intensity 300 on the A100. Is it compute bound?**

Against the rated ridge point of 161, yes. Against the ridge point computed from
achievable bandwidth, 245, it is still compute bound but only by 22%, so a real
kernel could land on either side. Intensities within a factor of two of the ridge
point deserve a measurement rather than a verdict.

**You fuse RMSNorm with the residual add and the byte count says you should save
20%. You measure 0.92x — a slowdown. What happened?**

The traffic you eliminated was never going to HBM. At that size the intermediate
fit in the 40 MB L2, so the unfused version was already reading it from cache at
several times HBM bandwidth, and the fused version paid extra register pressure
for nothing. The roofline counts HBM bytes; when the working set fits in cache,
its byte count is the wrong model. The repository measures exactly this: 1.10x
past L2 and 0.92x within it.

**Decode attention has intensity 6, and you serve batch 64. What is the
intensity now?**

Still 6. Both the FLOPs and the bytes scale with the batch, so the ratio does
not move. Batching raises intensity only for operations that share something
across sequences, and the KV cache is shared with nothing.

**Your matmul kernel moves 2 GB and runs in 2.1 ms. Is it good?**

$2 \times 10^{9} / (2.1 \times 10^{-3}) = 952$ GB/s, which is 75% of the 1275
GB/s copy ceiling and 49% of the rating. Quote the first: 75% of achievable is
a finished kernel, and the remaining work is to move less data, not to move it
faster.

## Lab

Write the roofline calculator the rest of the course uses.

`matmul_flops(m, k, n)` returns $2mkn$.

`mlp_cost(tokens, hidden, intermediate)` returns `(flops, bytes)` for one SwiGLU
block: three matmuls' worth of FLOPs, and weights read once plus the input read
and output written. The harness checks that one token costs
$3 \cdot 2 \cdot h \cdot i$ FLOPs and that the weight read is 535 MB, that
intensity at one token lands between 0.5 and 4, and that intensity at 2048
tokens exceeds five times the ridge point.

`decode_attention_cost(context_len, num_heads, num_kv_heads, head_dim)` returns
`(flops, bytes)` for one decode step's attention in one layer. The harness
checks that intensity at 1k context equals intensity at 64k, that it equals the
GQA group size of 6, that the operation is memory bound, and that a model with
24 KV heads instead of 4 has exactly one sixth the intensity.

`ridge_point()` must come out near 161.2, and `roofline(flops, bytes)` must
return `compute_ms`, `memory_ms`, `floor_ms`, `intensity`, and `bound_by`, with
the floor equal to the larger of the two times, the MLP at one token memory
bound, and the MLP at 2048 tokens compute bound.

## Further reading

- [Roofline: an insightful visual performance model for multicore architectures](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [Making deep learning go brrrr from first principles](https://horace.io/brrr_intro.html)
- [Efficiently scaling transformer inference](https://arxiv.org/abs/2211.05102) — the same analysis applied to a serving system end to end.
- [NVIDIA A100 tensor core GPU architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) — where the two peak rates come from.
