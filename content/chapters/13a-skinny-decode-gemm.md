---
title: Skinny GEMMs for decode
slug: 13a-skinny-decode-gemm
part: "Part 4 — Kernels"
summary: Why a decode GEMM with a short batch is memory bound, and what a W4A16 weight changes about those bytes.
minutes: 90
gpu: true
objectives:
  - "Compute the FLOPs and the bytes of one GEMM, including the output matrix."
  - "Show that a bfloat16 product on this model's up projection is memory bound at M = 1, 8, and 64, and compute bound at M = 4096."
  - "Count a W4A16 weight as a 4-bit payload plus one bfloat16 scale per group of 128, and place that product on the same ridge."
  - "Explain why a wide output tile does not raise the intensity of a one-row product."
  - "Unpack a signed int4 matrix-vector product, and run a masked row GEMM when CUDA is available."
lab: 13a-skinny-gemm
---

# Skinny GEMMs for decode

> [!TLDR]
> - A linear layer computes $C = AB$ with $A$ shaped $(\hla{M}, K)$. During decode, $\hla{M}$ is the batch. On the up projection, $K = 5120$ and $N = 17408$.
> - Bfloat16 intensity is $2\hla{M}KN$ over the bytes of $A$, $B$, and $C$. At $\hla{M} = 1$ that ratio is $0.9997$ FLOPs per byte. The rated ridge is $161.2$, so the product is memory bound.
> - $\hla{M} = 8$ and $\hla{M} = 64$ stay memory bound in bfloat16. $\hla{M} = 4096$ is compute bound. The bound flips between $168$ and $169$ rows.
> - A W4A16 weight is $33/64$ bytes per value when the group is 128. At $\hla{M} = 1$ the memory floor drops from $92.15$ µs to $23.77$ µs. The int4 product crosses the ridge between $43$ and $44$ rows.
> - Tiling and tensor cores are the [GPU course's GEMM chapter](https://gpu.ifkash.dev/c/10-gemm-from-scratch). This chapter counts the decode shapes and writes a row kernel.

Decode spends its time reading weights. A linear layer is a GEMM, $C = AB$, and the weight is the wide factor. The activation $A$ has $\hla{M}$ rows. $\hla{M}$ is the number of tokens in the step: one sequence generating alone, or a handful of sequences sharing the weights. $K$ and $N$ are thousands.

[Chapter 10](/c/10-roofline) already places the whole MLP near one FLOP per byte at batch 1. This chapter takes one of those products apart, keeps the output bytes in the count, and applies the same ridge to the int4 layout from [chapter 18](/c/18-quantization). A large tile, a tensor-core fragment, and wave quantization are how you make a big GEMM fast. They are the subject of the [GPU course's GEMM chapter](https://gpu.ifkash.dev/c/10-gemm-from-scratch) and its [tensor-core chapter](https://gpu.ifkash.dev/c/11-tensor-cores). They do not change the byte count of a product that has one row.

## Before you start

**The roofline, from [chapter 10](/c/10-roofline).** An operation takes at least the longer of its FLOPs divided by the arithmetic peak and its bytes divided by bandwidth. On the A100 80GB PCIe card the bfloat16 tensor-core peak is $312$ TFLOP/s and the rated bandwidth is $1935$ GB/s. The ridge, where those two floors meet, is $161.2$ FLOPs per byte. A plain copy on the card this course measured moves $1275$ GB/s, and the ridge against that copy is $244.7$. This chapter uses $161.2$ for the bound, the same ridge as chapter 10's MLP table. It does not time a kernel.

**One linear layer.** $A$ has shape $(\hla{M}, K)$ and $B$ has shape $(K, N)$, so $C$ has shape $(\hla{M}, N)$. The up projection uses $K = 5120$ and $N = 17408$, the same sizes as the MLP in chapter 10. The MLP is three products. The numbers in this chapter are one of them.

**Triton, from [chapter 13](/c/13-fusion-in-triton).** You write what one program does on a tile, and you mask the tail. Accumulation stays in float32.

**Int4 scales, from [chapter 18](/c/18-quantization).** Symmetric int4 stores an integer in $-8$ to $7$. Weights drop the zero point. A group of $128$ shares one bfloat16 scale, which is $2/128$ bytes of scale on top of the $1/2$ byte of payload.

## Count every matrix the product touches

This section writes the intensity of one GEMM, with the output included.

A product of $(\hla{M}, K)$ by $(K, N)$ performs one multiply and one add at each of the $\hla{M}KN$ entries of the sum:

$$
F = 2\hla{M}KN.
$$

Bfloat16 stores two bytes per value. The read of $A$, the read of $B$, and the write of $C$ are three terms:

$$
b = 2\hla{M}K + 2KN + \hld{2MN}.
$$

$\hld{2MN}$ is the output, and $b$ is the byte count. The matrix $B$ is the weight, so the byte count takes a different letter. Divide $F$ by $b$. The factor of two cancels, which is why a bfloat16 GEMM has a simple intensity:

$$
\boxed{\hlc{I} = \frac{\hla{M}KN}{\hla{M}K + KN + \hla{M}N}}
$$

> [!INTUITION]
> Each weight is used once per row of $A$. $\hla{M}$ rows reuse the weight $\hla{M}$ times, so $\hlc{I}$ grows with $\hla{M}$. The activation and the output add bytes that do not produce more FLOPs, so $\hlc{I}$ stays strictly smaller than $\hla{M}$.

Plug in the up projection at $\hla{M} = 1$. The weight holds $5120 \times 17408 = 89{,}128{,}960$ values. FLOPs are twice that, $178{,}257{,}920$. Bytes are twice the weight, plus $2 \times 5120$ for $A$, plus $\hld{2 \times 17408}$ for $C$:

$$
b = 178{,}257{,}920 + 10{,}240 + 34{,}816 = 178{,}302{,}976.
$$

$$
\hlc{I}(1) = \frac{178{,}257{,}920}{178{,}302{,}976} = 0.9997.
$$

The memory floor at the rated $1935$ GB/s is $178{,}302{,}976$ divided by $1.935 \times 10^{12}$, which is $92.15$ µs. The compute floor at $312$ TFLOP/s is $0.57$ µs. The memory floor is the one that binds.

> [!EXAMPLE] One token through the up projection
> Intensity $0.9997$ against a ridge of $161.2$ is a factor of about $161$ short of the ridge. The output is only $34{,}816$ of the $178{,}302{,}976$ bytes. Dropping it moves the intensity from $0.9997$ to $0.9999$ and leaves the bound as memory. The lab still compares the integers, because a later batch makes those bytes a large share of the total. `scripts/figures/13a-skinny-decode-gemm.py` prints both counts.

==One token does about one FLOP per byte.== That is the same place chapter 10 put the whole MLP. The output term is why the ratio is $0.9997$ rather than $1$.

## Four batch sizes on this layer

This section places $\hla{M} \in \{1, 8, 64, 4096\}$ on the rated ridge.

The formula is the same at every batch. Only $\hla{M}$ changes. The following table is that formula on the up projection. Intensity is rounded to four digits. The script prints the unrounded ratio.

| $\hla{M}$ | Bytes | Intensity | Bound at $161.2$ |
|---|---|---|---|
| 1 | 178,302,976 | 0.9997 | Memory |
| 8 | 178,618,368 | 7.984 | Memory |
| 64 | 181,141,504 | 62.98 | Memory |
| 4096 | 362,807,296 | 2012 | Compute |

$\hla{M} = 64$ is still short of the ridge: $62.98$ against $161.2$. $\hla{M} = 4096$ is past it. The script walks every integer batch. The bound flips between $168$ and $169$: intensity $161.16$ at $168$ rows, and $162.08$ at $169$. A rule that says "compute bound once $\hla{M}$ exceeds $161$" calls $168$ compute bound. The activations and the output keep it memory bound for seven more rows.

At $\hla{M} = 4096$ the output is $142{,}606{,}336$ bytes, $39\%$ of the $362{,}807{,}296$ total. Dropping $\hld{2MN}$ leaves $220{,}200{,}960$ bytes and an intensity of $3316$ instead of $2012$. Both numbers are past the ridge, so the bound does not catch the miss. The byte count does.

```viz
13a-skinny-intensity
```

The figure draws the same ratio, for bfloat16 and for the int4 weight in the next section. The microsecond readout divides the byte count by $1935$ GB/s. It is a floor. It is not a kernel time. Chapter 10's copy reaches $1275$ GB/s, so a real read of these bytes takes $1935/1275$ times as long as that floor.

> [!KEY] Batch 1 is memory bound because the weight is read once
> $\hlc{I}(1) = 0.9997$. The ridge is $161.2$. No schedule changes the fact that you move $178$ MB to do $178$ MFLOP. A faster kernel approaches the memory floor. It does not cross it.

## A tile does not invent rows

This section says what a large output tile can and cannot change about a one-row product.

A tiled GEMM loads a block of $A$ and a block of $B$, reuses both inside the block, and writes a block of $C$. The [Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) is that loop: a program owns a `BLOCK_SIZE_M` by `BLOCK_SIZE_N` output tile and streams $K$ in chunks. The tutorial is written to approach cuBLAS on a large product. The [GPU course](https://gpu.ifkash.dev/c/10-gemm-from-scratch) derives how the tile's intensity grows with the tile edge, and why that edge has to clear the ridge before arithmetic can be the limit.

On a decode step the tile edge along $\hla{M}$ is the batch. If the batch is $1$, the largest honest output tile is $1 \times N_{\text{tile}}$. Each weight element in that tile is used once. The activation vector is reused across the $N_{\text{tile}}$ columns, and the byte formula already counts that reuse: $A$ contributes $2K$ bytes, not $2KN$. Making $N_{\text{tile}}$ larger does not create a second row to reuse the weight against.

==A tile cannot reuse a row the batch does not have.== The intensity cap of a one-row tile is the same cap as $\hlc{I}(1)$, about one FLOP per byte. That is why this chapter writes a row kernel rather than a large-tile GEMM.

The column axis still gives you programs. $N = 17408$ and a tile of $128$ columns is $136$ programs for a single row. The A100 has $108$ SMs, so one row already launches more programs than one wave holds. You split $K$ when $N$ is short and those programs run out. This layer does not.

## Int4 cuts the weight, not the FLOPs

This section counts a W4A16 product. *W4A16* stores the weight in int4 and keeps the activation in bfloat16. The matmul still does $F = 2\hla{M}KN$ FLOPs after the integers are scaled back.

Chapter 18's group of $128$ costs half a byte of payload and $2/128$ bytes of scale:

$$
\frac{1}{2} + \frac{2}{128} = \frac{33}{64} = 0.515625
$$

bytes per weight, when $K$ is divisible by $128$. There is no zero point. On this layer the weight is $89{,}128{,}960 \times 33/64 = 45{,}957{,}120$ bytes, against $178{,}257{,}920$ bytes in bfloat16. The payload ratio is $128/33 = 3.879$. Scales are why it is not $4$.

$A$ and $C$ stay bfloat16. At $\hla{M} = 1$ the total is

$$
b_{\text{w4}} = 10{,}240 + 45{,}957{,}120 + 34{,}816 = 46{,}002{,}176.
$$

Intensity is $178{,}257{,}920 / 46{,}002{,}176 = 3.875$. The memory floor is $23.77$ µs, against $92.15$ µs in bfloat16. The ratio of the two totals is $3.876$. The activation and the output dilute the $3.879$ you would get from the weight alone.

| $\hla{M}$ | W4 bytes | W4 intensity | Bound at $161.2$ |
|---|---|---|---|
| 1 | 46,002,176 | 3.875 | Memory |
| 8 | 46,317,568 | 30.79 | Memory |
| 64 | 48,840,704 | 233.6 | Compute |
| 4096 | 230,506,496 | 3168 | Compute |

The int4 product crosses $161.2$ between $43$ and $44$ rows. Intensity is $160.04$ at $43$ and $163.61$ at $44$. At $\hla{M} = 64$ the bfloat16 product is still memory bound ($62.98$) and the int4 product is compute bound ($233.6$). Past that point the two products share the same compute floor, because they do the same $F$ and the byte cut no longer sets the time. At $\hla{M} = 4096$ both are compute bound.

> [!WARNING] The ridge you compare against moves the M = 64 verdict
> $233.6$ is past $161.2$ and short of $244.7$. Against the rated bandwidth the int4 layer at $\hla{M} = 64$ is compute bound. Against chapter 10's measured copy it is still memory bound. The lab uses $161.2$. A kernel you time on the copy bandwidth can still be waiting on bytes at a batch the rated ridge calls compute bound.

Dequantization adds work that $F = 2\hla{M}KN$ leaves out. Scaling each integer is one extra multiply per weight, $KN$ FLOPs, if you scale the weight once and reuse it across $\hla{M}$. At $\hla{M} = 1$ the FLOP count goes from $2KN$ to $3KN$ and the intensity goes from $3.875$ to $5.812$. That is still a factor of about $28$ short of $161.2$. The extra multiply does not move the bound at batch 1.

MARLIN reports the same shape on their kernels, not on this layer. Frantar and others describe weights loaded compressed and decompressed in registers. They put a typical Ampere FP16 FLOP-per-byte ratio in the range $100$ to $200$. This card's rated ridge, $161.2$, sits in that range. On an A10, their 4-bit kernels stay near a $4\times$ speedup against FP16 for batch sizes up to 16 to 32 on large matrices, and the speedup falls toward $1.5\times$ at batch $128$ as the product becomes compute bound. Their end-to-end figure with vLLM is up to $2.8\times$ at batch $16$. Those are their measurements. The numbers you can recompute here are the byte floors.

> [!EXAMPLE] Four rows, three columns, group size 2
> Stored integers, then the scale for each group of two rows:
>
> $$
> q = \begin{bmatrix} 1 & -2 & 7 \\ 0 & -8 & 3 \\ -1 & 4 & -4 \\ 2 & 5 & -7 \end{bmatrix},
> \quad
> s = \begin{bmatrix} 0.5 & 0.25 & 1 \\ 0.5 & 2 & 0.25 \end{bmatrix}.
> $$
>
> The scaled weight is $0.5, -0.5, 7$ on the first row and $1, 10, -1.75$ on the last. An activation of $(1, 1, 1, 1)$ produces $(1, 15.5, 7.25)$. Pack the integers as `value + 8` in a nibble. The low nibble of byte $j$ is column $2j$, and the high nibble is column $2j+1$. $N = 3$ leaves the last high nibble unused. The lab checks this vector, and a case whose $K$ is not a multiple of the group.

==The byte cut stops setting the time once the product is compute bound.== That is $\hla{M} \ge 44$ for this int4 layer at the rated ridge, and $\hla{M} \ge 169$ in bfloat16.

## Write the row, and unpack in the same loop

This section shows the kernel that matches the byte count: one row of $A$, a tile of columns of $B$, and a mask on the tail.

A program owns one row and `BLOCK_N` columns. It streams $K$ in chunks of `BLOCK_K`, multiplies, and accumulates in float32. Positions past $K$ or $N$ load as zero, so they add nothing:

```python
offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
mask_n = offs_n < N
acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    offs_k = k0 + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    a_vec = tl.load(a_ptr + row * stride_am + offs_k * stride_ak,
                    mask=mask_k, other=0.0).to(tl.float32)
    b_tile = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    acc += tl.sum(a_vec[:, None] * b_tile, axis=0)
tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)
```

The grid is $(\hla{M},\ \lceil N / \texttt{BLOCK\_N} \rceil)$. $\hla{M} = 7$ and a $K$ that is not a multiple of `BLOCK_K` both fall out of the masks. This is not the blocked `tl.dot` kernel in the Triton tutorial. That kernel is the large-tile case. This one is the decode case, where the row is already as short as the math allows.

The int4 product uses the same loop and unpacks the weight in it. Two signed values share a byte. The low nibble is column $2j$:

```python
nibble = packed & 0x0F          # 0..15
value = nibble.to(torch.int16) - 8   # -8..7
```

The high nibble is `packed >> 4`, and it is column $2j+1$. Row $t$ multiplies by `scales[t // group_size]`. Then the dot is the bfloat16 product you already counted. The lab grades that unpack in PyTorch, including a group that does not divide $K$ and an odd $N$. The Triton row kernel is graded only when a CUDA device is visible. This machine has no CUDA device, so the chapter does not report a runtime for it.

## What goes wrong

A few misses produce a bound that looks plausible and a byte count that is not:

- **Dropping the output.** At $\hla{M} = 1$ you omit $34{,}816$ bytes out of $178{,}302{,}976$, and the bound stays memory. At $\hla{M} = 4096$ you omit $142{,}606{,}336$ bytes. The lab compares $362{,}807{,}296$, not a tolerance around the intensity.
- **Calling $\hla{M} = 1$ compute bound.** The intensity is $0.9997$. The compute floor, $0.57$ µs, is the one that does not bind.
- **Using $\hla{M}$ itself as the intensity.** That rule flips at $161$ and calls $168$ compute bound. The counted intensity at $168$ is $161.16$, still memory bound.
- **Forgetting the scale, or rounding the group down.** $K = 100$ is one group of $128$, not zero groups. The scale is $2 \times N$ bytes even though the group is short.
- **Reading the high nibble as column $2j$.** The worked product $(1, 15.5, 7.25)$ becomes a different vector. So does forgetting to subtract $8$.
- **Quoting the floor as a measured GB/s.** $92.15$ µs is $178{,}302{,}976$ bytes divided by $1935$ GB/s. A kernel on this machine was not timed.

> [!RECAP]
> - $\hlc{I} = \hla{M}KN / (\hla{M}K + KN + \hla{M}N)$ for a bfloat16 GEMM. The last term in the denominator is the output.
> - On the up projection, $\hlc{I}(1) = 0.9997$, $\hlc{I}(8) = 7.984$, $\hlc{I}(64) = 62.98$, and $\hlc{I}(4096) = 2012$. The first three are memory bound at the rated ridge of $161.2$. The bound flips between $168$ and $169$.
> - A one-row tile reuses the activation across columns. It does not reuse the weight across rows. Large-tile GEMM and tensor cores are the GPU course.
> - W4A16 stores $33/64$ bytes per weight at group $128$. At $\hla{M} = 1$ the floor falls from $92.15$ µs to $23.77$ µs, and the product is still memory bound. It crosses the ridge between $43$ and $44$.
> - At $\hla{M} = 64$ the int4 intensity is $233.6$: compute bound at $161.2$, and still short of the $244.7$ ridge from the measured copy. The lab uses $161.2$.
> - The row kernel masks the tail and accumulates in float32. The int4 unpack is `nibble - 8`, then the group scale.

## Check your understanding

> [!QUESTION] The up projection at M = 1 does 178,257,920 FLOPs. Why is it memory bound?
> Those FLOPs sit on $178{,}302{,}976$ bytes, which is $0.9997$ FLOPs per byte. The rated ridge is $161.2$. The memory floor is $92.15$ µs and the compute floor is $0.57$ µs, so the bytes bind. The FLOP count is large in absolute terms and tiny next to the bytes.

> [!QUESTION] Dropping the output at M = 1 changes the intensity from 0.9997 to 0.9999. Why does the lab still reject that count?
> The bound does not move, and a tolerance of $0.1\%$ hides a gap of $34{,}816$ bytes. At $\hla{M} = 4096$ the same omission drops $142{,}606{,}336$ bytes and changes the intensity from $2012$ to $3316$. The check compares the integer $362{,}807{,}296$.

> [!QUESTION] At M = 64 the bfloat16 product is memory bound and the int4 product is compute bound. How can both be true?
> Both do the same $2\hla{M}KN$ FLOPs. The int4 weight is about $3.88\times$ smaller, so its intensity is $233.6$ instead of $62.98$. $62.98$ is short of $161.2$ and $233.6$ is past it. Against the $244.7$ copy ridge, $233.6$ would still be memory bound. The lab's ridge is $161.2$.

> [!QUESTION] A program owns 128 columns of a one-row product. Why does that tile not clear the ridge?
> Each weight element in the tile is used by one row, so it contributes one multiply-add and then leaves. Reusing the activation across the $128$ columns is already in the byte count, as $2K$ rather than $2KN$. The intensity stays $\hlc{I}(1) = 0.9997$. Clearing the ridge takes more rows, not a wider tile.

## Lab

> [!TRY]
> Count one GEMM and unpack a W4A16 product. You pass when `gemm_flops` returns $2mkn$, when `gemm_bytes` includes $C$ for bfloat16 and the group scale for W4A16, and when `gemm_roofline` matches the table: memory bound at $\hla{M} = 1, 8, 64$ in bfloat16, compute bound at $\hla{M} = 4096$, and the int4 crossings at $43$/$44$ and $168$/$169$. `w4a16_gemv` matches the worked vector $(1, 15.5, 7.25)$. On CUDA, `skinny_gemm` also matches a matmul. Without a device, that check skips.

The CPU checks run anywhere. Implement the following:

- `gemm_flops(m, k, n)`, equal to $2mkn$. The first check calls only this function.
- `gemm_bytes(m, k, n, storage)`, with `storage` of `"bf16"` or `"w4a16"`. Count $A$, the weight, and $C$. W4A16 uses two signed int4 values per byte and one bfloat16 scale per group of $128$ rows of $K$, rounded up.
- `gemm_roofline(m, k, n, storage)`, a dict with `flops`, `bytes`, `intensity`, `bound`, `memory_s`, and `compute_s`. The bound uses $312$ TFLOP/s and $1935$ GB/s. Intensity under $161.2$ is `"memory"`.
- `w4a16_gemv(x, packed, scales, group_size)`, with the low nibble of byte $j$ equal to column $2j$. Subtract $8$, multiply by the group scale, then the matmul.

On CUDA, also implement the following:

- `skinny_gemm(a, b)`, a masked row kernel. The check uses shapes whose $K$ is not a multiple of the $K$ tile, including $\hla{M} = 1$ and $\hla{M} = 7$. It compares against a matmul. It does not time the kernel.

A byte count that omits $C$ fails `gemm_bytes`. Calling $\hla{M} = 1$ compute bound fails `gemm_roofline`.

## Further reading

- [CS336, lecture 2, resource accounting, and lecture 6, Kernels, Triton](https://cs336.stanford.edu/). Percy Liang. Lecture 2 covers FLOPs, memory, and arithmetic intensity. Lecture 6 is the Triton lecture. The schedule on that page is the source for both titles.
- [CS 229S calendar, 4 October 2023, Hardware Aware Algorithm Design](https://cs229s.stanford.edu/fall2023/calendar/). Presented by Azalia. The calendar lists arithmetic intensity and hardware-aware algorithms for matrix multiplication.
- [Matrix Multiplication, Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). The blocked FP16 kernel, with a program per output tile, that the tutorial compares with cuBLAS. This chapter does not write that tile. It links the tutorial for the large-product case.
- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743). Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. Weights are loaded compressed and decompressed in registers. The Ampere FLOP-per-byte range, the A10 batch curve, and the vLLM speedups cited here are that paper's results.
- [GEMM from scratch](https://gpu.ifkash.dev/c/10-gemm-from-scratch) and [tensor cores](https://gpu.ifkash.dev/c/11-tensor-cores) in the GPU course. Tile intensity, the register-blocked kernel, and the tensor-core instruction. This chapter uses their conclusion and does not rebuild the tile.
