---
title: The forward pass
slug: 08-the-forward-pass
part: "Part 2 — A forward pass"
summary: Assembling 64 layers, deriving the SwiGLU block, and checking your logits against the reference.
minutes: 110
gpu: true
objectives:
  - Assemble a hybrid decoder stack from the layers you have written.
  - Derive the SwiGLU MLP and account for its parameters exactly.
  - Read the layer schedule and say which mixer runs at any layer index.
  - Explain what the output projection produces and what a logit means.
  - Map checkpoint tensor names onto your modules and catch mismatches early.
  - Validate your implementation against Hugging Face transformers.
  - Bisect a numerical mismatch to the first layer that causes it.
lab: 08-forward-pass
---

# The forward pass

You have RMSNorm, RoPE, both token mixers, and now the MLP. This chapter puts
them in order, loads real weights, and proves the result is right.

It is worth being thorough here for a reason that has nothing to do with this
chapter. From chapter 10 onwards, every optimization you write is checked
against the implementation you finish today. A fused kernel is compared with
your unfused one. A paged attention kernel is compared with your contiguous one.
A quantized matmul is compared with your bfloat16 one. All of those tests
compare you against you. If the model you build in this chapter is wrong, every
later test still passes, and nothing in the course will tell you.

Hugging Face `transformers` is the last independent implementation you get. Use
it properly once, now.

## Before you start

This chapter assumes:

- **Chapter 0a's notation.** Tensor shapes are written `(batch, seq, hidden)`.
  Einstein summation, softmax, bfloat16, and the GPU vocabulary (SM, HBM, L2)
  are all defined there.
- **The five components from chapters 3 to 7.** Token embeddings, RMSNorm,
  rotary position embeddings, the gated delta rule, and grouped-query
  attention. You do not need to remember their internals, but you do need to
  know their signatures and the shapes they accept.
- **PyTorch module composition.** `nn.Module`, `nn.ModuleList`, `nn.Linear`,
  `nn.Embedding`, and the fact that `nn.Linear(in_features, out_features)`
  stores a `weight` of shape `(out_features, in_features)` and computes
  $y = x W^\top$.
- **The model geometry from chapter 2.** Hidden size 5120, 64 layers, 24 query
  heads and 4 KV heads of width 256, intermediate size 17408, vocabulary
  248,320, every fourth layer full attention.

Two symbols recur. $h = 5120$ is the hidden size, the width of the residual
stream. $i = 17408$ is the intermediate size, the width inside the MLP.

## The block

Every one of the 64 layers has the same skeleton: two pre-norm sublayers, each
wrapped in a residual connection.

$$
x \leftarrow x + \operatorname{mixer}\big(\operatorname{norm}_1(x)\big)
$$

$$
x \leftarrow x + \operatorname{MLP}\big(\operatorname{norm}_2(x)\big)
$$

*Pre-norm* means the normalization sits inside the residual branch, not after
the addition. The residual stream itself is never normalized, so a gradient (or,
at inference time, a signal) can travel from layer 0 to layer 63 without passing
through a single scaling operation. Post-norm stacks, which normalize after the
addition, need a warmup schedule to train at this depth. Every modern decoder is
pre-norm.

In code that is four lines, and it is the same four lines everywhere:

```python
residual = x
hidden = self.input_layernorm(x)            # (batch, seq, 5120)
hidden = self.mixer(hidden, ...)            # (batch, seq, 5120)
x = residual + hidden                       # (batch, seq, 5120)
return x + self.mlp(self.post_attention_layernorm(x))
```

Only `self.mixer` changes down the stack. `config.layer_types` decides which one
a layer gets:

```python
self.is_full_attention = config.layer_types[layer_idx] == "full_attention"
```

### A full-attention layer, with shapes

Sixteen layers take this path. Batch $B$, $T$ tokens in this forward pass, $L$
tokens of context including the ones just written to the cache.

| Step | Tensor | Shape |
|---|---|---|
| Input | `x` | `(B, T, 5120)` |
| `input_layernorm` | `hidden` | `(B, T, 5120)` |
| `q_proj` | `q` | `(B, T, 12288)` |
| `chunk(2)` | `q`, `gate` | `(B, T, 6144)` each |
| view and transpose | `q` | `(B, 24, T, 256)` |
| `k_proj`, `v_proj` | `k`, `v` | `(B, 4, T, 256)` |
| `q_norm`, `k_norm` | unchanged | `(B, 24, T, 256)`, `(B, 4, T, 256)` |
| RoPE on the first `rotary_dim` channels | unchanged | as above |
| `cache.append` | `k`, `v` | `(B, 4, L, 256)` |
| attention | `out` | `(B, 24, T, 256)` |
| transpose and reshape | `out` | `(B, T, 6144)` |
| multiply by `sigmoid(gate)` | `out` | `(B, T, 6144)` |
| `o_proj` | `out` | `(B, T, 5120)` |

The query projection is 12288 wide, not 6144, because the second half is the
output gate from chapter 7. The keys and values are 1024 wide because there are
only 4 KV heads.

### A linear-attention layer, with shapes

Forty-eight layers take this path instead.

| Step | Tensor | Shape |
|---|---|---|
| Input | `x` | `(B, T, 5120)` |
| `in_proj_qkvz` | `qkvz` | `(B, T, 16384)` |
| split into `q`, `k`, `v`, `z` | | `2048`, `2048`, `6144`, `6144` |
| causal depthwise conv over `q,k,v` | `mixed` | `(B, T, 10240)` |
| view into heads | `q`, `k` | `(B, 16, T, 128)` |
| | `v` | `(B, 48, T, 128)` |
| `repeat_interleave` q and k 3 times | `q`, `k` | `(B, 48, T, 128)` |
| `in_proj_ba`, then gates | `alpha`, `beta` | `(B, 48, T)` |
| delta rule | `out` | `(B, 48, T, 128)` |
| | `state` | `(B, 48, 128, 128)` |
| gated RMSNorm with `silu(z)` | `out` | `(B, T, 48, 128)` |
| `out_proj` | `out` | `(B, T, 5120)` |

Both mixers consume `(B, T, 5120)` and return `(B, T, 5120)`. That is the whole
contract. The residual stream never changes width, which is what lets you swap
one mixer for the other without touching anything around it.

## The layer schedule

`full_attention_interval` is 4, and a layer is full attention when its index
$\ell$ satisfies

$$
(\ell + 1) \bmod 4 = 0 .
$$

So the full-attention layers are $\ell \in \{3, 7, 11, \ldots, 63\}$, which is
16 of them, and the other 48 are linear attention. The pattern repeats every
four layers:

| Layers | Mixer |
|---|---|
| 0, 1, 2 | Gated delta linear attention |
| 3 | Grouped-query attention, with a KV cache |
| 4, 5, 6 | Gated delta linear attention |
| 7 | Grouped-query attention, with a KV cache |
| … | … |
| 60, 61, 62 | Gated delta linear attention |
| 63 | Grouped-query attention, with a KV cache |

Two details fall out of this and both matter later.

The **last** layer is full attention. Whatever the final layer contributes to
the residual stream, it contributes with access to every past token
individually, not through a compressed state.

The **first three** layers are not. A bug in your full-attention path cannot
show up before layer 3, so if the residual stream already disagrees at layer 0,
the cause is upstream of both mixers: the embedding, or the first norm.

`ModelConfig` exposes the schedule directly, and you should read it from there
rather than recomputing the modulus in three places:

```python
config.full_attention_layers    # [3, 7, 11, ..., 63]
config.linear_attention_layers  # [0, 1, 2, 4, 5, 6, 8, ...]
```

## The MLP

The mixer moves information between positions. The MLP moves information
between channels, at each position independently. It is also where most of the
model's parameters live, so it is worth deriving rather than copying.

### From a feed-forward network to a gate

The original transformer's feed-forward block is two matrices with a
nonlinearity between them:

$$
\operatorname{FFN}(x) = W_{\text{down}}\,\phi\big(W_{\text{up}}\,x\big)
$$

where $x \in \mathbb{R}^{h}$ is one position's residual vector,
$W_{\text{up}} \in \mathbb{R}^{i \times h}$, $W_{\text{down}} \in
\mathbb{R}^{h \times i}$, and $\phi$ is applied elementwise. The intermediate
width $i$ is conventionally $4h$.

A *gated linear unit* replaces the single nonlinearity with a product of two
projections, one of which is passed through a nonlinearity and the other of
which is not:

$$
\operatorname{GLU}(x) = \big(W_{\text{up}}\,x\big) \odot \phi\big(W_{\text{gate}}\,x\big)
$$

Here $\odot$ is the elementwise product. The intuition is multiplicative
routing. In the ungated form, channel $j$ of the intermediate vector is a fixed
function of $x$. In the gated form it is a product of two functions of $x$, so
one of them can drive the other to zero. A channel can switch itself off based
on a direction in $x$ that has nothing to do with the value it would have
carried. A sum of two linear maps cannot do that; a product can.

### Why SiLU

SwiGLU is the GLU whose $\phi$ is SiLU, also called Swish:

$$
\operatorname{SiLU}(z) = z\,\sigma(z) = \frac{z}{1 + e^{-z}}
$$

where $\sigma$ is the logistic sigmoid. Three properties earn it the slot.

It is **smooth everywhere**. ReLU has a kink at $z = 0$. SiLU's derivative,

$$
\operatorname{SiLU}'(z) = \sigma(z)\big(1 + z\,(1 - \sigma(z))\big),
$$

is continuous, so the function has no point where a tiny change in the input
changes the local behavior discontinuously.

It is **non-monotonic**. Setting $\operatorname{SiLU}'(z) = 0$ gives
$1 + z - z\,\sigma(z) = 0$, whose root is $z^{\ast} \approx -1.278$, where
$\operatorname{SiLU}(z^{\ast}) \approx -0.278$. So SiLU dips slightly negative
before returning to zero. Small negative pre-activations produce a small
negative output rather than being erased, which keeps a little signal in a
region ReLU discards entirely.

It **saturates to the right shapes**. As $z \to -\infty$,
$\operatorname{SiLU}(z) \to 0$; as $z \to +\infty$, $\operatorname{SiLU}(z) \to
z$. So it behaves like ReLU at both extremes and differs only near the origin,
which is exactly where a hard kink is most costly.

### The block

Putting it together, with the down projection applied last:

$$
\operatorname{SwiGLU}(x) = W_{\text{down}}\Big(\operatorname{SiLU}\big(W_{\text{gate}}\,x\big) \odot \big(W_{\text{up}}\,x\big)\Big)
$$

That is one line of PyTorch, and `engine/layers/mlp.py` is barely longer:

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):                 # x: (batch, seq, 5120)
        gate = self.gate_proj(x)          # (batch, seq, 17408)
        up   = self.up_proj(x)            # (batch, seq, 17408)
        return self.down_proj(F.silu(gate) * up)   # (batch, seq, 5120)
```

No biases. Every projection in this model is bias-free, which is standard for
decoders trained after about 2021: with a normalization layer immediately
upstream, the bias adds parameters and buys nothing measurable.

### Why 17408

The conventional intermediate size is $4h = 20480$. This model uses 17408.
Two forces set that number, and they pull in opposite directions.

**Parameters.** A GLU has three matrices where an ungated FFN has two. An
ungated block at $4h$ holds

$$
2 \cdot h \cdot 4h = 8h^{2} = 209{,}715{,}200
$$

parameters. A gated block holds $3 h i$. Keeping the count equal would require
$i = \tfrac{8}{3}h \approx 13{,}653$, and that two-thirds rule is what the
original SwiGLU paper used to compare fairly against an ungated baseline.

This model does not follow the rule. At $i = 17408 = 3.4h$ the block holds

$$
3 \cdot 5120 \cdot 17408 = 267{,}386{,}880
$$

parameters, which is 27.5% more than the ungated block it replaces. The
architecture spends that on the gate deliberately: parameters in the MLP are
cheap to serve, because the three matrices are read once per forward pass no
matter how many tokens are in it. Chapter 10 makes that statement precise.

**Alignment.** $17408 = 17 \times 1024$. It is divisible by 1024, so it is
divisible by 128 and by 256, which are the tile widths a bfloat16 tensor-core
GEMM wants. It also splits evenly for tensor parallelism: across 8 GPUs each
shard is $17408 / 8 = 2176$, still a multiple of 128; across 16 GPUs each shard
is 1088, a multiple of 64. The two-thirds answer of 13,653 has none of those
properties, and the nearest aligned value would have to be chosen by hand
anyway.

Across 64 layers the MLP holds $64 \times 267{,}386{,}880 = 17.1$ billion
parameters, which is 64% of the model. Chapter 2 counted it; now you know where
the three matrices came from.

## The head

After layer 63 the residual stream gets one more normalization and one matrix
multiply.

```python
x = self.norm(x)                     # (batch, seq, 5120)
if last_token_only:
    x = x[:, -1:, :]                 # (batch, 1, 5120)
return self.lm_head(x)               # (batch, seq_out, 248320)
```

The final norm exists for the same reason the per-layer norms do. Nothing has
constrained the scale of the residual stream since the embedding; 64 residual
additions can leave it at any magnitude. The output projection was trained
against a normalized input, so it needs one.

`lm_head` is `nn.Linear(5120, 248320, bias=False)`. It holds

$$
5120 \times 248{,}320 = 1{,}271{,}398{,}400
$$

parameters, 2.54 GB in bfloat16, and `tie_word_embeddings` is false so the
embedding table is a second, separate matrix of the same size. Between them
the vocabulary costs 2.54 billion parameters, 9.5% of the model, before a
single layer runs.

### Slice before you project

The `last_token_only` slice is not a micro-optimization. During prefill of 2048
tokens, projecting every position costs

$$
2 \cdot 2048 \cdot 5120 \cdot 248{,}320 \approx 5.21 \times 10^{12}
$$

FLOPs, which at the A100's 312 TFLOP/s is a floor of 16.7 ms, and produces a
logits tensor of $2048 \times 248{,}320 \times 2 = 1.02$ GB. You need one row of
it: the last position's, which is what predicts the first generated token.
Slicing first drops the cost to $2 \cdot 5120 \cdot 248{,}320 = 2.54$ GFLOPs and
a 497 KB output. The arithmetic above is a roofline bound, not a measurement;
chapter 10 explains how to compute one.

### What a logit is

The output is a vector $z \in \mathbb{R}^{V}$ per position, with $V = 248{,}320$.
It is an unnormalized log-probability. The distribution over the next token is

$$
p_j = \frac{e^{z_j}}{\sum_{m=1}^{V} e^{z_m}} .
$$

Two consequences are worth holding on to.

**Softmax is shift-invariant.** Replacing $z$ by $z + c\mathbf{1}$ for any
scalar $c$ leaves $p$ unchanged, because $e^{c}$ factors out of numerator and
denominator. That is why every softmax implementation subtracts
$\max_j z_j$ first: it costs nothing and it stops $e^{z_j}$ from overflowing.

**Only differences carry information.** The absolute magnitude of a logit is not
meaningful on its own, which is why chapter 11 works in terms of gaps between
the top logits rather than their values. In this model, trained logits typically
land between about 10 and 30 in magnitude.

## Loading weights

The tensor names in the checkpoint will not match your module names. Write an
explicit mapping rather than trying to make your class hierarchy mirror the
file. An explicit map is easier to read, and it fails loudly when a name is
missing instead of silently leaving a tensor at its initialized value.

`engine/weights.py` reads `model.safetensors.index.json`, which maps each tensor
name to the shard holding it, and serves tensors by name from a memory-mapped
file. Opening a shard costs nothing; you pay only for the tensors you read. That
is what makes it possible to load one layer at a time, move it to the GPU, and
never hold more than one layer's worth of weights in host memory.

Three checks catch nearly every loading bug.

**Every checkpoint tensor is consumed.** Track which names you read and assert
the set is complete at the end. An unconsumed tensor means you skipped a
component: a norm, a gate, a convolution weight.

**Every parameter is written.** Build the model on the `meta` device, or fill
every parameter with NaN before loading. A parameter still NaN afterwards tells
you immediately. A parameter left at its random initialization does not — it
produces output that is slightly worse in a way you might not notice for hours.

**Shapes match exactly.** `nn.Linear(in_features, out_features).weight` has
shape `(out_features, in_features)`, the transpose of the mathematical
convention. Checkpoints usually follow the same convention, so a direct
assignment is right. A transposed load raises only when the two dimensions
differ, and for a square matrix it does not raise at all. It computes
something else instead.

## Validating against the reference

Load the same weights into `transformers`, run both models on the same token
ids, and compare the outputs.

Compare **logits, not generated text**. Greedy decoding takes an argmax, which
discards everything except the ranking of the top element. Two implementations
can produce the same 200 tokens while disagreeing badly about the distribution
underneath, and that disagreement will surface the moment you turn on
temperature, or top-p, or speculative decoding. Three numbers:

```python
max_abs_diff = (mine - reference).abs().max()
correlation  = torch.corrcoef(
    torch.stack([mine.flatten(), reference.flatten()]))[0, 1]
top1_match   = (mine.argmax(-1) == reference.argmax(-1)).float().mean()
```

The lab adds a fourth, the mean KL divergence from the reference distribution to
yours:

$$
D_{\mathrm{KL}}(p \,\|\, q) = \sum_{j=1}^{V} p_j \,\log \frac{p_j}{q_j}
$$

with $p$ from the reference logits and $q$ from yours, averaged over positions.
KL is the right summary because it weights a disagreement by how much
probability the reference actually assigned there. A large difference on a token
the reference gave $10^{-9}$ probability is not a bug worth chasing; the same
difference on the top token is.

Compute all four in float32 whatever the model dtype is. `corrcoef` over
bfloat16 will quietly lose the last digits you care about.

Expect a max absolute difference in the region of $10^{-2}$ on logits of
magnitude 10 to 30, correlation above 0.9999, and top-1 agreement of 1.0. Of the
three, the max is the least stable — it is one worst entry out of 248,320 — and
the correlation is the most sensitive. Judge by correlation and KL.

### Why bit-exact agreement is impossible

Two reasons, and neither is a bug you can fix.

**Floating-point addition is not associative.** For finite-precision values,
$(a + b) + c \ne a + (b + c)$ in general, because each addition rounds. A matmul
of inner dimension 5120 is a sum of 5120 products, and the order in which a
kernel accumulates them depends on its tile size, its split-K strategy, and how
many warps it assigns to the reduction. Your kernel and cuBLAS's kernel will
choose differently, so they will land on different sums of the same numbers.

**bfloat16 has 8 bits of significand.** One sign bit, 8 exponent bits, 7 stored
mantissa bits, plus the implicit leading 1. The relative spacing between
representable values is therefore

$$
\varepsilon = 2^{-8} \approx 3.9 \times 10^{-3} .
$$

Tensor cores accumulate in float32 and round the result back to bfloat16 at each
matmul's output, so every one of the model's several hundred matmuls introduces
a rounding of that relative size. Those roundings are not errors relative to a
"true" answer either implementation is trying to hit; they are two different,
equally valid roundings.

The consequence for testing: never assert equality, always assert a tolerance,
and pick the tolerance from the dtype rather than from what happened to pass
once.

## Bisecting a mismatch

When the logits disagree, do not stare at the logits. Find the first layer where
the residual streams diverge; everything after it is downstream of one bug.

Register a forward hook on each decoder layer of both models, run one prompt,
and collect the outputs into two lists in layer order. Then walk them:

```python
def first_divergent_layer(mine, reference, tol=1e-2):
    for index, (a, b) in enumerate(zip(mine, reference)):
        if (a.float() - b.float()).abs().max().item() > tol:
            return index
    return None
```

The tolerance is a threshold, not a physical constant. Set it well above the
bfloat16 noise floor for a hidden state — $10^{-2}$ on a residual stream whose
entries are order 1 is roughly three times $\varepsilon$ — and well below the
size of a real bug. If every layer is just over the line, your tolerance is too
tight; if the first divergence is layer 63, it is too loose.

A rough guide to what the first bad layer tells you:

| First divergence | Likely cause |
|---|---|
| Layer 0, immediately | Embedding lookup or a transposed weight. |
| Layer 3, the first full-attention layer | The output gate split, or the mask offset. |
| Layer 0 but only past a fixed position | The partial rotary boundary. |
| Gradual drift across all layers | A norm reducing in bfloat16 instead of float32. |
| Correct for one token, wrong after | Cache indexing or the decode position. |

## Four bugs that produce plausible output

The dangerous failures are the ones that still generate English. Each of these
passes a casual read of the output and fails the reference comparison.

**A transposed weight.** You assign `module.weight = checkpoint_tensor.T`, or
you forget that a checkpoint stores $(\text{out}, \text{in})$ and transpose it
to "fix" a shape you misread. If the matrix is square — and in this model
`o_proj` at $6144 \times 5120$ is not, but several norms and the delta-rule
state are — nothing raises. The symptom is a model that is fluent and
semantically wrong: correct grammar, confident tone, no relationship to the
prompt. Bisection points at layer 0 or at whichever layer owns the matrix.

**RoPE on the wrong axis.** RoPE pairs channels within a head and rotates each
pair by an angle that depends on position. Apply it after the transpose to
`(batch, heads, seq, dim)` when your cosine table is indexed for
`(batch, seq, heads, dim)` and you rotate by head index instead of by position.
The symptom is a model with no sense of order: it produces plausible words with
scrambled syntax, and it gets *worse* with longer prompts rather than better.
The check is cheap — rotate a single token at position 0 and confirm it comes
back unchanged, because the angle is zero there.

**A mask off by one.** With a cached prefix, query row $r$ of the current block
sits at absolute position $r + (L - T)$, where $L$ is the total context and $T$
the number of new tokens. Drop that offset and the mask is correct during
prefill, when $L = T$, and wrong on every decode step. Let a query see one
position into the future and the model leaks the answer to itself during prefill
and produces confident nonsense during decode. The symptom is a model that
handles its prompt well and then drifts within a few dozen tokens.

**A norm applied to the wrong tensor.** Pre-norm means
`x + mixer(norm(x))`, not `norm(x + mixer(x))` and not `norm(x) + mixer(norm(x))`.
The second is post-norm, the third normalizes the residual stream itself. Both
run, both produce finite numbers, and both are a different model. The symptom is
degradation that compounds with depth, so bisection shows a small divergence at
layer 0 growing steadily to a large one at layer 63 — the signature in the table
above.

## Two forward passes, one model

Prefill and decode run the same code with different shapes: $T$ large and the
cache empty, versus $T = 1$ and the cache holding everything. They must produce
the same numbers.

The test is direct. Run a full forward pass over a sequence with no cache. Then
create a cache, prefill a prefix, feed the remaining tokens one at a time, and
compare the logits at matching positions.

```python
reference = model(input_ids)                       # (1, T, vocab)
cache = make_cache()
model(input_ids[:, :prefill_len], cache=cache, last_token_only=True)
for position in range(prefill_len, input_ids.shape[1]):
    logits = model(input_ids[:, position:position + 1],
                   cache=cache, last_token_only=True)
    worst = max(worst,
                (logits[:, -1] - reference[:, position]).abs().max().item())
```

That single test exercises the KV cache append, the linear-attention state
carry, the convolution window, the mask offset, and the decode position all at
once. It is the highest-value test in the engine, and the reference
implementation passes it:

```text
token-by-token vs full-forward maxerr: 5.96e-07
```

That figure is a float32 run on a small config, which is why it is seven orders
of magnitude tighter than the bfloat16 logit comparison above. The two numbers
are measuring different things: this one asks whether one implementation agrees
with itself under a reshape, and the answer should be yes to near machine
precision.

## Check your understanding

**Layer 12 of the model — which mixer does it use, and does it own a KV cache?**

Linear attention, and no. A layer is full attention when $(\ell + 1) \bmod 4 =
0$; $13 \bmod 4 = 1$, so layer 12 is one of the 48 linear-attention layers. It
carries a fixed-size recurrent state and a three-step convolution window
instead, neither of which grows with context.

**Your implementation and the reference agree to correlation 0.99999 and top-1
agreement 1.0, but the max absolute difference is 0.4. Is that a bug?**

Probably not, and the way to settle it is the KL divergence. A max of 0.4 on a
248,320-wide vector is one entry; if it sits on a token the reference gave
negligible probability, the KL will be tiny and the difference is bfloat16
noise on a logit that never mattered. If the KL is large, the disagreement is on
a token with real probability mass, and the correlation is hiding it because
correlation over 248,320 values is dominated by the bulk.

**Why must the cache cursor advance after all 64 layers rather than inside each
layer?**

Because every layer in one forward pass writes at the same position. Advancing
inside a layer makes layer 4 believe the sequence is one token longer than layer
3 does, so it writes to the wrong slot and computes its mask offset against the
wrong length. Chapter 9 builds the cache that enforces this.

**Prefill of 4096 tokens with `last_token_only=False` runs out of memory on an
80GB card, with the weights taking only 53.8 GB. Where did the rest go?**

The logits. $4096 \times 248{,}320$ values in bfloat16 is 2.03 GB for the output
alone, and the float32 copy a downstream `log_softmax` makes is another 4.07 GB,
on top of an activation workspace that already holds two $4096 \times 17408$
MLP intermediates. Slicing to the last position removes the largest of these
outright.

## Lab

Write the three validation tools and use them on the engine's own model.

`logit_metrics(mine, reference)` returns `max_abs_diff`, `correlation`,
`top1_agreement`, and `mean_kl`, computed in float32. The harness checks it
three ways: identical inputs must give correlation 1.0, top-1 agreement 1.0, and
zero KL; a perturbation of 0.01 must stay above 0.999 correlation with a small
positive KL; unrelated logits must correlate near zero and agree on the argmax
at roughly chance.

`first_divergent_layer(mine, reference, tol)` returns the index of the first
layer exceeding the tolerance, or `None`. The harness plants a divergence at
layer 5 and checks you find it, and checks that identical states report nothing.

`prefill_decode_equivalence(model, input_ids, prefill_len, make_cache)` runs the
equivalence test above against `engine.model.HybridLanguageModel` on a small
hybrid config, and must return a maximum absolute error below $10^{-4}$.

The lab uses a tiny 8-layer configuration rather than the 27B model, because a
wrong answer in three seconds teaches you more than a wrong answer in four
minutes. The geometry is scaled down but the structure is identical: 8 layers
with full attention at 3 and 7, an output gate, partial rotary, and both kinds
of cache state.

## Further reading

- [The transformers modeling code for Qwen3](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)
- [GLU variants improve transformer](https://arxiv.org/abs/2002.05202) — where SwiGLU comes from.
- [Searching for activation functions](https://arxiv.org/abs/1710.05941) — the Swish/SiLU paper.
- [On layer normalization in the transformer architecture](https://arxiv.org/abs/2002.04745) — why pre-norm.
- [What every computer scientist should know about floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
