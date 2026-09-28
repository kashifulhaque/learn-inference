---
title: The model on disk
slug: 01-the-model-on-disk
part: "Part 1 — Ground truth"
summary: How a safetensors file lays out its bytes, how the shard index finds any tensor, why memory mapping lets you load one layer at a time, and what the tensor names tell you about the architecture.
minutes: 65
gpu: false
objectives:
  - Parse a safetensors header by hand and work out where any tensor's bytes live.
  - Use a safetensors index to find any tensor without loading the model.
  - Infer a model's architecture from its tensor names and shapes alone.
  - Explain why memory mapping makes lazy, layer-by-layer loading possible.
lab: 01-inspect-weights
---

# The model on disk

> [!TLDR]
> - A safetensors file is a short length field, a JSON table of contents, and
>   then raw numbers. You can read the table of contents alone and learn every
>   tensor's name, number format, and shape without reading any weights.
> - Memory mapping lets the engine pull the model onto the GPU one layer at a
>   time, so the CPU side needs room for about one layer (0.8 GB), not the
>   whole 54 GB model.
> - The model is split across 18 files, and a small index file says which file
>   holds each tensor.
> - The tensor names and shapes describe the architecture more reliably than the
>   model card does. They show that the model mixes two kinds of layer, and that
>   one projection is twice the width you'd expect.

Qwen3.8-27B, the model this course serves, ships as a folder of files: 18
weight files called *shards*, a JSON index that says what's in each shard, a
tokenizer, and a config. Together they're about 54 GB.

Before you write a single layer, look at what those files actually contain. The
tensor names are ==the most honest description of the architecture== you can
get, and they often disagree with the model card.

This is the only chapter where you touch bytes on disk. Every later chapter
assumes you can ask "what shape is layer 37's key projection?" and get an
answer in milliseconds, without loading 54 GB.

## Before you start

Three ideas carry this chapter.

**A file is an array of bytes, and a tensor is a contiguous run of them.** A
tensor has a *dtype*, its number format, which fixes how many bytes each element
takes and how to read them. It also has a shape. Multiply the shape out, then
multiply by the element size, and you have the tensor's length in bytes. Nothing
else is stored.

**Row-major means the last index moves fastest.** A tensor of shape
`(248320, 5120)` stores row 0's 5120 values first, then row 1's, and so on. So
the element at `[i, j]` sits at offset $(i \times 5120 + j) \times
\text{bytes per element}$ from the start. This layout is also called C order,
and it's the only one safetensors stores.

**Memory mapping makes a file look like memory.** The `mmap` system call asks
the operating system kernel to reserve a range of addresses backed by a file
rather than by RAM, and it reads nothing yet. When your program touches an
address in that range, the kernel fetches the 4 KB page that contains it, either
from disk or from the *page cache*, the kernel's in-RAM copy of recently read
files. Touch a hundred bytes of a 3 GB file and you've read a hundred bytes, not
3 GB.

If any of that is unfamiliar, or if the difference between GB and GiB is, the
[notation chapter](/c/00a-notation-and-prerequisites) covers both.

## The safetensors format

This section answers the first question an engine asks of a weight file: given a
tensor's name, where are its bytes? Once you know the layout, the answer is one
addition.

Think of a safetensors file as a book whose table of contents lists page ranges
counted from the first page of chapter 1, not from the front cover. To find a
tensor, you skip the cover and the table of contents, then count forward.

Concretely, the file has three parts, in this order:

```text
[ 8 bytes ][ N bytes of JSON header ][ raw tensor bytes ]
```

- **The first 8 bytes** hold one number, $N$: the length of the header. It's
  stored as an unsigned 64-bit integer, least significant byte first
  (little-endian).
- **The next $N$ bytes** are the header, written as UTF-8 JSON.
- **Everything after that** is a flat buffer of tensor data. There are no
  separators, no padding between tensors, and no metadata of any kind.

The header maps each tensor name to three fields:

```json
{
  "model.language_model.layers.3.self_attn.k_proj.weight": {
    "dtype": "BF16",
    "shape": [1024, 5120],
    "data_offsets": [0, 10485760]
  },
  "__metadata__": {"format": "pt"}
}
```

`data_offsets` is a half-open byte range, `[begin, end)`: it includes `begin`
and stops just before `end`. The range is measured ==from the end of the
header==, not from the start of the file. To find a tensor's first byte in the
file, skip the length prefix and the header, then add `begin`:

$$
\text{absolute offset} = \hla{8} + \hlb{N} + \hlc{\text{begin}}
$$

That's the $\hla{8}$-byte length prefix, the $\hlb{N}$-byte header, and the
tensor's own $\hlc{\text{begin}}$.

Because nothing else is stored, the range's length has to match the shape and
the dtype exactly:

$$
\boxed{\text{end} - \text{begin} = \text{elements in shape} \times \text{bytes per element}}
$$

That identity always holds. Checking it is how you catch a truncated or corrupt
file.

> [!EXAMPLE] Check the `k_proj` entry
> The shape is `1024 x 5120`, and `BF16` (bfloat16) takes two bytes per
> element:
>
> $$
> 1024 \times 5120 \times 2 = 10{,}485{,}760\ \text{bytes}
> $$
>
> That's exactly $\text{end} - \text{begin}$. Because `begin` is 0, this
> tensor's first byte sits at $8 + N$ in the file.

The `__metadata__` key is the one reserved name. It holds a flat map from
strings to strings, and no tensor data.

## Why the format looks like this

Next, you see why the layout is this bare. Each of the following four properties
follows from it, and each one is something your engine relies on:

1. **Listing tensors is one small read.** Read 8 bytes, read $N$ more, and parse
   the JSON. You now know every tensor's name, dtype, and shape without touching
   a byte of weight data. The lab depends on this.
2. **There are no strides.** A *stride* is the step between neighbouring
   elements in memory, which lets a framework describe a transposed or sliced
   view without moving data. Safetensors has none: every tensor is stored
   C-contiguous, so a framework that wants to save a transposed view must copy it
   into that order first. Reading is offset arithmetic and nothing else, and the
   shape in the header is the shape in memory.
3. **Reading a tensor is a memory map, not a copy.** The loader maps the file and
   hands you a tensor whose storage points into the mapping. Pages arrive when
   you touch them. Moving that tensor to the GPU is what triggers the read, one
   tensor at a time.
4. **Nothing executes.** The header is JSON with a fixed schema, and the rest is
   numbers.

> [!NOTE] Why not `torch.save`?
> `torch.save` uses pickle. Reading anything requires deserializing arbitrary
> Python objects, so you must trust the file, and you must load the whole thing
> before you can inspect any of it. Safetensors exists mostly because "download a
> stranger's weights" shouldn't mean "run a stranger's code."

The dtype strings are the format's own names, not PyTorch's: `F64`, `F32`,
`F16`, `BF16`, `I64`, `I32`, `I16`, `I8`, `U8`, and `BOOL`. This model's weights
are all `BF16`.

When a loader converts dtypes, it does so after mapping, so ==a conversion costs
a full copy==. That's one reason to keep the engine in bfloat16 end to end.

## Why memory mapping matters here

How do you get 54 GB onto the GPU without first needing 54 GB of CPU RAM?
Memory mapping lets you load the model one layer at a time instead.

The naive way to load a 54 GB model onto an 80 GB GPU is to read the whole
checkpoint into *host memory*, the CPU's RAM, and then copy it across. That
needs 54 GB of RAM you might not have. It also serializes the work: nothing
reaches the GPU until everything has been read.

With a mapped file you can go layer by layer. In the following loop, the comment
`H2D copy` means a host-to-device copy, from CPU RAM to GPU memory over PCIe,
the bus that connects them:

```python
for layer_idx in range(config.num_hidden_layers):
    for suffix, name in index.layer_tensors(layer_idx):
        tensor = index.get(name, device="cuda")   # page in, then H2D copy
```

Peak host memory is now ==one layer, about 0.8 GB, not 54 GB==. The kernel
reclaims pages behind you once nothing references them.

Pages still in the page cache from a previous run come back for free. That's why
the second load of a model is much faster than the first, even though your code
cached nothing explicitly.

Two gotchas:

- **Mapped tensors are read-only in practice.** Mapped pages are shared, so if
  you write to one, you either fault or dirty the page cache.
- **A mapped tensor on the CPU still reads from disk on first touch.** If you
  time a load and get "instant," you measured the mapping, not the read.

## Shards and the index

You can read one file. With 18 of them, how do you know which one to open?

A single 54 GB file is awkward to download and impossible to resume, so the model
is split into 18 shards of roughly 3 GB each. The file
`model.safetensors.index.json` maps each tensor name to the shard that holds it:

```json
{
  "metadata": {"total_size": 53974302720},
  "weight_map": {
    "model.language_model.layers.0.mlp.gate_proj.weight": "model-00001-of-00018.safetensors",
    "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001-of-00018.safetensors"
  }
}
```

`total_size` is the sum of every tensor's bytes, not the sum of the file sizes,
so the headers aren't counted. Chapter 2 counts 53.79 GB for the language model
alone. The index reports a little more, because the checkpoint also carries
tensors outside the text path.

To load a tensor, look up its shard in `weight_map`, open that shard, and read
the byte range. `WeightIndex` in `engine/weights.py` does exactly this. It also
caches the open file handles, so that loading 64 layers doesn't reopen 18 files
64 times:

```python
class WeightIndex:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        index_file = self.dir / "model.safetensors.index.json"
        if index_file.is_file():
            self.map = json.loads(index_file.read_text())["weight_map"]
        else:
            single = "model.safetensors"
            with safe_open(self.dir / single, framework="pt") as f:
                self.map = {name: single for name in f.keys()}
        self._handles = {}
```

The `else` branch matters for small models, which ship as one file with no
index. Building the map from `f.keys()` gives the rest of the class one code
path instead of two.

The `get` method fetches one tensor:

```python
    def get(self, name, device="cpu", dtype=None):
        shard = self.map[name]
        tensor = self._handle(shard).get_tensor(name)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(device)
```

`get_tensor` returns a view into the mapping, so it copies nothing. The two
`.to()` calls are where any real work happens:

- A dtype change copies the tensor.
- A device change copies it across PCIe.

Calling `get` with `device="cuda"` and no dtype is the cheap path: one copy,
straight from the page cache to the GPU.

To inspect tensors without loading them, `describe` uses a slice handle instead:

```python
with safe_open(index.dir / index.map[name], framework="pt") as f:
    slice_ = f.get_slice(name)
    rows.append({"name": name, "shape": list(slice_.get_shape()),
                 "dtype": slice_.get_dtype()})
```

> [!KEY] `get_slice` never touches the data buffer
> `get_slice` reads the header entry and returns an object that knows the shape
> and dtype. The data buffer stays untouched. This is the operation the lab is
> built around.

## Reading the architecture from the names

With the plumbing in place, you can ask the more interesting question: what does
the model look like inside? Group the tensor names by layer index, and the model
describes itself.

A little vocabulary first. Every layer of this model has a *token mixer*, the
part that lets each position look at earlier positions, followed by an MLP. This
model uses two kinds of mixer:

- **Full attention** (`self_attn`), standard transformer attention. Each
  position compares a *query* vector against the *keys* of all earlier
  positions, then averages their *values*. The work is split across several
  independent *heads*.
- **Linear attention** (`linear_attn`), which replaces that comparison with a
  fixed-size running summary. Chapter 6 covers it in depth.

In this checkpoint, layer 0 and layer 3 hold different things:

```text
# Layer 0 — linear attention
model.language_model.layers.0.linear_attn.in_proj_qkvz.weight   [16384, 5120]
model.language_model.layers.0.linear_attn.in_proj_ba.weight     [96, 5120]
model.language_model.layers.0.linear_attn.conv1d.weight         [10240, 1, 4]
model.language_model.layers.0.linear_attn.A_log                 [48]
model.language_model.layers.0.linear_attn.dt_bias               [48]
model.language_model.layers.0.linear_attn.norm.weight
model.language_model.layers.0.linear_attn.out_proj.weight       [5120, 6144]

# Layer 3 — full attention
model.language_model.layers.3.self_attn.q_proj.weight           [12288, 5120]
model.language_model.layers.3.self_attn.k_proj.weight           [1024, 5120]
model.language_model.layers.3.self_attn.v_proj.weight           [1024, 5120]
model.language_model.layers.3.self_attn.o_proj.weight           [5120, 6144]
```

Work through the full-attention layer first, because the reasoning is the same
everywhere. Take the four shapes one at a time.

**Linear weights are stored output-first.** PyTorch's `nn.Linear` holds a weight
of shape `(out_features, in_features)` and computes $y = x W^\top$. So the second
dimension is always the input width. Here it's 5120, the hidden size: the width of
the vector that represents each token between layers. All four shapes confirm
it.

**The key projection tells you the key and value geometry.** `k_proj` has 1024
output channels. The config says `head_dim`, the width of one head, is 256. So
$1024 / 256 = 4$ key heads. The same holds for `v_proj` and its value heads.

**The output projection tells you the query geometry.** `o_proj` maps 6144
channels back to 5120, and $6144 / 256 = 24$ query heads.

With 24 query heads and 4 key heads, six query heads share each key head. Sharing
heads like this is called *grouped-query attention* (GQA), six is its group
size, and chapter 7 is about what it buys.

**The query projection is a surprise.** With 24 heads of width 256, you expect
$24 \times 256 = 6144$ output channels. The file says 12288, exactly double.
That's the *output gate*: the projection emits the queries and a gate side by
side, and the layer splits them apart. `config.json` confirms it with
`attn_output_gate: true`.

> [!WARNING] Trust the file, not the shape arithmetic
> If you size `q_proj` from the head counts instead of from the file, you write
> a layer that loads without error and produces nonsense.

## The linear-attention layer

The linear-attention layer needs more decoding, and this is where reading the
names really earns its keep.

Start from what the config tells you: `linear_num_key_heads` is 16,
`linear_num_value_heads` is 48, and both head dims are 128. The name
`in_proj_qkvz` suggests four blocks: queries (q), keys (k), values (v), and a
gate (z). Queries and keys get one block per key head. Values and the gate get
one block per value head. Add them up:

$$
\underbrace{\hla{16 \times 128}}_{q} + \underbrace{\hla{16 \times 128}}_{k}
+ \underbrace{\hlb{48 \times 128}}_{v} + \underbrace{\hlb{48 \times 128}}_{z}
= 2048 + 2048 + 6144 + 6144 = \boxed{16384}
$$

That's exactly `in_proj_qkvz`'s first dimension. The $\hla{16 \times 128}$
blocks come from the key heads, and the $\hlb{48 \times 128}$ blocks from the
value heads.

So the name lists its own contents: q, k, v, and z, ==fused into one matrix== so
the layer does one matrix multiply (a *GEMM*) instead of four. The `z` block is
another output gate, the same trick as in the full-attention layers.

The other shapes decompose the same way, as the following table shows:

| Tensor | Shape | What it decomposes into |
|---|---|---|
| `in_proj_ba` | `[96, 5120]` | $2 \times 48$: one $\beta$ and one $a$ per value head |
| `conv1d` | `[10240, 1, 4]` | $2048 + 2048 + 6144$ channels, kernel width 4 |
| `A_log`, `dt_bias` | `[48]` | one scalar per value head |
| `out_proj` | `[5120, 6144]` | $48 \times 128$ value channels back to hidden |

Here $\beta$ is the delta rule's write strength and $a$ its forget gate. Chapter
6 explains both. For now, it's enough that each value head gets one of each.

These shapes support three conclusions, and none of them required opening a
weight file:

- **The layers aren't uniform.** Some have `self_attn`, and some have
  `linear_attn`. Cross-check against `config.json`, where `layer_types` lists
  all 64 entries and `full_attention_interval` is 4. Counting confirms 16 full
  and 48 linear.
- **`A_log` and `dt_bias` are state-space parameters.** A state-space model
  updates a running state at each step instead of attending over the past. A
  model carrying these tensors runs a recurrence in those layers, not attention.
  `A_log` stores the log of a decay rate, which keeps the rate in $(0, 1)$
  without needing a constraint during training. Chapter 6 uses both.
- **`conv1d` is depthwise.** A shape of `[channels, 1, kernel]` is what
  `nn.Conv1d(groups=channels)` stores: each channel has its own length-4 filter,
  and channels never mix. The 1 in the middle is the number of input channels
  per group, and it's the tell.

> [!TIP] Decode an unfamiliar name
> Given a name you haven't seen, ask three things: which layer, which submodule,
> and does the shape factor into head counts you already know? The answer is
> almost always yes. When it isn't, the config has a field you've been ignoring.

## The config that matters

The names tell you the structure, and the config gives you the exact numbers.
This section picks out the fields you need before chapter 2.

Qwen3.8-27B is multimodal, so `config.json` nests the language model under
`text_config` and puts a vision tower alongside it. This course uses the text
path only. `engine/config.py` handles both layouts with one line:

```python
text = raw.get("text_config", raw)
```

If `text_config` is present, the function reads the language model's fields from
it. Otherwise, the file is already flat. The same function builds `layer_types`
when the config omits it:

```python
layer_types = [
    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
    for i in range(text["num_hidden_layers"])
]
```

> [!WARNING] Full attention is the last layer of each group
> Note the `(i + 1)`. Full attention lands on layers 3, 7, 11, and so on: the
> last layer of each group of four, not the first. Get this off by one and every
> entry in the KV cache (the stored keys and values that chapter 9 builds) goes
> to the wrong layer, and the model still runs.

The following fields are worth knowing before chapter 2. The *residual stream*
is the running per-token vector that every layer reads from and adds to, and the
*MLP* is the feed-forward block in each layer:

| Field | Value | Why it matters |
|---|---|---|
| `hidden_size` | 5120 | Width of the residual stream. |
| `num_hidden_layers` | 64 | 16 full attention, 48 linear. |
| `num_attention_heads` | 24 | Query heads in full-attention layers. |
| `num_key_value_heads` | 4 | KV heads. Six queries share each one. |
| `head_dim` | 256 | Not `hidden_size / num_heads`. Read it, don't derive it. |
| `intermediate_size` | 17408 | MLP width. Most parameters live here. |
| `vocab_size` | 248320 | The embedding alone is 1.27B parameters. |
| `partial_rotary_factor` | 0.25 | Only 64 of 256 channels get rotated. |
| `attn_output_gate` | true | Why `q_proj` is double width. |
| `linear_num_value_heads` | 48 | Value heads in the recurrent layers. |

`head_dim` is the one that catches people. Here, $24 \times 256 = 6144$, not
5120, so the attention block projects into a wider space than the residual
stream and then projects back out.

If you derive the head width as $5120 / 24 = 213$ instead, ==every shape
downstream is wrong==. `ModelConfig.from_dict` reads the field and falls back to
the division only when the field is absent.

## What goes wrong

These are the loading mistakes that fail quietly, with no error to point you at
them.

**A truncated shard.** The header still parses, and the tensors near the front
still load. The ones at the end read past the end of the file, or, worse, read
zeros. The symptom is a model that produces fluent text for a few tokens and then
degenerates.

The repository includes `crc32.txt`. Checking it after a 54 GB download costs a
few minutes and rules this out before you spend an afternoon on it.

**The wrong name prefix.** Some checkpoints use
`model.language_model.layers.{i}.` and some use `model.layers.{i}.`.
`WeightIndex.layer_tensors` tries both:

```python
prefix = f"model.language_model.layers.{layer_idx}."
alt = f"model.layers.{layer_idx}."
```

A loader that knows only one prefix finds no tensors, loads nothing, and leaves
your layer at its random initialization. ==Every shape is right and every output
is noise.==

**A silent dtype conversion.** Calling `get(name, dtype=torch.float32)` doubles
the memory for that tensor and copies it. Do that in a loop over 64 layers, and
the host-memory advantage of mapping disappears.

**Tied embeddings.** When `tie_word_embeddings` is true, the checkpoint stores
one matrix and the output projection reuses it, so `lm_head.weight` is missing
from the index. For this model the flag is false and both exist, which is why
chapter 2 counts the embedding twice.

> [!RECAP]
> - A tensor's absolute offset is $8 + N + \text{begin}$, and
>   $\text{end} - \text{begin}$ must equal elements times bytes per element.
> - Headers and the index tell you every name, dtype, and shape without reading
>   weight data, and `get_slice` is how you ask.
> - Memory mapping plus layer-by-layer loading keeps peak host memory near one
>   layer. Any dtype or device change is a copy.
> - `q_proj` is double width because of the output gate, and `in_proj_qkvz`
>   fuses q, k, v, and z into 16384 rows.
> - Read `head_dim` from the config, and remember that full attention sits on
>   layers 3, 7, 11, and so on.

## Check your understanding

> [!QUESTION] A safetensors header says a tensor has shape `[17408, 5120]` and dtype `BF16`, with `data_offsets` `[0, 178257920]`. Is the file consistent?
> Yes: $17408 \times 5120 \times 2 = 178{,}257{,}920$. That's one MLP `up_proj`,
> and the identity between shape, dtype, and offset range is the cheapest
> integrity check available.

> [!QUESTION] Why can you list every tensor in an 18-shard checkpoint in milliseconds?
> Each shard's header is a small JSON blob at a known position, and the index is
> a single flat file naming which shard holds what. Neither requires reading the
> data buffer. The lab does all its work on the index alone.

> [!QUESTION] You see `linear_attn.in_proj_qkvz.weight` with shape `[16384, 5120]` and no config. What can you conclude?
> The layer fuses four projections into one matrix, its input is 5120 wide, and
> 16384 must factor into head counts times head dims. The shape doesn't tell you
> the split by itself: `[2048, 2048, 6144, 6144]` and `[4096, 4096, 4096, 4096]`
> both sum to 16384. The config's head counts resolve it, which is the general
> lesson: names and shapes narrow the possibilities, and the config picks one.

## Lab

> [!TRY]
> Open the model's index and answer questions about the architecture without
> loading a single weight. You pass when your functions find 64 layers, 16 of
> them full attention on every fourth layer from index 3, and tell the real
> `q_proj` from a normal-width one.

You're given the weight map, from tensor name to shard, and a table of tensor
shapes. You write four functions:

- **`count_layers`** extracts layer indices from names like
  `model.language_model.layers.7.mlp.gate_proj.weight` and counts the distinct
  ones.
- **`classify_layers`** returns a dict from layer index to `"full_attention"` or
  `"linear_attention"`, decided by whether the layer's tensors mention
  `self_attn` or `linear_attn`.
- **`largest_tensor`** returns the name and element count of the biggest tensor.
- **`detects_output_gate`** returns `True` when a `q_proj` has twice the output
  rows that `num_heads * head_dim` calls for.

The harness checks the following:

- You find 64 layers.
- 16 layers are full attention and 48 are linear.
- The full-attention layers land on every fourth layer, counting from index 3.
- The largest tensor is embedding-sized.
- Your gate detector reports `True` on the real shape and `False` on a
  normal-width `q_proj`.

The lab runs on CPU against a synthetic index shaped like the real checkpoint, so
it starts in seconds.

## Further reading

- [The safetensors format specification](https://github.com/huggingface/safetensors)
- [Qwen3.8-27B on Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)
- [mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html) — the system call underneath it all.
