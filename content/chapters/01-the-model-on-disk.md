---
title: The model on disk
slug: 01-the-model-on-disk
part: "Part 1 — Ground truth"
summary: The safetensors byte layout, shard indexes, memory mapping, and what the tensor names tell you about the architecture.
minutes: 60
gpu: false
objectives:
  - Parse a safetensors header by hand and compute where any tensor's bytes live.
  - Read a safetensors index and locate any tensor without loading the model.
  - Infer a model's architecture from its tensor names and shapes alone.
  - Explain why memory mapping makes lazy, layer-by-layer loading possible.
lab: 01-inspect-weights
---

# The model on disk

Qwen3.8-27B ships as 18 safetensors shards, a JSON index, a tokenizer, and a
config. Together they are about 54 GB. Before you write a single layer, read what
the files actually contain: the tensor names are the most honest description of
the architecture you can get, and they often disagree with the model card.

This chapter is the only one where you touch bytes on disk. Everything after it
assumes you can ask "what shape is layer 37's key projection" and get an answer
in milliseconds, without loading 54 GB.

## Before you start

Three ideas do all the work here.

**A file is an array of bytes, and a tensor is a contiguous run of them.** A
tensor has a dtype (how many bytes per element, and how to interpret them) and a
shape. Multiply the shape out, multiply by the element size, and you have the
tensor's length in bytes. Nothing else is stored.

**Row-major means the last index moves fastest.** A tensor of shape
`(248320, 5120)` stores row 0's 5120 values first, then row 1's, and so on. The
element at `[i, j]` sits at offset $(i \times 5120 + j) \times
\text{bytes per element}$ from the start. This layout is also called C order, and
it is the only one safetensors stores.

**Memory mapping makes a file look like memory.** `mmap` asks the kernel to
reserve a range of addresses backed by a file rather than by RAM. Nothing is
read at that moment. When your program touches an address in the range, the
kernel faults in the 4 KB page containing it, from disk or from the page cache.
Touch a hundred bytes of a 3 GB file and you have read a hundred bytes.

If any of that is unfamiliar, or if the GB/GiB distinction is, the
[notation chapter](/c/00a-notation-and-prerequisites) covers both.

## The safetensors format

A safetensors file has three parts, in this order:

```text
[ 8 bytes ][ N bytes of JSON header ][ raw tensor bytes ]
```

The first 8 bytes are an unsigned 64-bit little-endian integer: $N$, the length
of the header. The next $N$ bytes are UTF-8 JSON. Everything after that is a flat
buffer of tensor data with no separators, no padding between tensors, and no
metadata of any kind.

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

`data_offsets` is a half-open byte range `[begin, end)` measured from the *end of
the header*, not from the start of the file. So the absolute position of the
first byte of that tensor is $8 + N + \text{begin}$.

Check the arithmetic on that entry. The shape is `1024 x 5120` and `BF16` is two
bytes per element:

$$
1024 \times 5120 \times 2 = 10{,}485{,}760\ \text{bytes}
$$

which is exactly $\text{end} - \text{begin}$. That identity always holds, and
verifying it is how you catch a truncated file. The `__metadata__` key is the one
reserved name; it holds a flat string-to-string map and no tensor data.

Four properties follow from this layout, and they are the reasons the format
exists.

**Listing tensors is one small read.** Read 8 bytes, read $N$ more, parse JSON.
You now know every tensor's name, dtype, and shape without touching a byte of
weight data. The lab depends on this.

**There are no strides.** Every tensor is stored C-contiguous. A framework that
wants to save a transposed view must materialize it first. This makes reading
trivial — offset arithmetic and nothing else — and it means the shape in the
header is the shape in memory.

**Reading a tensor is a memory map, not a copy.** The loader maps the file and
hands you a tensor whose storage points into the mapping. Pages arrive when you
touch them. Moving that tensor to the GPU is what actually triggers the read, and
it happens one tensor at a time.

**Nothing executes.** The header is JSON with a fixed schema, and the rest is
numbers. Compare `torch.save`, which uses pickle: reading anything requires
deserializing arbitrary Python objects, so you must trust the file, and you must
load the whole thing before you can inspect any of it. Safetensors exists mostly
because "download a stranger's weights" should not mean "run a stranger's code".

The dtype strings are the format's own, not PyTorch's: `F64`, `F32`, `F16`,
`BF16`, `I64`, `I32`, `I16`, `I8`, `U8`, `BOOL`. This model's weights are all
`BF16`. When a loader converts, it does so after mapping, which means a
conversion costs a full copy — a reason to keep the engine in bfloat16 end to
end.

## Why memory mapping matters here

The naive way to load a 54 GB model onto an 80 GB GPU is to read the whole
checkpoint into host RAM, then copy it across. That needs 54 GB of RAM you may
not have, and it serializes: nothing reaches the GPU until everything has been
read.

With a mapped file you can go layer by layer:

```python
for layer_idx in range(config.num_hidden_layers):
    for suffix, name in index.layer_tensors(layer_idx):
        tensor = index.get(name, device="cuda")   # page in, then H2D copy
```

Peak host memory is now one layer — about 0.8 GB, not 54 GB — because the
kernel reclaims pages behind you once nothing references them. Pages still in the
page cache from a previous run come back for free, which is why the second load
of a model is much faster than the first even though nothing was explicitly
cached.

Two gotchas. Mapped pages are shared, so a mapped tensor is read-only in
practice; write to it and you either fault or dirty the page cache. And a mapped
tensor on CPU still reads from disk on first touch, so timing a load and getting
"instant" means you measured the mapping, not the read.

## Shards and the index

A single 54 GB file is awkward to download and impossible to resume, so the model
is split into 18 shards of roughly 3 GB each. `model.safetensors.index.json` maps
each tensor name to the shard that holds it:

```json
{
  "metadata": {"total_size": 53974302720},
  "weight_map": {
    "model.language_model.layers.0.mlp.gate_proj.weight": "model-00001-of-00018.safetensors",
    "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001-of-00018.safetensors"
  }
}
```

`total_size` is the sum of every tensor's bytes, not the sum of the file sizes —
the headers are not counted. Chapter 2 counts 53.79 GB for the language model
alone; the index reports a little more, because the checkpoint also carries
tensors outside the text path.

To load a tensor: look up its shard in `weight_map`, open that shard, read the
range. `WeightIndex` in `engine/weights.py` does exactly this, and caches the
open handles so that loading 64 layers does not reopen 18 files 64 times:

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

The `else` branch matters for small models, which ship as one file with no index.
Building the map from `f.keys()` gives the rest of the class one code path
instead of two.

```python
    def get(self, name, device="cpu", dtype=None):
        shard = self.map[name]
        tensor = self._handle(shard).get_tensor(name)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(device)
```

`get_tensor` returns a view into the mapping. The two `.to()` calls are where any
real work happens: a dtype change copies, and a device change copies across PCIe.
Calling `get` with `device="cuda"` and no dtype is the cheap path — one copy,
straight from the page cache to the GPU.

For inspection without loading, `describe` uses a slice handle instead:

```python
with safe_open(index.dir / index.map[name], framework="pt") as f:
    slice_ = f.get_slice(name)
    rows.append({"name": name, "shape": list(slice_.get_shape()),
                 "dtype": slice_.get_dtype()})
```

`get_slice` reads the header entry and returns an object that knows the shape and
dtype and has not touched the data buffer. This is the operation the lab is built
around.

## Reading the architecture from the names

Group the tensor names by layer index and the model describes itself. In this
checkpoint, layer 0 and layer 3 hold different things:

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
everywhere.

**Linear weights are stored output-first.** PyTorch's `nn.Linear` holds a weight
of shape `(out_features, in_features)` and computes $y = x W^\top$. So the second
dimension is always the input width, and here it is 5120 — the hidden size,
confirmed four times over.

**The key projection tells you the KV geometry.** 1024 output channels, and the
config says `head_dim` is 256, so $1024 / 256 = 4$ key heads. Same for values.

**The output projection tells you the query geometry.** `o_proj` maps 6144 back
to 5120, and $6144 / 256 = 24$ query heads. Twenty-four queries, four keys: six
queries share each key head. That ratio is the GQA group size, and chapter 7 is
about what it buys.

**The query projection is a surprise.** With 24 heads of width 256 you expect
6144 output channels. The file says 12288 — exactly double. That is the *output
gate*: the projection emits queries and a gate side by side, and the layer splits
them. `config.json` confirms it with `attn_output_gate: true`. Trust the shape
arithmetic instead of the file and you write a layer that loads without error and
produces nonsense.

Now the linear-attention layer, which is where the names really earn their keep.
You know from the config that `linear_num_key_heads` is 16,
`linear_num_value_heads` is 48, and both head dims are 128. So:

$$
\underbrace{16 \times 128}_{q} + \underbrace{16 \times 128}_{k}
+ \underbrace{48 \times 128}_{v} + \underbrace{48 \times 128}_{z}
= 2048 + 2048 + 6144 + 6144 = 16384
$$

which is exactly `in_proj_qkvz`'s first dimension. The name lists its own
contents: q, k, v, and z, fused into one matrix so the layer does one GEMM
instead of four. The `z` block is another output gate, the same trick as the
full-attention layers.

The other shapes fall out the same way:

| Tensor | Shape | What it decomposes into |
|---|---|---|
| `in_proj_ba` | `[96, 5120]` | $2 \times 48$: one $\beta$ and one $a$ per value head |
| `conv1d` | `[10240, 1, 4]` | $2048 + 2048 + 6144$ channels, kernel width 4 |
| `A_log`, `dt_bias` | `[48]` | one scalar per value head |
| `out_proj` | `[5120, 6144]` | $48 \times 128$ value channels back to hidden |

Three conclusions, none of which required opening a weight file.

**The layers are not uniform.** Some have `self_attn`, some have `linear_attn`.
Cross-check against `config.json`, where `layer_types` lists all 64 entries and
`full_attention_interval` is 4. Counting confirms 16 full and 48 linear.

**`A_log` and `dt_bias` are state-space parameters.** A model carrying those is
running a recurrence in those layers, not attention. `A_log` stores the log of a
decay rate, which is how you keep a value in $(0, 1)$ parameterized without a
constraint. Chapter 6 uses both.

**`conv1d` is depthwise.** A shape of `[channels, 1, kernel]` is what
`nn.Conv1d(groups=channels)` stores: each channel has its own length-4 filter and
they never mix. That 1 in the middle is the input-channels-per-group, and it is
the tell.

The reverse direction is worth practising too. Given a name you have not seen,
ask: which layer, which submodule, and does the shape factor into head counts you
already know? The answer is almost always yes, and when it is not, the config has
a field you have been ignoring.

## The config that matters

Qwen3.8-27B is multimodal, so `config.json` nests the language model under
`text_config` and puts a vision tower alongside it. This course uses the text
path only. `engine/config.py` handles both shapes with one line:

```python
text = raw.get("text_config", raw)
```

If `text_config` is present, read the language model's fields from it; otherwise
the file is already flat. The same function synthesizes `layer_types` when the
config omits it:

```python
layer_types = [
    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
    for i in range(text["num_hidden_layers"])
]
```

Note the `(i + 1)`. Full attention lands on layers 3, 7, 11, and so on — the last
layer of each group of four, not the first. Getting this off by one puts every
KV cache entry in the wrong layer, and the model still runs.

The fields worth knowing before chapter 2:

| Field | Value | Why it matters |
|---|---|---|
| `hidden_size` | 5120 | Width of the residual stream. |
| `num_hidden_layers` | 64 | 16 full attention, 48 linear. |
| `num_attention_heads` | 24 | Query heads in full-attention layers. |
| `num_key_value_heads` | 4 | KV heads. Six queries share each one. |
| `head_dim` | 256 | Not `hidden_size / num_heads`. Read it, do not derive it. |
| `intermediate_size` | 17408 | MLP width. Most parameters live here. |
| `vocab_size` | 248320 | The embedding alone is 1.27B parameters. |
| `partial_rotary_factor` | 0.25 | Only 64 of 256 channels get rotated. |
| `attn_output_gate` | true | Why `q_proj` is double width. |
| `linear_num_value_heads` | 48 | Value heads in the recurrent layers. |

`head_dim` is the one that catches people. $24 \times 256 = 6144$, not 5120, so
the attention block projects into a wider space than the residual stream and
projects back out. Deriving it as $5120 / 24 = 213$ makes every shape downstream
wrong. `ModelConfig.from_dict` reads the field and only falls back to the
division when it is absent.

## What goes wrong

**A truncated shard.** The header still parses, and the tensors near the front
still load. The ones at the end read past the end of the file, or worse, read
zeros. The symptom is a model that produces fluent text for a few tokens and then
degenerates. The repository includes `crc32.txt`; checking it after a 54 GB
download costs a few minutes and rules this out before you spend an afternoon on
it.

**The wrong name prefix.** Some checkpoints use
`model.language_model.layers.{i}.` and some use `model.layers.{i}.`.
`WeightIndex.layer_tensors` tries both:

```python
prefix = f"model.language_model.layers.{layer_idx}."
alt = f"model.layers.{layer_idx}."
```

A loader that only knows one prefix finds no tensors, loads nothing, and leaves
your layer at its random initialization. Every shape is right and every output is
noise.

**A silent dtype conversion.** Calling `get(name, dtype=torch.float32)` doubles
the memory for that tensor and copies. Do it in a loop over 64 layers and the
host-memory advantage of mapping disappears.

**Tied embeddings.** When `tie_word_embeddings` is true, the checkpoint stores
one matrix and the output projection reuses it, so `lm_head.weight` is missing
from the index. For this model the flag is false and both exist, which is why
chapter 2 counts the embedding twice.

## Check your understanding

**A safetensors header says a tensor has shape `[17408, 5120]` and dtype `BF16`,
with `data_offsets` `[0, 178257920]`. Is the file consistent?** Yes:
$17408 \times 5120 \times 2 = 178{,}257{,}920$. That is one MLP `up_proj`, and
the identity between shape, dtype, and offset range is the cheapest integrity
check available.

**Why can you list every tensor in an 18-shard checkpoint in milliseconds?**
Because each shard's header is a small JSON blob at a known position, and the
index is a single flat file naming which shard holds what. Neither requires
reading the data buffer. The lab does all its work on the index alone.

**You see `linear_attn.in_proj_qkvz.weight` with shape `[16384, 5120]` and no
config. What can you conclude?** That the layer fuses four projections into one
matrix, that its input is 5120 wide, and that 16384 must factor into head counts
times head dims. It does not tell you the split by itself — `[2048, 2048, 6144,
6144]` and `[4096, 4096, 4096, 4096]` both sum to 16384. The config's head counts
resolve it, which is the general lesson: names and shapes narrow the possibilities
and the config picks one.

## Lab

Open the model's index and answer questions about the architecture without
loading a single weight. You are given the weight map, name to shard, and a table
of tensor shapes.

You write four functions. `count_layers` extracts layer indices from names like
`model.language_model.layers.7.mlp.gate_proj.weight` and counts the distinct
ones. `classify_layers` returns a dict from layer index to `"full_attention"` or
`"linear_attention"`, decided by whether the layer's tensors mention `self_attn`
or `linear_attn`. `largest_tensor` returns the name and element count of the
biggest one. `detects_output_gate` returns `True` when a `q_proj` has twice the
output rows that `num_heads * head_dim` calls for.

The harness checks that you find 64 layers, that 16 are full attention and 48 are
linear, that the full-attention layers land on every fourth layer counting from
index 3, that the largest tensor is embedding-sized, and that your gate detector
reports `True` on the real shape and `False` on a normal-width `q_proj`.

The lab runs on CPU against a synthetic index shaped like the real checkpoint, so
it starts in seconds.

## Further reading

- [The safetensors format specification](https://github.com/huggingface/safetensors)
- [Qwen3.8-27B on Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)
- [mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html) — the system call underneath it all.
