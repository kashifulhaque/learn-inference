---
title: Tokens and embeddings
slug: 03-tokens-and-embeddings
part: "Part 2 — A forward pass"
summary: How byte-pair encoding builds a vocabulary, why no input is unrepresentable, and what the 248,320-row embedding table costs.
minutes: 70
gpu: false
objectives:
  - Explain what a byte-pair merge is and how a vocabulary is built from one.
  - Explain why byte-level BPE never fails on unseen input, and why the tokenizer is not a neural network.
  - Apply a chat template and identify the tokens that control generation.
  - Compute the embedding's parameter count and its share of the weight budget.
  - Describe the embedding lookup as a gather, and say what it would cost as a matrix multiply.
  - Explain weight tying and why this model does not use it.
lab: 03-tokenizer
---

# Tokens and embeddings

The model is a function on integers. It has no notion of text, characters, or
words. Something outside the network has to turn a string into a list of integers
before the first matrix multiply, and turn integers back into a string afterwards.
That something is the tokenizer, and the first layer of the network is the table
that maps each integer to a vector.

This is the shortest chapter in the forward pass and the one with the most
expensive table. The embedding and the output projection between them hold 2.54B
of the model's 26.9B parameters — 9.5% of the weight budget — and they exist
entirely because the vocabulary is large. This chapter works out why that trade
is worth making, and what the layer actually does on the GPU.

## Before you start

You need three things.

**A vector space of hidden states.** Chapter 1 established that `hidden_size` is
5120. Every token, at every layer, is a point in $\mathbb{R}^{5120}$. The
embedding table is what puts the first point there.

**UTF-8.** UTF-8 encodes a Unicode code point as one to four bytes. ASCII
characters are one byte. Latin-1 accents and Cyrillic are two. Most CJK and
Devanagari characters are three. Emoji are four, and an emoji with a skin-tone
modifier is two code points, so eight bytes. The important property for this
chapter is that UTF-8 is a *prefix code with continuation bytes*: given a partial
byte sequence you can tell whether it ends mid-character, which is what makes
incremental decoding possible at all.

**What a gather is.** A gather reads rows of a matrix at indices given by another
tensor. In PyTorch, `table[idx]` where `idx` is an integer tensor. It moves only
the rows you asked for. This is not a matrix multiply, and the distinction is
worth about nine orders of magnitude here.

The [notation chapter](/c/00a-notation-and-prerequisites) covers tensor-shape
notation and the floating-point formats if you want the background first. You do
not need it here; bfloat16 appears only as "2 bytes per number".

## What a tokenizer is, and is not

A tokenizer is not a neural network. It has no weights trained by gradient
descent, does no floating-point arithmetic, and runs on the CPU. It is two data
structures and a greedy loop:

- a **vocabulary**, mapping byte strings to integer ids;
- an ordered list of **merge rules**, each a pair of pieces that combine into one.

Both are fitted to a corpus by counting, once, before the model is trained. After
that they are frozen. You cannot fine-tune a tokenizer with the model, and you
cannot change one without retraining the embedding table, because token id 5000
means whatever the table's row 5000 learned it means.

The consequence for an inference engine: the tokenizer is a correctness
dependency, not a performance one. It costs microseconds. Getting it wrong costs
you the model.

## Byte-pair encoding, from the bottom

Byte-pair encoding was a compression algorithm before it was a tokenizer. The
idea is to start with the smallest possible alphabet and repeatedly promote the
most common adjacent pair into a new symbol.

### The base alphabet

Qwen uses *byte-level* BPE, so the starting alphabet is the 256 possible byte
values. Not characters, not Unicode code points — bytes. Text is encoded to UTF-8
first, and everything after that is a sequence over an alphabet of size 256.

### What a merge is

A merge is an ordered pair of existing vocabulary entries and the single new entry
they produce. Write it as $(a, b) \to ab$. Nothing more.

Training counts, over the whole corpus, how often each adjacent pair of current
pieces occurs. It takes the most frequent pair, adds the concatenation to the
vocabulary as a new id, rewrites every occurrence of that pair in the corpus as
the new piece, and repeats. Each round adds exactly one vocabulary entry, so
after $M$ rounds the vocabulary holds

$$
|V| = 256 + M + S
$$

entries, where $S$ is the number of special tokens added by hand afterwards. With
$|V| = 248{,}320$ and a few dozen special tokens, this model's tokenizer learned
roughly 248,000 merges.

### A worked example

Take a corpus that is mostly the words `low`, `lower`, and `lowest`. Start with
every word split into bytes:

```text
l o w        l o w e r        l o w e s t
```

The pair `(l, o)` is the most frequent, so merge it:

```text
lo w         lo w e r         lo w e s t
```

Now `(lo, w)` is most frequent:

```text
low          low e r          low e s t
```

Then `(e, s)`, then `(es, t)`:

```text
low          low e r          low est
```

Four merges, four new vocabulary entries: `lo`, `low`, `es`, `est`. The word
`lowest` now costs two tokens instead of six bytes. Nothing here is learned in the
machine-learning sense. It is frequency counting.

### Encoding applies the merges in order

Encoding a new string reverses the process. Split it into bytes, then apply the
merge rules in the order they were learned, repeatedly, until no rule applies.
Order matters: `lo` must exist before `low` can be formed. A tokenizer that
applied merges by rank of the resulting token, or in any other order, would
produce different ids for the same string, and the model would see nonsense.

This is why two tokenizer implementations that agree on the vocabulary can still
disagree on output. If your generated text looks subtly wrong, check that the
prompt round-trips: encode it, decode it, compare with the original.

### Pre-tokenization

Before BPE runs, the text is split by a regular expression into chunks —
typically words with a leading space, runs of digits, runs of punctuation. Merges
never cross a chunk boundary. This is what stops the tokenizer from learning a
single token for `the quick brown fox`, and it is also why a leading space is
usually part of the token: `" cat"` and `"cat"` are different ids.

## Why nothing is unrepresentable

The base alphabet is all 256 byte values, and every one of them has a vocabulary
entry. So for any input at all — valid UTF-8, invalid UTF-8, a null byte, the
middle of a JPEG pasted into a chat box — the encoder can always fall back to
emitting raw bytes. There is no unknown token, because there is nothing unknown.

Compare that with a character-level or word-level vocabulary. Both have to decide
what to do with a symbol they never saw during training, which means an `<unk>`
token, which means information is destroyed before the model sees it. Byte
fallback removes that failure mode entirely. The cost is spread across the
representation instead: a character the tokenizer has never seen in context
becomes three or four separate byte tokens, which the model can still read, just
less efficiently.

That inefficiency is not evenly distributed. English averages about 4 characters
per token. Code averages fewer, because indentation and punctuation split.
Languages written in non-Latin scripts often cost 2 to 3 times more tokens for the
same content, since each character is 3 bytes and fewer merges were learned over
them. For a user that is a real cost difference — more tokens per message, more
latency, more money — not a curiosity.

## Special tokens and the chat template

Special tokens are added to the vocabulary after training and are excluded from
the merge rules. They are matched literally, before BPE runs, so they can never be
produced by merging ordinary text. That is what makes them usable as control
signals: the model can tell a real `<|im_end|>` from a user who typed the same
characters.

Instruction-tuned models expect structure. The chat template, shipped as
`chat_template.jinja`, converts a list of messages into a single string:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a KV cache?<|im_end|>
<|im_start|>assistant
```

The last line has no closing tag. The prompt ends exactly where the model is
supposed to continue, and the first thing it generates is the assistant's reply.

Two ids from `generation_config.json` matter to the engine:

- `eos_token_id` is `[248046, 248044]`. It is a list, so the engine must stop on
  either. Checking only the first is a common bug, and it shows up as generations
  that run to `max_tokens` with trailing garbage after a perfectly good answer.
- `pad_token_id` is 248044. Padding only matters when you batch sequences of
  different lengths into one rectangular tensor, and the paged cache in chapter 15
  removes the need for it entirely.

Getting the template wrong does not raise an error. It degrades quality in a way
that looks like the model being bad. When output seems worse than it should,
print the exact string you tokenized before you look anywhere else.

## What the vocabulary costs

A vocabulary entry is not free. Each one owns a row of the embedding table and a
row of the output projection, both 5120 wide.

The embedding is:

$$
248{,}320 \times 5120 = 1{,}271{,}398{,}400 \approx 1.27\text{B parameters}
$$

In bfloat16, at 2 bytes each:

$$
1{,}271{,}398{,}400 \times 2 = 2{,}542{,}796{,}800 \approx 2.54\ \text{GB}
$$

`tie_word_embeddings` is false, so the output projection is a second matrix of
exactly the same size. Together:

$$
2 \times 2.54 = 5.09\ \text{GB}
$$

against a total weight footprint of 53.8 GB. That is

$$
\frac{5.09}{53.8} = 9.5\%
$$

of the model spent on the vocabulary, before a single layer of it has done any
reasoning. All of these are arithmetic from the config in chapter 1, not
measurements.

Is it worth it? The benefit is that a bigger vocabulary means fewer tokens per
document, and both phases of inference scale with token count. Prefill does one
forward pass per prompt token. Decode does one full read of the weights per
generated token. Halving the vocabulary would return 2.54 GB of HBM and lengthen
every sequence, which costs prefill FLOPs and decode steps in proportion. The
designers took the memory.

There is a subtlety worth carrying into chapter 18. The 2.54 GB embedding table
sits in HBM permanently but is barely read: a decode step gathers one row, 10 KiB.
The output projection, in contrast, is read in full on every step. Two matrices of
identical size with completely different access patterns is exactly the situation
where mixed-precision quantization pays.

## The embedding layer

### The lookup, two ways

Textbooks introduce the embedding as a matrix multiply against a one-hot vector.
Let $E \in \mathbb{R}^{V \times d}$ be the table, with $V = 248{,}320$ and
$d = 5120$, and let $\mathbf{1}_{v} \in \mathbb{R}^{V}$ be the one-hot vector with
a 1 in position $v$. Then the embedding of token $v$ is

$$
e = E^{\top} \mathbf{1}_{v}
$$

which is correct, and a catastrophic way to compute it. That product does
$2Vd$ floating-point operations and reads the entire table:

$$
2 \times 248{,}320 \times 5120 = 2.54\ \text{GFLOP}
$$

$$
248{,}320 \times 5120 \times 2\ \text{bytes} = 2.54\ \text{GB read}
$$

The same result as a gather reads one row:

$$
5120 \times 2\ \text{bytes} = 10{,}240\ \text{bytes} = 10\ \text{KiB}
$$

and does zero arithmetic. The ratio is $2.54 \times 10^9 / 10{,}240 = 248{,}320$,
which is $V$ exactly — the one-hot product wastes the whole table to keep one row.
Every framework implements `nn.Embedding` as a gather. The one-hot picture is
useful for reasoning about gradients during training and for nothing else.

In the engine it is one line:

```python
x = self.embed_tokens(input_ids)
```

with shapes:

| Tensor | Shape | Dtype |
|---|---|---|
| `input_ids` | `(batch, seq)` | int64 |
| `embed_tokens.weight` | `(248320, 5120)` | bfloat16 |
| `x` | `(batch, seq, 5120)` | bfloat16 |

For a 2000-token prompt the gather moves $2000 \times 10\ \text{KiB} = 20$ MB,
scattered across a 2.54 GB table. The scatter is the reason it is not quite free:
the reads have no locality, so they miss L2 and each one pays full HBM latency.
It is still the cheapest layer in the model by a wide margin.

### The residual stream at layer 0

The output of the embedding is the initial value of the residual stream, the
running $(batch, seq, 5120)$ tensor that every layer reads from and adds to.
Chapter 4 develops it properly. Two things about its starting state matter here.

It carries no position information. Two occurrences of the same token anywhere in
the prompt produce identical 5120-vectors. Everything that distinguishes them is
added later, by the rotation in chapter 5 and by attention itself.

Its absolute scale barely matters. The first thing layer 0 does is

```python
hidden = self.input_layernorm(x)
```

and RMSNorm divides out the magnitude of the row. Some architectures multiply the
embedding by $\sqrt{d}$ on the way out to compensate for a small initialization;
a pre-norm model does not need to, and this one does not do it. If you are porting
weights and see a stray $\sqrt{5120} \approx 71.6$ factor in someone else's code,
that is what it is for.

## The output projection

At the other end of the stack, the same shape does a completely different job:

```python
x = self.norm(x)
if last_token_only:
    x = x[:, -1:, :]
return self.lm_head(x)
```

`lm_head` is `nn.Linear(5120, 248320, bias=False)`, so its weight is
$(248320, 5120)$ — the same shape as the embedding, transposed in the sense that
it maps a hidden state to a score per vocabulary entry rather than a vocabulary
entry to a hidden state. Row $v$ of `lm_head.weight` is the direction in the
residual stream that means "token $v$", and the logit for $v$ is the inner product
of the final hidden state with that direction.

This one is a real GEMM, and during decode it runs for a single position:

$$
2 \times 5120 \times 248{,}320 = 2.54\ \text{GFLOP}
$$

reading 2.54 GB of weights. Its arithmetic intensity is

$$
\frac{2.54 \times 10^9\ \text{FLOP}}{2.54 \times 10^9\ \text{bytes}} = 1\ \text{FLOP/byte}
$$

against the A100's ridge point of about 161 FLOPs per byte. It is as memory bound
as an operation gets. Of the 53.8 GB of weights, the decode step reads about
51.3 GB — everything except the embedding table — and the output projection is
2.54 GB of that, just under 5%.

### Slice before you project

The `last_token_only` slice is what keeps prefill honest. Without it, a
2000-token prompt computes logits for all 2000 positions:

$$
2000 \times 248{,}320 \times 2\ \text{bytes} = 993\ \text{MB}
$$

of output tensor, plus

$$
2 \times 2000 \times 5120 \times 248{,}320 = 5.08\ \text{TFLOP}
$$

of arithmetic, and you throw away all but the last row. Slicing first turns that
into one row: 2.54 GFLOP and a 497 KB output. Training needs every position
because every position has a target. Inference needs one.

### Weight tying, and why not here

Weight tying means using the same matrix for both jobs: $E$ for the lookup and
$E^{\top}$ for the projection. It halves the vocabulary's parameter cost and has a
tidy justification — the same vector should mean the same token going in and
coming out.

`tie_word_embeddings` is false here, so the matrices are separate. Two reasons
this is the common choice at this scale.

The saving matters less as models grow. Tying originated when vocabularies were
30,000 entries and models were a few hundred million parameters, where the
embedding was a large fraction of the total. Here, untying costs 1.27B parameters
out of 26.9B: 4.7%. That is a real cost, but it buys the output matrix the freedom
to encode "what predicts this token next" rather than "what this token means",
which are related but not the same question.

The two matrices also want different treatment at serving time, as noted above:
one is gathered a row at a time, the other is streamed in full every step.

The engine has to check the flag, not assume it. `ModelConfig.param_counts` does:

```python
embed = self.vocab_size * h
lm_head = 0 if self.tie_word_embeddings else self.vocab_size * h
```

Assume tying on an untied checkpoint and `lm_head.weight` never gets loaded from
disk. Depending on how you initialized the module, you get either a crash or —
worse — a model that runs at full speed and emits fluent, confident nonsense.

## Detokenization is stateful

Streaming output back to a user is harder than it looks, because one token is not
one printable unit. A token is a run of bytes, and a multi-byte UTF-8 character
can straddle two tokens. Decode each token independently and you get replacement
characters at the seams.

The fix is to keep per-sequence state: append the new id, decode the accumulated
ids, and emit only the suffix that is new *and* complete. The lab's solution is
the whole idea in three lines:

```python
def push(self, token_id: int) -> str:
    self.ids.append(token_id)
    text = self.decode(self.ids).decode("utf-8", errors="ignore")
    new = text[self.emitted:]
    self.emitted = len(text)
    return new
```

`errors="ignore"` drops a trailing incomplete sequence rather than replacing it,
so the partial character does not appear yet. The token that completes it
makes the whole character appear at once. `flush` at the end of a generation uses
`errors="replace"` instead, because at that point an incomplete sequence is not
going to be completed and hiding it would lose output.

Some pushes return an empty string. That is correct behaviour, not a bug, and the
lab checks for it: a three-byte Devanagari character split across two-byte tokens
has to wait.

Doing this wrong is visible to users as flickering mojibake in the middle of
otherwise fine text.

## What goes wrong

**The prompt is not what you think it is.** The single highest-value debugging
habit in this course: print `repr()` of the exact string you tokenized, and the id
list. Missing newline after `<|im_start|>assistant`, a system message the template
silently dropped, a double-applied template — all of these produce degraded output
and no error.

**Stopping on only the first EOS id.** `eos_token_id` is a list. Build a set and
test membership.

**Double-encoding special tokens.** If you build the prompt string yourself and
then encode it with special-token parsing disabled, `<|im_start|>` becomes a dozen
ordinary byte tokens. The model has never seen that pattern and behaves as if the
conversation has no structure.

**Computing logits for every prefill position.** Silent, and it costs about 1 GB
and 5 TFLOP on a 2000-token prompt. It shows up as prefill being mysteriously
slower than your roofline estimate, which is exactly what chapter 10 teaches you
to notice.

**Decoding each streamed token independently.** Mojibake at multi-byte character
boundaries, and worse for scripts where most characters are multi-byte.

## Check your understanding

**The vocabulary has 248,320 entries and the base alphabet has 256. Where did the
other 248,064 come from, and what would happen if you deleted the last 1,000
merges?**

They are learned merges, plus the special tokens. Deleting the last 1,000 merges
would still let you encode every possible input — byte fallback guarantees that —
but strings that used those merges would take more tokens, and every id above the
deletion point would shift, so the embedding table would no longer match. The
vocabulary is a contract with the weights.

**The embedding and the output projection have the same shape. Why is one of them
nearly free and the other one of the most expensive operations in a decode step?**

Access pattern. The embedding is indexed: one row, 10 KiB. The output projection
is a full GEMM against all 248,320 rows, 2.54 GB, and at batch 1 it does one FLOP
per byte read, which is 1/161 of what the A100 needs to stay compute bound.

**Your engine emits one token per step and the user sees occasional black diamond
question marks. Which layer is at fault?**

None of them. The model is fine; the detokenizer is decoding each token on its own
and splitting multi-byte UTF-8 characters. Buffer the ids and emit only complete
characters.

## Lab

Two parts, both on CPU.

First, `build_prompt`: render a list of `{"role", "content"}` messages into the
exact string the model expects, with each turn wrapped in `<|im_start|>` and
`<|im_end|>`, and the assistant turn opened but left unclosed. The harness
compares against the expected string character for character and checks that no
`<|im_end|>` follows the final `<|im_start|>assistant`.

Second, `IncrementalDetokenizer`: given a `decode` callable that maps a list of
ids to bytes, implement `push` and `flush` so that the concatenation of everything
you emit equals the original text, with no replacement characters anywhere. The
harness runs ASCII, Devanagari, emoji with skin-tone modifiers, and a mixed
string, using a tokenizer that chops text into fixed-size byte runs so characters
split across tokens exactly as they do with a real byte-level BPE vocabulary. It
also checks that some pushes emit nothing, which is how it knows you are buffering
rather than guessing.

## Further reading

- [Neural machine translation of rare words with subword units](https://arxiv.org/abs/1508.07909) — the original BPE paper.
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — introduced byte-level BPE, in section 2.2.
- [The tokenizers library](https://github.com/huggingface/tokenizers)
- [Using the output embedding to improve language models](https://arxiv.org/abs/1608.05859) — the weight-tying paper.
