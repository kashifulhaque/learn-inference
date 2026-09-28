---
title: Tokens and embeddings
slug: 03-tokens-and-embeddings
part: "Part 2 — A forward pass"
summary: How byte-pair encoding builds a vocabulary, why no input is unrepresentable, and what the 248,320-row embedding table costs.
minutes: 70
gpu: false
objectives:
  - Explain what a byte-pair merge is and how a vocabulary is built from merges.
  - Explain why byte-level BPE never fails on unseen input, and why the tokenizer isn't a neural network.
  - Apply a chat template and identify the tokens that control generation.
  - Compute the embedding's parameter count and its share of the weight budget.
  - Describe the embedding lookup as a gather, and say what it would cost as a matrix multiply.
  - Explain weight tying and why this model doesn't use it.
lab: 03-tokenizer
---

# Tokens and embeddings

> [!TLDR]
> - The tokenizer, which turns text into integer ids and back, isn't a neural
>   network. It's a vocabulary, an ordered list of merge rules, and a short loop
>   that runs on the CPU in microseconds.
> - It starts from all 256 possible byte values, so it can encode any input at
>   all. There's no "unknown" token.
> - The vocabulary is huge, 248,320 entries, and the two tables sized by it hold
>   about a tenth of the model's weights.
> - The input table is cheap to use: each token reads one small row. The output
>   table has the same shape, but every generated token reads all of it.
> - Chat templates, stop tokens, and streaming text back to the user all fail
>   without an error. When output looks wrong, print exactly what you tokenized
>   before you debug anything else.

The model is a function on integers. It has no notion of text, characters, or
words. Something outside the network turns a string into a list of integers
before the first matrix multiply, and turns integers back into a string
afterwards. That something is the *tokenizer*.

On the network side, the first layer is the *embedding*: a table that maps each
integer to a vector. This chapter follows a piece of text through both.

This is the shortest chapter in the forward pass, and the one with the most
expensive table. The embedding and the output projection hold 2.54B of the
model's 26.9B parameters between them, 9.5% of the weight budget, and they exist
entirely because the vocabulary is large. You see why that trade is worth
making, and what the layer does on the GPU.

## Before you start

You need four things.

**A vector space of hidden states.** Chapter 1 established that `hidden_size` is
5120. Every token, at every layer, is a point in $\mathbb{R}^{5120}$, and the
embedding table is what puts the first point there.

**UTF-8.** UTF-8 encodes each Unicode character (code point) as one to four
bytes, as the following table shows:

| Characters | Bytes per character | Example |
|---|---|---|
| ASCII | 1 | `a` |
| Latin-1 accents, Cyrillic | 2 | `é`, `ж` |
| Most CJK and Devanagari | 3 | `中`, `क` |
| Emoji | 4 | `😀` |
| Emoji with a skin-tone modifier | 8, as two code points | `👍🏽` |

UTF-8 is a *prefix code with continuation bytes*: the first byte of a character
says how many bytes follow. So given a partial byte sequence, you can tell
whether it ends mid-character. That property is what makes incremental decoding
possible at all.

**What a gather is.** A gather reads rows of a matrix at indices given by
another tensor, as `table[idx]` does in PyTorch when `idx` is an integer tensor.
It moves only the rows you asked for. It isn't a matrix multiply, and here the
distinction is worth a factor of 248,320 in bytes read.

**Prefill and decode.** An engine runs in two phases. *Prefill* runs the whole
prompt through the model in one pass. *Decode* then generates one token per
step, and each step reads the weights from HBM, the GPU's main memory.

The [notation chapter](/c/00a-notation-and-prerequisites) covers tensor-shape
notation and the floating-point formats. You don't need it here: bfloat16
appears only as "2 bytes per number".

## What a tokenizer is, and isn't

This section answers what kind of component the tokenizer is, because that
decides how you treat it in the engine. A tokenizer is ==not a neural network==.
It has no weights trained by gradient descent, does no floating-point
arithmetic, and runs on the CPU.

It's two data structures and a greedy loop:

- A **vocabulary**, mapping byte strings to integer ids.
- An ordered list of **merge rules**, each a pair of pieces that combine into
  one.

Both are fitted to a corpus by counting, once, before the model is trained, and
then frozen. You can't fine-tune a tokenizer along with the model.

You also can't change a tokenizer without retraining the embedding table. Token
id 5000 means whatever row 5000 of the table learned it means, so the ids and
the table only make sense together.

> [!KEY] The tokenizer is a correctness dependency, not a performance one
> It costs microseconds. Getting it wrong costs you the model.

## How byte-pair encoding builds a vocabulary

Next, you see where the 248,320 entries come from. *Byte-pair encoding* (BPE)
builds a vocabulary by starting with the smallest possible alphabet, then
repeatedly promoting the most common adjacent pair into a new symbol. It was a
compression algorithm before it was a tokenizer.

### The base alphabet

Qwen uses *byte-level* BPE, so the starting alphabet is the 256 possible byte
values. That means bytes: not characters, and not Unicode code points. Text is
encoded to UTF-8 first, and everything after that is a sequence over an alphabet
of size 256.

### What a merge is

A merge is an ordered pair of existing vocabulary entries and the single new
entry they produce. Write it as $(a, b) \to ab$. Nothing more.

Training repeats one round until the vocabulary reaches the size it wants:

1. Count how often each adjacent pair of current pieces occurs over the whole
   corpus.
2. Add the concatenation of the most frequent pair to the vocabulary as a new
   id.
3. Rewrite every occurrence of that pair in the corpus as the new piece.

Each round adds exactly one entry. So the vocabulary is the $\hla{256}$ bytes,
plus the $\hlb{M}$ merges from $M$ rounds, plus the $\hlc{S}$ special tokens
added by hand afterwards:

$$
|V| = \hla{256} + \hlb{M} + \hlc{S}
$$

With $|V| = 248{,}320$ and a few dozen special tokens, this model's tokenizer
learned roughly 248,000 merges.

### A worked example

Take a corpus that's mostly the words `low`, `lower`, and `lowest`. Start with
every word split into bytes:

```text
l o w        l o w e r        l o w e s t
```

The pair `(l, o)` is the most frequent, so merge it:

```text
lo w         lo w e r         lo w e s t
```

Now `(lo, w)` is the most frequent:

```text
low          low e r          low e s t
```

Then `(e, s)`, then `(es, t)`:

```text
low          low e r          low est
```

Four merges give four new vocabulary entries: `lo`, `low`, `es`, and `est`. The
word `lowest` now costs two tokens instead of six bytes. Nothing here is learned
in the machine-learning sense. It's ==frequency counting==.

## How encoding applies the merges

Training built the rules. This section answers how the engine uses them on a
string it has never seen. Encoding replays the training merges: split the
string into bytes, then apply the merge rules in the order they were learned,
repeatedly, until no rule applies.

==Order matters==: `lo` must exist before `low` can form. A tokenizer that
applied merges in any other order, for example by the rank of the resulting
token, would produce different ids for the same string, and the model would see
nonsense. That's why two tokenizer implementations that agree on the vocabulary
can still disagree on output.

> [!TIP]
> If your generated text looks subtly wrong, check that the prompt round-trips:
> encode it, decode it, and compare the result with the original.

### Pre-tokenization

Before BPE runs, a regular expression splits the text into chunks: typically
words with a leading space, runs of digits, and runs of punctuation. Merges
never cross a chunk boundary.

This rule stops the tokenizer from learning a single token for
`the quick brown fox`. It's also why a leading space is usually part of the
token: `" cat"` and `"cat"` are different ids.

## Why nothing is unrepresentable

This section answers a question every engine meets on its first strange input:
what happens to text the tokenizer never saw in training? Every one of the 256
byte values has a vocabulary entry, so the encoder can always fall back to
emitting raw bytes.

That holds for any input at all: valid UTF-8, invalid UTF-8, a null byte, or the
middle of a JPEG pasted into a chat box. There's ==no unknown token==, because
there's nothing unknown.

Character-level and word-level vocabularies don't have that property. Both must
decide what to do with a symbol they never saw during training. That means an
`<unk>` (unknown) token, which means information is destroyed before the model
sees it. Byte fallback removes that failure mode entirely.

The cost moves into the representation instead. A character the tokenizer has
never seen in context becomes three or four separate byte tokens, which the
model can still read, less efficiently. That inefficiency isn't evenly
distributed:

- **English** averages about 4 characters per token.
- **Code** averages fewer, because indentation and punctuation split.
- **Non-Latin scripts** often cost 2 to 3 times more tokens for the same
  content, since each character is 3 bytes and fewer merges were learned over
  them.

For a user, that's a real cost difference, not a curiosity: more tokens per
message, more latency, and more money.

## Special tokens and the chat template

So far every token is ordinary text. This section covers the handful that
aren't: the tokens that tell the model where a turn starts and ends, and tell
the engine when to stop.

*Special tokens* are added to the vocabulary after training and are excluded
from the merge rules. They're matched literally, before BPE runs, so merging
ordinary text can never produce them. That's what makes them usable as control
signals: the model can tell a real `<|im_end|>` from a user who typed the same
characters.

Instruction-tuned models expect structure. The *chat template*, shipped as
`chat_template.jinja`, converts a list of messages into a single string:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a KV cache?<|im_end|>
<|im_start|>assistant
```

The last line has no closing tag. The prompt ends exactly where the model
continues, so the first thing it generates is the assistant's reply.

Two ids from `generation_config.json` matter to the engine:

- **`eos_token_id` is `[248046, 248044]`.** EOS means end of sequence: when the
  model emits one of these, the reply is finished. It's a list, so the engine
  must stop on either. Checking only the first is a common bug. It shows up as
  generations that run to `max_tokens` with trailing garbage after a perfectly
  good answer.
- **`pad_token_id` is 248044.** Padding only matters when you batch sequences of
  different lengths into one rectangular tensor, and the paged cache in
  chapter 15 removes the need for it entirely.

> [!WARNING] A wrong template raises no error
> It degrades quality in a way that looks like the model being bad. When output
> seems worse than it should, print the exact string you tokenized before you
> look anywhere else.

## What the vocabulary costs

With the tokenizer done, you cross into the network. This section answers what
the large vocabulary costs in memory. Each vocabulary entry owns a row of the
embedding table and a row of the output projection, both 5120 wide, which makes
the vocabulary about a tenth of the model.

Take it one step at a time. First, multiply the rows by the width to get the
embedding's parameter count:

$$
248{,}320 \times 5120 = 1{,}271{,}398{,}400 \approx 1.27\text{B parameters}
$$

Then multiply by 2 bytes for its bfloat16 size:

$$
1{,}271{,}398{,}400 \times 2 = 2{,}542{,}796{,}800 \approx 2.54\ \text{GB}
$$

`tie_word_embeddings` is false, so the output projection is a second matrix of
exactly the same size. Together they take $2 \times 2.54 = 5.09$ GB. Divide by
the total weight footprint of 53.8 GB:

$$
\frac{5.09}{53.8} = \boxed{9.5\%}
$$

That's the share of the model spent on the vocabulary before a single layer has
done any reasoning. All of these numbers are arithmetic from the config in
chapter 1, not measurements.

Is it worth it? A bigger vocabulary means fewer tokens per document, and both
phases of inference scale with token count:

- Prefill does work in proportion to the number of prompt tokens.
- Decode does one full read of the weights per generated token.

Halving the vocabulary would return 2.54 GB of HBM, but it would lengthen every
sequence. That costs prefill FLOPs (floating-point operations) and decode steps
in proportion. The designers took the memory.

> [!NOTE] Same size, different access patterns
> Carry this into chapter 18. The 2.54 GB embedding table sits in HBM
> permanently but is barely read: a decode step gathers one row, 10 KiB. The
> output projection is read in full on every step. Two matrices of identical
> size with completely different access patterns is exactly the situation where
> mixed-precision quantization pays.

## The embedding layer

This section answers what the embedding actually computes, and why the textbook
description of it is the wrong thing to implement.

### The lookup, two ways

Textbooks introduce the embedding as a matrix multiply against a one-hot vector,
a vector that's all zeros except for a single 1. Set up the pieces:

- Let $E \in \mathbb{R}^{V \times d}$ be the table, with $V = 248{,}320$ and
  $d = 5120$.
- Let the one-hot vector $\hld{\mathbf{1}_{v}} \in \mathbb{R}^{V}$ hold a 1 in
  position $v$.

Then the embedding of token $v$ is the table transposed, times the one-hot:

$$
e = E^{\top} \hld{\mathbf{1}_{v}}
$$

In words: multiplying by $\hld{\mathbf{1}_{v}}$ zeroes out every row except row
$v$, so the result is row $v$ of $E$. That's correct, and a catastrophic way to
compute it. The following table compares it with the gather that every framework
uses:

| Method | Arithmetic | Bytes read |
|---|---|---|
| One-hot product | $2Vd = 2 \times 248{,}320 \times 5120 = 2.54$ GFLOP | $248{,}320 \times 5120 \times 2 = 2.54$ GB |
| Gather | Zero | $5120 \times 2 = 10{,}240$ bytes $= 10$ KiB |

> [!KEY] The one-hot product wastes the whole table to keep one row
> The ratio of bytes read is $2.54 \times 10^9 / 10{,}240 = 248{,}320$, which is
> $V$ exactly.

Every framework implements `nn.Embedding` as a gather. The one-hot picture is
useful for reasoning about gradients during training, and for nothing else. In
the engine it's one line:

```python
x = self.embed_tokens(input_ids)
```

The tensors have the following shapes:

| Tensor | Shape | Dtype |
|---|---|---|
| `input_ids` | `(batch, seq)` | int64 |
| `embed_tokens.weight` | `(248320, 5120)` | bfloat16 |
| `x` | `(batch, seq, 5120)` | bfloat16 |

For a 2000-token prompt, the gather moves $2000 \times 10\ \text{KiB} = 20$ MB,
scattered across a 2.54 GB table. The scatter is why it isn't quite free.

The rows sit far apart, so the reads have no locality. They miss L2, the GPU's
on-chip cache, and each one pays the full latency of a trip to HBM. It's still
the cheapest layer in the model by a wide margin.

### The residual stream at layer 0

The output of the embedding is the starting value of the *residual stream*: the
running $(batch, seq, 5120)$ tensor that every layer reads from and adds to.
Chapter 4 develops it properly. Two things about its starting state matter here.

**It carries no position information.** Two occurrences of the same token
anywhere in the prompt produce identical 5120-vectors. Everything that
distinguishes them comes later, from the rotation in chapter 5 and from
attention itself.

**Its absolute scale barely matters.** The first thing layer 0 does is
normalize it:

```python
hidden = self.input_layernorm(x)
```

RMSNorm divides each row by its own root-mean-square size, so the row's
magnitude drops out. Some architectures multiply the embedding by $\sqrt{d}$ on
the way out, to compensate for a small initialization.

A *pre-norm* model, one that normalizes at the start of each layer, doesn't need
to, and this one doesn't. If you're porting weights and see a stray
$\sqrt{5120} \approx 71.6$ factor in someone else's code, that's what it's for.

## The output projection

At the other end of the stack, a matrix of the same shape does a completely
different job. This section answers why it's one of the most expensive
operations in a decode step:

```python
x = self.norm(x)
if last_token_only:
    x = x[:, -1:, :]
return self.lm_head(x)
```

`lm_head` is `nn.Linear(5120, 248320, bias=False)`, so its weight is
$(248320, 5120)$. That's the same shape as the embedding, but it runs in the
opposite direction. It maps a hidden state to one score per vocabulary entry,
rather than a vocabulary entry to a hidden state.

Row $v$ of `lm_head.weight` is the direction in the residual stream that means
"token $v$". The score for token $v$, its *logit*, is the inner product of the
final hidden state with that direction. The sampler in chapter 11 turns those
248,320 logits into a choice of next token.

This one is a real GEMM, a general matrix multiply. During decode it runs for a
single position:

- It does $2 \times 5120 \times 248{,}320 = 2.54$ GFLOP of arithmetic.
- It reads 2.54 GB of weights.

Divide one by the other to get its *arithmetic intensity*, the work done per
byte read:

$$
\frac{2.54 \times 10^9\ \text{FLOP}}{2.54 \times 10^9\ \text{bytes}} = \boxed{1\ \text{FLOP/byte}}
$$

> [!INTUITION]
> Every weight is read once and used once. The A100 can do about 161 FLOPs in
> the time it takes to read one byte from HBM, so it needs about 161 FLOPs per
> byte to stay compute bound. At 1, the output projection spends nearly all its
> time waiting for memory: it's as memory bound as an operation gets.

Of the 53.8 GB of weights, the decode step reads about 51.3 GB: everything
except the embedding table. The output projection is 2.54 GB of that, just
under 5%.

### Slice before you project

The `last_token_only` slice is what keeps prefill honest. Without it, a
2000-token prompt computes logits for all 2000 positions, and you throw away all
but the last row:

| Quantity | Without the slice | With the slice |
|---|---|---|
| Output tensor | $2000 \times 248{,}320 \times 2\ \text{bytes} = 993$ MB | 497 KB |
| Arithmetic | $2 \times 2000 \times 5120 \times 248{,}320 = 5.09$ TFLOP | 2.54 GFLOP |

Training needs every position, because every position has a target.
==Inference needs one.==

### Weight tying, and why not here

*Weight tying* uses the same matrix for both jobs: $E$ for the lookup and
$E^{\top}$ for the projection. It halves the vocabulary's parameter cost, and it
has a tidy justification: the same vector means the same token going in and
coming out.

`tie_word_embeddings` is false here, so the matrices are separate. That's the
common choice at this scale, for two reasons:

- **The saving matters less as models grow.** Tying originated when
  vocabularies were 30,000 entries and models were a few hundred million
  parameters, where the embedding was a large fraction of the total. Here,
  untying costs 1.27B parameters out of 26.9B: 4.7%. In exchange, the output
  matrix is free to encode "what predicts this token next" rather than "what
  this token means", which are related but not the same question.
- **The two matrices want different treatment at serving time.** One is
  gathered a row at a time; the other is streamed in full every step.

The engine has to check the flag, not assume it. `ModelConfig.param_counts`
does:

```python
embed = self.vocab_size * h
lm_head = 0 if self.tie_word_embeddings else self.vocab_size * h
```

> [!WARNING] Assuming tying on an untied checkpoint
> `lm_head.weight` never gets loaded from disk. Depending on how you initialized
> the module, you get either a crash or, worse, a model that runs at full speed
> and emits fluent, confident nonsense.

## Detokenization is stateful

The last step runs the tokenizer backwards: turning generated ids into text the
user sees as it streams. This section answers why that needs state.
==One token isn't one printable unit==. A token is a run of bytes, and a
multi-byte UTF-8 character can straddle two tokens.

Here's a concrete case. The Devanagari letter `क` is three bytes, `E0 A4 95`.
With two-byte tokens, the first token carries `E0 A4` and the second starts with
`95`. Decode the first token on its own, and you have two-thirds of a character.

Decode each token independently, and you get replacement characters, `�`, at the
seams. Mixed into otherwise fine text, that garbage is called *mojibake*.

The fix is to keep per-sequence state. Append the new id, decode all the
accumulated ids, and emit only the suffix that's new *and* complete. The lab's
solution is the whole idea in a few lines:

```python
def push(self, token_id: int) -> str:
    self.ids.append(token_id)
    text = self.decode(self.ids).decode("utf-8", errors="ignore")
    new = text[self.emitted:]
    self.emitted = len(text)
    return new
```

`errors="ignore"` drops a trailing incomplete sequence rather than replacing it,
so the partial character doesn't appear yet. The token that completes it makes
the whole character appear at once.

```viz
03-streaming-detokenizer
```

`flush`, at the end of a generation, uses `errors="replace"` instead. At that
point an incomplete sequence is never going to be completed, and hiding it would
lose output.

Some pushes return an empty string. That's correct behaviour, not a bug, and the
lab checks for it: a three-byte Devanagari character split across two-byte
tokens has to wait. Getting this wrong is visible to users as flickering
mojibake in the middle of otherwise fine text.

## What goes wrong

Every mistake in this list produces bad output with no error message, which is
what makes them expensive.

**The prompt isn't what you think it is.** Here's the single highest-value
debugging habit in this course: print `repr()` of the exact string you
tokenized, and the id list. A missing newline after `<|im_start|>assistant`, a
system message the template silently dropped, and a double-applied template all
produce degraded output and no error.

**Stopping on only the first EOS id.** `eos_token_id` is a list. Build a set and
test membership.

**Double-encoding special tokens.** If you build the prompt string yourself and
then encode it with special-token parsing turned off, `<|im_start|>` becomes a
dozen ordinary byte tokens. The model has never seen that pattern, and behaves
as if the conversation has no structure.

**Computing logits for every prefill position.** It's silent, and it costs about
1 GB and 5 TFLOP on a 2000-token prompt. It shows up as prefill being
mysteriously slower than your roofline estimate, which is exactly what
chapter 10 teaches you to notice.

**Decoding each streamed token independently.** You get mojibake at multi-byte
character boundaries, and it's worse for scripts where most characters are
multi-byte.

> [!RECAP]
> - A tokenizer is a frozen vocabulary plus ordered merge rules. Apply the merges
>   in training order, or the ids are wrong.
> - Byte-level BPE can encode any byte sequence, so there's no unknown token.
>   Unfamiliar text costs more tokens instead.
> - Special tokens and the chat template control generation. Stop on every
>   `eos_token_id`, and print the exact prompt when quality drops.
> - The untied embedding and output projection cost 5.09 GB, 9.5% of the
>   weights. The embedding is a 10 KiB gather; the output projection is a
>   1 FLOP/byte GEMM over all 2.54 GB.
> - Slice to the last position before `lm_head`, and detokenize with
>   per-sequence state.

## Check your understanding

> [!QUESTION] The vocabulary has 248,320 entries and the base alphabet has 256. Where did the other 248,064 come from, and what would happen if you deleted the last 1,000 merges?
> They're learned merges, plus the special tokens. Deleting the last 1,000
> merges would still let you encode every possible input, because byte fallback
> guarantees that. But strings that used those merges would take more tokens,
> and every id above the deletion point would shift, so the embedding table
> would no longer match. The vocabulary is a contract with the weights.

> [!QUESTION] The embedding and the output projection have the same shape. Why is one of them nearly free and the other one of the most expensive operations in a decode step?
> Access pattern. The embedding is indexed: one row, 10 KiB. The output
> projection is a full GEMM against all 248,320 rows, 2.54 GB, and at batch 1 it
> does one FLOP per byte read, which is 1/161 of what the A100 needs to stay
> compute bound.

> [!QUESTION] Your engine emits one token per step and the user sees occasional black diamond question marks. Which layer is at fault?
> None of them. The model is fine; the detokenizer is decoding each token on its
> own and splitting multi-byte UTF-8 characters. Buffer the ids and emit only
> complete characters.

## Lab

> [!TRY]
> On CPU, build `build_prompt`, which renders a chat as the exact prompt string,
> and `IncrementalDetokenizer`, which streams text without splitting characters.
> You pass when both match the expected text character for character.

The lab has two parts.

**`build_prompt`.** Render a list of `{"role", "content"}` messages into the
exact string the model expects. Wrap each turn in `<|im_start|>` and
`<|im_end|>`, and open the assistant turn but leave it unclosed. The harness
compares against the expected string character for character, and checks that
no `<|im_end|>` follows the final `<|im_start|>assistant`.

**`IncrementalDetokenizer`.** Given a `decode` callable that maps a list of ids
to bytes, implement `push` and `flush` so that the concatenation of everything
you emit equals the original text, with no replacement characters anywhere.

The harness runs four strings: ASCII, Devanagari, emoji with skin-tone
modifiers, and a mixed string. It uses a tokenizer that chops text into
fixed-size byte runs, so characters split across tokens exactly as they do with
a real byte-level BPE vocabulary. It also checks that some pushes emit nothing,
which is how it knows you're buffering rather than guessing.

## Further reading

- [Neural machine translation of rare words with subword units](https://arxiv.org/abs/1508.07909) — the original BPE paper.
- [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — introduced byte-level BPE, in section 2.2.
- [The tokenizers library](https://github.com/huggingface/tokenizers)
- [Using the output embedding to improve language models](https://arxiv.org/abs/1608.05859) — the weight-tying paper.
