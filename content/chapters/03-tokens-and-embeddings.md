---
title: Tokens and embeddings
slug: 03-tokens-and-embeddings
part: "Part 2 — A forward pass"
summary: Byte-level BPE, chat templates, and the lookup that starts every forward pass.
minutes: 40
gpu: false
objectives:
  - Explain why byte-level BPE never fails on unseen input.
  - Apply a chat template and identify the tokens that control generation.
  - Describe what the embedding layer costs and why it is a gather, not a matmul.
lab: 03-tokenizer
---

# Tokens and embeddings

The model works on integers. Everything before the first matrix multiply is the
job of turning text into those integers and looking up a vector for each one.

## Byte-level BPE

Qwen uses byte-level byte-pair encoding. Text becomes UTF-8 bytes, and merge
rules combine frequent adjacent pairs into single tokens, applied in the order
they were learned.

Working on bytes rather than characters means the base vocabulary is 256 symbols
and every possible input is representable. There is no unknown token, and no
failure mode on emoji, Cyrillic, or a corrupt byte in the middle of a log file. A
character-level vocabulary has to decide what to do with a character it never saw;
a byte-level one never has that problem.

The trade is that tokens don't align with words, and they align worse the further
you get from the training distribution's language mix. English averages about 4
characters per token. Code averages fewer, because whitespace and punctuation
split. Languages written in non-Latin scripts often cost 2 to 3 times more tokens
for the same content, which is a real cost difference for users, not a curiosity.

This model's vocabulary is 248,320 entries — large, and deliberately so. A bigger
vocabulary means fewer tokens per document, so both prefill and decode do less
work per unit of text. It costs 1.27B parameters in the embedding and another
1.27B in the output projection.

## Special tokens and the chat template

Instruction-tuned models expect structure. The chat template, shipped as
`chat_template.jinja`, converts a list of messages into a single string with
control tokens:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a KV cache?<|im_end|>
<|im_start|>assistant
```

Note that the last line has no closing tag. The prompt ends where the model is
supposed to continue.

Two ids from `generation_config.json` matter to the engine:

- `eos_token_id` is `[248046, 248044]`. It's a list, so the engine must stop on
  either. Checking only the first is a common bug and shows up as generations
  that run to `max_tokens` with trailing garbage.
- `pad_token_id` is 248044. Padding only matters when you batch sequences of
  different lengths in one tensor, and the paged cache in chapter 15 removes the
  need for it entirely.

Getting the template wrong doesn't raise an error. It degrades quality in a way
that looks like the model being bad. When output seems worse than it should, print
the exact string you tokenized before you look anywhere else.

## The embedding layer

The embedding is a table of 248,320 rows, each 5120 wide. Looking up a token is a
gather, not a matrix multiply — you read one row. It's the cheapest layer in the
model by arithmetic and one of the largest by bytes.

At the other end, the output projection is a real matrix multiply against all
248,320 rows, and during decode it runs for a single position. That one GEMM is
5120 x 248,320, about 2.5 GFLOPs and 2.5 GB of reads per step. On a memory-bound
decode step, the output projection alone is a measurable fraction of the time,
which is why `HybridLanguageModel.forward` slices to the last position before
applying it:

```python
if last_token_only:
    x = x[:, -1:, :]
return self.lm_head(x)
```

Skipping that slice makes prefill compute logits for every prompt position — a
2000-token prompt produces a 2000 x 248,320 tensor, roughly 1 GB in bfloat16, and
you throw away all but the last row.

## Detokenization is stateful

Streaming output back to a user is harder than it looks, because one token is not
one printable unit. A multi-byte UTF-8 character can split across two tokens, so
decoding each token independently produces replacement characters at the seams.

The engine keeps a per-sequence decode buffer: append the new token, decode the
accumulated ids, and emit only the part that's new and complete. `tokenizers`
exposes this as an incremental decoder. Doing it wrong is visible to users as
flickering mojibake in the middle of otherwise fine text.

## Lab

Tokenize text with the real Qwen tokenizer, apply the chat template, and measure
tokens per character across English, code, and Hindi. Then implement incremental
detokenization and demonstrate that it handles a multi-byte character split across
a token boundary.

## Further reading

- [Neural machine translation of rare words with subword units](https://arxiv.org/abs/1508.07909) — the original BPE paper.
- [The tokenizers library](https://github.com/huggingface/tokenizers)
