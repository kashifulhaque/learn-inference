---
title: Constrained decoding
slug: 11a-constrained-decoding
part: "Part 3 — Making it fast"
summary: How to force a model's output to match a regex or a JSON shape, by compiling the grammar to a byte-level automaton and turning each automaton state into a token mask.
minutes: 100
gpu: false
objectives:
  - Compile a regex to a byte-level DFA and lift it to a token mask over a vocabulary that has multi-character and byte-fallback tokens.
  - Apply the mask as -inf logits at the right place in the sampling pipeline, and handle the empty mask.
  - Cache masks per automaton state, and say what XGrammar's split between context-independent and context-dependent tokens adds.
  - Skip the model for forced text with jump-forward decoding, and explain the retokenization problem it exposes.
  - Explain why nested JSON needs a pushdown automaton, and what per-request masks do to batching and speculation.
lab: 11a-constrained-decoding
---

# Constrained decoding

> [!TLDR]
> - A grammar constraint is a mask on the logits. At each step, find the
>   tokens that can still lead to a valid output, set every other logit to
>   $-\infty$, and sample as usual.
> - Compile the regex to a DFA over bytes. A token is legal in a state when
>   walking all of its bytes never falls off the automaton. Byte-fallback
>   tokens work with no special case.
> - The mask depends only on the automaton state, so build it once per state
>   and cache it. Apply it before top-k and top-p, and raise an error when it
>   is empty.
> - Text the grammar forces needs no forward pass. Jump forward, then encode
>   the forced bytes as a whole. Token paths that the mask allows are not
>   always the paths a tokenizer produces.
> - A DFA has no counter, so it can't match unbounded nesting. JSON needs a
>   stack, which is the job of a pushdown automaton.

An agent loop calls a tool by writing JSON. If one brace is missing, the
call fails, and the loop either retries or stops. You can ask the model
nicely, or you can change the sampler so that an invalid token can't be drawn.
The second option is constrained decoding, and every serving stack ships it:
vLLM exposes `regex`, `json`, `choice`, and `grammar` options for structured
outputs.

The mechanism is small. It is [chapter 11's](/c/11-sampling) logits-processor
hook with a mask inside it. The work is in building that mask fast, because the
vocabulary is 248,320 entries and the mask changes at every step. This chapter
builds the whole path on the CPU, with a toy vocabulary: regex, automaton,
token mask, sampling, cache, jump-forward, and per-row state for a batch.

## Before you start

**The sampling order, from [chapter 11](/c/11-sampling).** Processors that
edit logits run first, then temperature, then top-k and top-p, then the draw.
A processor has the shape `(input_ids, logits) -> logits`. You build one.

**Tokens are byte strings, from [chapter 3](/c/03-tokens-and-embeddings).**
A token id stands for a run of bytes. A character can straddle two tokens,
and a rare character can be spelled with single-byte tokens, the *byte
fallback*. This chapter works on bytes, so that case needs no special code.

**A DFA, in one paragraph.** A *deterministic finite automaton* has a set of
states, a start state, and, for each state and input byte, at most one next
state. Some states are *accepting*. A string matches when you can feed all of
its bytes from the start and end in an accepting state. A byte with no edge
means the string can never match. That's all the machinery the lab needs.

**The serving loop, from [chapter 16a](/c/16a-the-serving-loop-and-api).**
Sampling happens on the CPU between forwards. The mask is CPU work too, and
the later sections place it on that timeline.

## A mask on the logits

This section answers what the constraint does to the distribution.

Picture the vocabulary as a row of 248,320 cells. The model fills each cell
with a logit. The constraint is a stencil with holes over some of the
cells. Wherever the stencil is solid, the logit becomes $-\infty$, and softmax
turns $-\infty$ into probability 0. What is left is the model's own preference
among the tokens the grammar allows.

Write $L_s$ for the set of legal tokens at grammar state $s$. The masked
distribution renormalizes over that set:

$$
\boxed{p'(t) = \frac{p(t)\,\mathbb{1}[t \in L_s]}{\sum_{u \in L_s} p(u)}}
$$

In words, you keep the model's relative odds among legal tokens and drop the
rest. Setting the illegal logits to $-\infty$ before the softmax gives the
same $p'$, with no extra step.

> [!EXAMPLE] Five tokens, one of them illegal
> The pattern is `(yes|no)`. Five candidate tokens at the start have logits
> 4, 2, 1, 3, and 0 for `maybe`, `y`, `yes`, `n`, and `no`. At temperature 1 the
> model's own probabilities are 0.6364, 0.0861, 0.0317, 0.2341, and 0.0117.
> The model prefers `maybe` by a wide margin, and the grammar forbids it.
>
> Mask `maybe` and renormalize. The legal mass is $1 - 0.6364 = 0.3636$, so
> `y` becomes $0.0861 / 0.3636 = 0.2369$, `yes` 0.0871, `n` 0.6439, and `no`
> 0.0321. The ratios among the legal tokens are the same as before. The
> script `scripts/figures/11a-constrained-decoding.py` prints both rows.

This is a *local* guarantee. Every step draws from the model's distribution
restricted to legal tokens. It isn't the model's distribution over whole
strings, restricted to valid ones. A legal first token can lead only to
low-probability continuations, and the mask can't see that. The constraint
guarantees that the output is valid. It doesn't guarantee that the output is
what the model would have written, had the model been free to try again.

## From a regex to a byte-level DFA

This section answers how a pattern becomes a machine that you can step one byte at a time.

The pattern `(yes|no)` has a three-letter path and a two-letter path. A DFA
for it has a start state, a branch on `y` or `n`, a chain of `e` then `s`,
a chain of `o`, and one accepting state at the end of each path. That is the
picture. A compiler builds it in four steps, and the lab provides one:

1. **Parse** the pattern to a tree of byte sets, concatenations,
   alternations, and repeats.
2. **Build an NFA** from the tree, by Thompson's construction. An NFA can
   have several next states for a byte, and empty moves.
3. **Run the subset construction.** Each DFA state is a set of NFA states,
   so the next state is always one set.
4. **Trim.** Delete every state from which no accepting state is reachable.

The fourth step matters for decoding. After trimming, a missing edge means
that no continuation can match. Without the trim, a token could walk into a
state that still exists but is a dead end, and the mask would let the model
write itself into a corner.

The alphabet is bytes, not characters. A non-ASCII character such as an
e with an acute accent compiles to its two UTF-8 bytes in sequence. The
automaton then handles a token that ends in the middle of that character
without any extra rule.

> [!NOTE] The provided compiler is small on purpose
> It supports literals, classes such as `[a-z]` and `[^"]`, `\d`, `.`,
> groups, `|`, `*`, `+`, `?`, and `{m,n}`. A negated class is a complement
> inside printable ASCII. Libraries such as Outlines, XGrammar, and
> llguidance take full regex or grammar syntax. The idea is the same.

A pattern for a small JSON object shows why the grammar must be tighter than
you first write it:

```text
\{"name": "[a-z]{1,8}", "age": (0|[1-9][0-9]{0,2})\}
```

The age is `(0|[1-9][0-9]{0,2})` and not `[0-9]{1,3}`, because the looser
pattern accepts `007`. Python's `re.fullmatch` accepts that text, and
`json.loads` rejects it. A constraint is only as strict as its pattern. The
compiled automaton for this pattern has 34 states.

## Lifting the automaton to tokens

This section answers which tokens are legal in a state.

The model doesn't emit bytes. It emits tokens, and a token is a byte string
of one or more bytes. Write $\delta^*(s, w)$ for the state you reach by
feeding the bytes of $w$ from state $s$. A token $t$ is legal when its bytes
keep the automaton alive:

$$
\boxed{t \in L_s \iff \delta^*(s,\ \mathrm{bytes}(t)) \neq \text{dead}}
$$

Two tokens need a separate rule. The end-of-sequence token has no bytes. It is
legal only when $s$ is accepting, because stopping there leaves a complete
match. A padding or other special token with empty bytes is never legal. If it
were, the walk would return $s$ for it, and the sampler could draw it forever
without making progress.

> [!EXAMPLE] The mask for `(yes|no)` over a 296-token toy vocabulary
> The vocabulary has the end token, a padding token, 256 single-byte tokens
> (the byte fallback), and 38 multi-character tokens such as `ye`, `es`,
> `yes`, `no`, and `maybe`.
>
> - At the start, 5 tokens are legal: `y`, `n`, `ye`, `yes`, and `no`.
> - After `y`, 2 are legal: `e` and `es`. The multi-character token `es` is
>   legal because both of its bytes keep the match alive.
> - After `ye`, only `s`. After `yes`, only the end token.
>
> The figure lets you step through these states.

```viz
11a-token-mask
```

The wrong shortcut is to check only the first byte of a token. After `a`, in
the pattern `ac|abd`, the token `bc` starts with a legal byte, `b`. Its second byte
`c` can't follow `ab`. A first-byte check admits `bc` and the output no
longer matches. The lab's mask check catches that.

The code is short. The only cost is the loop over the vocabulary:

```python
def build_token_mask(dfa, state, vocab_bytes, eos_id):
    allowed = []
    for token_id, token in enumerate(vocab_bytes):
        if token_id == eos_id:
            allowed.append(dfa.accepting(state))
        elif len(token) == 0:
            allowed.append(False)
        else:
            allowed.append(walk(dfa, state, token) != DEAD)
    return torch.tensor(allowed, dtype=torch.bool)
```

## Where the mask goes in the sampling pipeline

This section answers where the mask sits relative to temperature, top-k, and top-p, and what happens when nothing is legal.

[Chapter 11](/c/11-sampling) runs logits processors on the raw logits,
before temperature. The mask is such a processor. Put it first for two
reasons.

- Temperature divides every logit by the same positive number, and
  $-\infty / T = -\infty$. So the mask and temperature commute, and the order
  between those two is a matter of taste.
- Top-k and top-p don't commute with the mask. Top-k keeps the $k$ largest
  logits. If the $k$ largest are all illegal, and you mask afterward, no token
  survives. Mask first, and top-k keeps the $k$ best *legal* tokens.

> [!EXAMPLE] Top-k that keeps only illegal tokens
> Take the five-token row from the earlier example, with top-k set to 1.
> Top-k first keeps `maybe`. The mask then removes `maybe`, and the set is
> empty. Mask first, and top-k keeps `n`, the best legal token.

The empty mask is a different failure. It happens when no token in the
vocabulary can continue the match, for example when the pattern needs a `}`
and no token contains one. The softmax of a row that is all $-\infty$ is NaN,
and `torch.multinomial` on a NaN row raises a runtime error about a probability
tensor that contains `inf`, `nan`, or a negative element. That message says nothing
about the grammar. The lab raises an `EmptyMaskError` instead. A real engine turns that
error into a failed request, because silently drawing an illegal token breaks
the guarantee the user asked for.

The full step is the following, with the lab's name for each piece:

1. `fsm.mask(state)` returns the legal set for this request's state.
2. `sample_constrained(logits, mask, temperature, top_k)` masks, scales,
   truncates, and draws.
3. `fsm.advance(state, token)` moves to the next state, and raises
   `ValueError` for an illegal token.

Because the processor has chapter 11's `(input_ids, logits) -> logits`
shape, `ConstraintProcessor` plugs into the same hook as a ban-token
processor. The difference is that it carries state between steps.

## Cache the mask per state

This section answers how to avoid rebuilding the mask at every step.

The mask depends on the automaton state, not on the position in the output or
on the prompt. Two requests in the same state have the same mask, and one
request that returns to a state, inside a loop such as `[a-z]*`, has it again.
So build each mask once and store it by state.

The pattern above has 34 states, so the cache holds at most 34 masks. At one
byte per token and 248,320 tokens, a mask is 248,320 bytes, and 34 of them
come to 8,442,880 bytes, about 8.4 MB.

Rebuilding costs more. The lab's loop is pure Python, and a 248,320-token
vocabulary takes about 34 ms per mask on the author's CPU, against about
0.07 µs for a cached lookup. Your machine will differ. The 42.2 ms decode
step lower bound from chapter 0 (53.8 GB at 1275 GB/s) puts that rebuild in the same range as a
whole forward pass, which is why every engine caches. The lab's `TokenFSM`
counts `builds` so that a test can check that 201 calls build once.

Production libraries go much further than a dict. llguidance's page reports
about 50 µs of CPU time per token for a 128k-token vocabulary, by walking a
prefix tree of the vocabulary instead of looping over it. XGrammar splits the
vocabulary by how a token's legality is decided. A *context-independent*
token is decided by the automaton state alone, so its answer can be prechecked
and cached. A *context-dependent* token depends on what is on the grammar's
stack, so it has to be checked at run time. Outlines indexes the
whole vocabulary against the automaton ahead of time. In this chapter's
regex-only lab, every token is context-independent, so the per-state cache is the
whole story.

## Jump-forward: skip the model for forced text

This section answers what to do when the grammar leaves no choice.

Look at the JSON pattern at the start. The only legal first byte is `{`,
then `"`, then `n`, and so on through `{"name": "`. Ten bytes with no choice.
A forward pass over a 27-billion-parameter model to decide each of them
produces no information.

A state is *forced* when it isn't accepting and has exactly one outgoing byte.
Follow forced states until you reach one that branches or accepts. The bytes on
the way are the forced text. **Jump-forward decoding** appends all of it at
once, then resumes sampling. For the JSON pattern, the forced text at the start is 10
bytes, and the encoded form is 3 tokens: `{"`, `name`, and `": "`. In the example
`{"name": "bo", "age": 7}`, 19 of the 24 bytes are forced.

Two details trip up first implementations:

- **Stop at an accepting state.** In `yes(!)?`, after `yes` the only outgoing
  byte is `!`. The state is accepting, so the model may choose to stop. The
  next byte isn't forced.
- **Count forced bytes, not forced tokens.** At the start of the JSON pattern
  the mask allows two tokens, `{` and `{"`. Both spell forced text. A rule that
  jumps only when exactly one token is legal would miss it.

> [!KEY] Jump-forward works on bytes, then encodes
> The unit of "forced" is a byte of the automaton. Turning those bytes into
> token ids is a second step, and it is where the next problem begins.

## The retokenization problem

This section answers why the tokens the mask allows aren't always the tokens a tokenizer would produce.

A tokenizer encodes a string in one deterministic way. The lab's stand-in is
`greedy_encode`, longest match from the left. Real byte-pair encoding applies
merge rules, but the point is the same. The mask doesn't enforce it. It
admits every token path that spells a valid string.

Here is one run from the lab. The model sampled the path
`{"`, `name`, `": "`, `z`, `x`, `w`, `", "`, `age`, `"`, `:`, ` `, `1`, `}`.
The text is `{"name": "zxw", "age": 1}`. Longest match encodes the same text
with a different tail: `age`, `":`, ` `, `1}` where the model sampled `age`,
`"`, `:`, ` `, `1`, `}`. Both paths spell the same valid string. They aren't
the same ids. Over 250 seeded runs against the JSON
pattern, the sampled path differed from the greedy encoding every time.

This matters in three places:

- **Re-encoding.** The next turn of a chat re-encodes the transcript. The model
  now reads ids that differ from the ones it produced.
- **Prefix caching.** [Chapter 15a](/c/15a-prefix-caching-and-radix-trees)
  keys its blocks on token ids. A different path misses the cache.
- **Jump-forward.** SGLang's compressed-FSM write-up describes the same
  trap. If you encode the forced text on its own, the next token's probabilities
  differ from what natural generation would have given. Their fix is to
  append the forced text as a string and re-tokenize the whole sequence, which
  they report adds about 4% overhead.

A related trap is the boundary. A model that was trained on text often emits a
closing quote fused with a comma, as `",`. A rule that forces the quote as a
separate step, character by character, removes that token from the mask and
pushes the model onto a path that the tokenizer wouldn't produce. The same write-up notes that this
can loop. The byte-level mask in this chapter avoids it, because `",` is
legal wherever both of its bytes are.

## Nested JSON needs a stack

This section answers why a regex can't describe all of JSON.

A DFA has a fixed number of states, so it can only count up to a fixed
bound. JSON arrays and objects nest to any depth, and a closing bracket must
match the opener. Balancing $n$ opening brackets with $n$ closing brackets, for every $n$,
needs unbounded memory.
By the pumping lemma, no DFA recognizes it. You *can* write a regex for
nesting up to depth $d$, but each extra level costs more states.

The lab's `nested_array_regex(d)` builds that regex. The compiled automaton
has 7, 17, 37, 77, and 157 states for depths 1 to 5. These are unminimized
subset-construction counts, so read them as growth, not as a tight bound. A
depth-$d$ automaton rejects an array nested to depth $d + 1$. No single
automaton accepts them all.

A *pushdown automaton* adds a stack. Push on `[`, pop on `]`, and accept only
when the stack is empty. That is the machine for a context-free grammar, and
it is what the JSON grammar needs. The state of the mask is now a pair, the
automaton state and the stack. A context-independent token still has one
answer. A token such as `]` is legal or not depending on the top of the stack,
so it is context-dependent, and checking it has to be done at run time. That is the
split XGrammar uses, and it is why a pure per-state cache doesn't cover a
grammar with nesting.

## Batching, speculation, and the serving loop

This section answers how per-request masks interact with the rest of the engine.

**Batching.** Requests in one batch are in different grammar states, and some
have no constraint. The mask is a tensor of shape $(B, V)$, one row per
request. For $V = 248{,}320$ and $B = 128$, a boolean mask is 31,784,960 bytes.
Moving it over PCIe at the 32 GB/s one-way rate from
[chapter 19](/c/19-tensor-parallelism) takes 0.993 ms, about 2.4% of the 42.2 ms
decode lower bound. A bit-packed mask is 8 times smaller, 3,973,120 bytes, and
takes 0.124 ms. Both figures are arithmetic, not measurements. Rows without a
constraint get an all-true mask. The lab's `ConstraintProcessor` keeps one
state per row, and a version that masks every row with row 0's state fails
its check.

**The serving loop.** In [chapter 16a](/c/16a-the-serving-loop-and-api)'s
terms, the mask for step $t$ depends only on the state after token $t - 1$. It
is known before forward $t$ starts, so you can build it during the forward and
hide it the way you hide the next schedule. `advance` needs the sampled token,
so it joins the post stage and stays on the clock. XGrammar's abstract makes the
same point about overlapping grammar work with GPU execution.

**Speculation.** [Chapter 20](/c/20-speculative-decoding) verifies $\gamma$
drafted tokens in one pass. Under a grammar, the mask at draft position $j$
depends on the state after drafted tokens $1$ to $j - 1$. Walk the automaton
along the draft, keep the list of states, and build one mask per position. Apply
each mask to the target's distribution *before* the accept test. A drafted token the grammar
forbids then has target probability 0 and is rejected, and the residual
resample draws only legal tokens. On a rejection, roll the automaton back to the
saved state for that position. A state is an integer, so the rollback is a list
index.

## What goes wrong

These are the failures the lab is built to catch. Each one still produces
output, which makes it hard to spot.

**Checking only a token's first byte.** The mask admits `bc` after `a` in `ac|abd`.
Walk every byte, and treat the first dead byte as a veto.

**Applying top-k before the mask.** The $k$ survivors can all be illegal, and
the mask then empties the set. Mask first.

**Letting empty tokens through.** A padding token walks to the same state, so
it looks legal, and the sampler can draw it forever. Allow empty bytes only for
the end token, and only in an accepting state.

**Forcing through an accepting state.** In `yes(!)?`, the byte `!` is not forced
after `yes`, because stopping is also legal.

**Trusting `re.fullmatch` over the real parser.** `[0-9]{1,3}` matches `007`
and `json.loads` rejects it. Write the pattern for the parser you will use.

**One mask for the whole batch.** Rows are in different states. A mask built
from row 0's state hands every other row illegal tokens.

> [!RECAP]
> - A constraint is a $-\infty$ mask on the logits. It keeps the model's odds
>   among legal tokens, and it is a local, step-by-step restriction.
> - Compile the grammar to a byte-level DFA, trimmed so that a missing edge
>   means no match is possible. A token is legal when all of its bytes keep
>   the automaton alive. The end token is legal only in an accepting state.
> - Mask before top-k. Raise an error on an empty mask. The mask is a
>   chapter 11 processor, and it carries state per request.
> - Cache one mask per state. The 34-state JSON pattern needs 8.4 MB of masks at a
>   248,320-token vocabulary. Rebuilding one takes about 34 ms in pure
>   Python.
> - Jump over forced bytes, stop at accepting states, and encode the bytes as a
>   whole. The mask allows token paths that a tokenizer wouldn't produce.
> - Nesting needs a stack. A DFA for depth $d$ rejects depth $d + 1$.

## Check your understanding

> [!QUESTION] In `ac|abd`, the text so far is `a`. Is the token `bc` legal?
> No. Its first byte `b` keeps the match alive, but after `ab` only `d` can
> follow, so the second byte `c` falls off the automaton. A mask that checks
> only the first byte would admit it.

> [!QUESTION] Top-k is 1, and the model's best token is illegal. What does each order of operations do?
> Top-k first keeps only the illegal token, the mask removes it, and nothing is
> left. Mask first, and top-k keeps the best legal token, so the draw succeeds.

> [!QUESTION] The pattern `\{"a": [0-9]{1,3}\}` produces `{"a": 007}`. Is the output valid JSON?
> No. `re.fullmatch` accepts it, and `json.loads` rejects the leading zero.
> The constraint can't be stricter than the pattern. Use `(0|[1-9][0-9]{0,2})`.

> [!QUESTION] After `yes` in the pattern `yes(!)?`, why doesn't jump-forward emit `!`?
> The state is accepting, so the model may choose to stop. Only a state that is
> not accepting and has one outgoing byte is forced.

> [!QUESTION] A batch of 128 requests has 128 different grammar states. How big is the mask, and does the cache still help?
> The mask is $128 \times 248{,}320$ booleans, 31,784,960 bytes. The cache still helps,
> because each row's mask is looked up by state, and the pattern has only 34
> states. Each request pays for a lookup, not for a rebuild.

## Lab

> [!TRY]
> Implement the token mask, the constrained draw, a cached `TokenFSM`, the
> generation loop, jump-forward decoding, and a per-row processor on the CPU.
> You pass when 1000 seeded completions across four patterns all match
> `re.fullmatch` (and `json.loads`, for the JSON ones), the cached mask is
> built once and is faster, and jump-forward equals the constrained decode.

The lab provides the regex compiler, the `DFA` class, and a 296-token toy
vocabulary with the byte fallback. You write ten pieces, from the smallest
to the whole. The first check calls only `walk`.

- `walk` feeds a token's bytes to the automaton and returns the state or
  `DEAD`.
- `build_token_mask` returns the legal set as a boolean tensor, with the end
  token and empty-token rules. The check includes a byte-fallback case: after
  `caf`, the mask allows `e` and the first byte of an accented e, then only
  its second byte.
- `sample_constrained` masks first, then scales, then applies top-k, and raises
  `EmptyMaskError` on an empty mask.
- `TokenFSM.advance` moves the state and raises `ValueError` for an illegal
  token. `mask` builds once per state. A vocabulary of about 30,000 tokens shows the
  cached call at least 5 times faster than a build.
- `generate` drives the loop. Four patterns times 250 seeds all match.
- `greedy_encode` is the longest-match encoder.
- `jump_forward` returns the forced token ids and the state, and agrees with
  the constrained decode for 30 seeds.
- `nested_array_regex(d)` builds the depth-bounded pattern.
- `ConstraintProcessor` masks each row with its own state.

The tests also run named wrong versions: a first-byte mask, top-k before the
mask, jump-forward through an accepting state, and a one-state batch. Each
one fails a check. The lab reports `mask_build_ms`, `cached_mask_us`,
`completions_matching`, `retokenize_mismatch_rate`, and `jump_tokens_saved`.

## Further reading

- [Efficient guided generation for large language models](https://arxiv.org/abs/2307.09702) — Outlines. It indexes the vocabulary against a finite-state machine.
- [XGrammar: flexible and efficient structured generation engine for large language models](https://arxiv.org/abs/2411.15100) — the split between context-independent and context-dependent tokens, and the overlap of grammar work with GPU execution.
- [Fast JSON decoding for local LLMs with compressed finite state machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/) — SGLang's jump-forward decoding and its retokenization discussion.
- [llguidance](https://github.com/guidance-ai/llguidance) — a grammar engine that computes masks from a prefix tree of the vocabulary.
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) — the `regex`, `json`, `choice`, `grammar`, and `structural_tag` options, and the backends behind them.
