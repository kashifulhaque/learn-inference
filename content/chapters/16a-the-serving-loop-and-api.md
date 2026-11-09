---
title: The serving loop and the streaming API
slug: 16a-the-serving-loop-and-api
part: "Part 5 — Serving"
summary: The engine-core loop that turns a schedule into streamed text, including stop strings that span tokens, cancellation, and a simulated clock for CPU overlap.
minutes: 110
gpu: false
objectives:
  - Trace the engine-core loop and say which stages sit on the CPU critical path.
  - Hold a stop string that spans a token boundary, and emit it only once it is resolved.
  - Cancel a request so its blocks and its recurrent-state slot return to the pool.
  - Explain why CPU time between forwards shows up as GPU idle time, and how overlapping the next schedule with the forward hides it on a simulated clock.
lab: 16a-serving-loop
---

# The serving loop and the streaming API

> [!TLDR]
> - One engine step is six stages: add, schedule, forward, sample, detokenize,
>   and stream. The forward is the GPU stage. Scheduling, sampling, and
>   detokenizing run on the CPU between forwards.
> - A stop string can start in one token and finish in the next. Hold any
>   suffix that is still a prefix of a stop string. Emit it when the next
>   token shows it is not the stop, and do not emit the stop itself.
> - When the client disconnects, cancel that request. Its KV blocks and its
>   recurrent-state slot return to the pool. A block another request still
>   holds stays allocated.
> - CPU work between forwards is GPU idle time. The next schedule can run
>   during the current forward. Sampling cannot, because it needs the logits.
>   The lab measures that gap on a simulated clock.

[Chapter 16](/c/16-continuous-batching) decides who runs in the next forward.
Nothing in that chapter turns the decision into text a client can read, or
stops the work when the client goes away. The kernels, the cache, and the
sampler are already built. This chapter is the loop that calls them once per
step and the bytes that leave the process.

A serving stack splits that loop into two processes. One speaks HTTP. The
other, the *engine core*, owns the scheduler and the model. You write the
engine core against a fake model, `step(batch) -> logits`, so the tests stay
on the CPU. Chapter 17 defines time to first token and goodput. Its lab does
not drive this loop yet. `EngineCore` is the object that benchmark
calls.

## Before you start

**The scheduler, from
[chapter 16](/c/16-continuous-batching).** A step still means one forward over
a chosen set of tokens. The scheduler admits, prefills, decodes, and retires.
This chapter calls that scheduler. It does not re-derive continuous batching,
chunked prefill, or preemption.

**Detokenizing, from
[chapter 3](/c/03-tokens-and-embeddings).** A token is a run of bytes, and a
character can straddle two tokens. The incremental detokenizer appends the
new id, decodes the whole buffer, and emits only the new suffix that is
complete UTF-8. Stop strings sit on the text that the decoder emits. You hold a suffix of
the *text* that might still grow into a stop string. This chapter does not
re-teach byte-pair encoding.

**Sampling, from [chapter 11](/c/11-sampling).** The forward returns one row
of logits per sequence. Sampling turns that row into a token id. The lab
uses argmax, with the lowest index winning a tie, so the output is
deterministic.

**Blocks and slots, from [chapter 9](/c/09-the-kv-cache) and
[chapter 15](/c/15-paged-attention).** Full-attention layers store keys and
values in fixed-size blocks from a shared pool. The 48 linear-attention
layers store a fixed recurrent state in a slot. Chapter 2 counts that state
at 147.8 MiB per sequence. Cancel returns both the blocks and the slot.
[Chapter 18](/c/18-quantization) changes how many bytes a block holds. The
pool still hands out ids, so this chapter does not re-derive the formats.
[Chapter 20](/c/20-speculative-decoding) may accept more than one token in a
step. The stop hold still walks the resulting text in order. The lab samples
one token per step.

**Server-sent events.** A streaming HTTP response is a series of records.
Each record is a line starting with `data: `, and a blank line ends the
record. The chat API uses that framing for `chat.completion.chunk` objects.

## The engine-core loop

This section names the six stages and the process that owns each one.

A request arrives as HTTP. The API process checks the body, tokenizes the
prompt, and sends an engine request to the core. The core does not parse
JSON. It appends the request to the scheduler and, on the next step, gives
it blocks.

vLLM documents `EngineCore.step` as "Schedule, execute, and make output."
The v1 engine blog describes the core as an isolated loop that owns the
scheduler and the model executor, with the API process on the other side of
a ZeroMQ socket. SGLang's `Engine` docstring splits the same idea into three
pieces: a `TokenizerManager` in the server process, a `Scheduler`
subprocess that batches and runs the model, and a `DetokenizerManager`
subprocess that turns ids into text. TensorRT-LLM's executor takes the
client's requests through `enqueueRequest`, runs in-flight batching inside
the executor, and returns tokens through `awaitResponses`.

The loop you implement is the inner one. One step does the following:

1. **Add.** Take any requests that arrived since the last step, and any
   cancels. Allocate blocks and a recurrent slot for each new request.
2. **Schedule.** Ask the scheduler which requests run, and how many tokens
   each one contributes. Chapter 16 is this stage.
3. **Forward.** Run the model on that batch. This is the only stage that
   needs the GPU. The lab replaces it with `model.step(ids) -> logits`.
4. **Sample.** Turn each logit row into a token id.
5. **Detokenize.** Append the id to that request's byte buffer, then apply
   the stop hold to the new text.
6. **Stream.** Send the text that is safe to show, or a finish reason, back
   to the API process.

> [!KEY] The core owns the step, and the API process owns the socket
> Tokenizing the HTTP body and writing the response can run beside the
> core. The forward, the sample, and the update of the scheduler stay in
> the core, in that order, because each one reads the previous one's output.

The following sketch is the step, with the GPU call marked:

```python
def step(self):
    self._take_new_requests_and_cancels()
    batch = self.scheduler.schedule()
    if not batch:
        return []
    logits = self.model.step(batch.ids)  # GPU in a real engine
    tokens = [argmax(row) for row in logits]
    return self._detokenize_and_finish(batch, tokens)
```

`_detokenize_and_finish` is where a request leaves the batch. A finished
request returns its blocks and its slot before the next schedule. That is
the same retire step as chapter 16, called from the engine instead of from
a test harness.

Walk one step with two requests, A and B, both decoding. The scheduler's
output is two ids. The fake model returns two logit rows. Argmax picks one
token for A and one for B. A's piece completes a stop string, so the step
emits A's text before the stop, sets `finish_reason` to `stop`, and
releases A's blocks and slot before it returns. B's piece is ordinary
text, so the step emits it and leaves B in the running set. The next
step's schedule sees only B. Nothing in that story depends on how the
logits were computed, which is why a one-hot fake model is enough to test
the loop.

The API process can be doing other work during the forward. It can
tokenize a third request that arrived during the step, and it can write
B's previous token to its socket. It cannot sample A's new token yet. That
split, host work beside the forward versus host work that reads the
logits, is the whole of the idle-time section.

On a real engine the batch is not a Python list of ids. vLLM's v1 scheduler
represents the decision as a map from request id to a token count, so one
structure covers a prefill chunk and a decode token. You already build that
decision in chapter 16. This chapter consumes it.

## Where the CPU sits on the critical path

This section answers which of the six stages the GPU has to wait for.

The forward is the GPU stage. Everything that prepares its inputs, and
everything that reads its outputs, runs on the CPU. In a serial loop the
order is fixed:

1. The CPU builds the schedule and the input tensors.
2. The GPU runs the forward.
3. The CPU samples, detokenizes, and pushes the stream.

The GPU cannot start the next forward until step 3 finishes, because the
next schedule depends on which requests finished on this step. Call the three
durations $\hla{S}$ for the schedule, $\hlb{F}$ for the forward, and
$\hlc{P}$ for the post stage (sample, detokenize, stream). Over $n$ serial
steps the wall time is as follows:

$$
n(\hla{S} + \hlb{F} + \hlc{P})
$$

The GPU is busy only during $\hlb{F}$. The idle fraction of the wall is the
share that is not a forward:

$$
\frac{\hla{S} + \hlc{P}}{\hla{S} + \hlb{F} + \hlc{P}}
$$

That fraction is the quantity a profile calls GPU idle time, even though
the cause is host work. The device is waiting on the CPU.

Two published engines describe the same split. The vLLM v1 blog says the
core loop depends on CPU work between forwards, and that the gap shows up
once the GPU step gets short: for Llama-8B on an H100, they put that GPU
time as low as about 5 ms. SGLang moves detokenizing into its own process
so that work is not inside the scheduler's loop. TensorRT-LLM's executor
hides the loop from the client and still has to do it.

Chapter 0's arithmetic for *this* model is a different point on the same
curve. One decode step reads 53.8 GB. At the measured 1275 GB/s copy, the
memory lower bound is about 42 ms, and the arithmetic lower bound is about
0.17 ms. A schedule of a few milliseconds is then a small slice of the
step. The H100 number is a smaller model on a faster GPU, which is why the
same host work dominates there and not here. Both figures are other
people's arithmetic or their report. This machine has no GPU, so the lab
does not time either one.

> [!INTUITION] Idle here means the forward is not running
> The GPU is idle whenever the clock is advancing and no forward is in
> progress. Scheduling, sampling, and detokenizing are useful work. They
> still count as idle time for the GPU, because the GPU cannot do them.

The lab makes $\hla{S}$ artificially large, in ticks, so you can watch the
fraction move. Overlapping the next schedule, in the section of that
name, hides $\hla{S}$ inside $\hlb{F}$. $\hlc{P}$ stays put.

## Stop strings that span tokens

This section answers what you may send the client after each sampled token.

The chat API accepts up to four stop sequences. The returned text does not
contain the stop sequence. `finish_reason` is `stop` when the model hits a
stop sequence or its own end token, and `length` when the request hits its
maximum token count.

A stop sequence is a string, and a token is not a string boundary you
control. The sequence `END` can arrive as the piece `EN` and then the piece
`D`. If you emit `EN` as soon as you sample it, the client has already seen
a prefix of the stop, and you cannot take it back.

The rule has two cases. Keep a pending string, which is text the detokenizer
has produced and you have not yet sent.

- If a stop string occurs in the pending text, emit only the text before
  it. Drop the stop and anything after it. Finish with reason `stop`.
- Otherwise, hold the longest suffix of the pending text that is a proper
  prefix of some stop string. Emit the rest. The held suffix stays pending
  for the next token.

"Proper" matters. The full stop is the first case, not a hold. A hold is a
prefix that might still grow into the stop, and might not.

Apply this *after* chapter 3's detokenizer. That layer already hid an
incomplete UTF-8 sequence. The stop layer hides an incomplete stop string
in the text that layer emitted. The two buffers are different, and you need
both. The lab's pieces are already text, so the lab starts at the stop
layer.

> [!EXAMPLE] `END` arrives as `EN` and then `D`
> The pieces are `Say `, `EN`, and `D`. The only stop string is `END`.
>
> After `Say `, nothing is a prefix of `END`. Emit `Say `.
>
> After `EN`, the pending suffix `EN` is a prefix of `END`, and `END` does
> not occur yet. Emit nothing. Hold `EN`.
>
> After `D`, the pending text is `END`. The stop matches. Emit nothing
> more, and finish with reason `stop`. The client saw `Say `.
>
> If the third piece had been `!` instead of `D`, the pending text would be
> `EN!`. That is not a prefix of `END`, and `END` does not occur. Emit
> `EN!`. The hold resolves by releasing, not by finishing.

When the request hits its token limit without a match, flush the hold. The
held characters are ordinary text that will not become a stop. Hiding
them would drop the end of the reply. A match does not flush: the hold is
empty, because the stop consumed it.

If several stop strings match, take the earliest start. If two start at the
same index, take the longer one, so a short stop does not leave the tail of
a longer stop in the output.

> [!WARNING] Emitting the first piece of a stop
> The failure is silent. The final string can look almost right, except
> the client already received `EN` on an earlier event. Compare the text
> of each event with the offline string, not only the concatenation at
> the end. The lab's spanning check does that on the step that samples
> `EN`.

The offline string is the concatenation of the pieces, cut at the first
stop, with the stop itself removed. A correct stream's pieces join to that
string. An implementation that emits `EN` and then tries to finish joins
to something longer. The lab checks 50 seeded requests against that offline
cut, and then checks the intermediate event on its own.

## Cancellation and the block pool

This section answers what goes back to the pool when the client leaves.

A stream ends because the client disconnects, because a proxy times out,
or because the caller sends a cancel. The API process notices first. It
tells the core to drop that request id. vLLM's core exposes this as
`abort_requests`. TensorRT-LLM's executor exposes `cancelRequest`. In both
cases the request must leave the scheduler, and the memory it held must
become usable by someone else.

Two kinds of memory come back:

- **KV blocks**, from the paged pool in chapter 15.
- **One recurrent-state slot**, the fixed state of the linear-attention
  layers. Chapter 2's 147.8 MiB sits in that slot until you return it.

Finish and cancel take the same path. A request that hits `stop` or
`length` releases in the step that samples its last token. A request that
is cancelled releases even if it has not sampled anything yet. Releasing an
id that is not in the pool does nothing, so a second cancel is safe.

Blocks can be shared. Chapter 15's prefix sharing lets two
requests reference one block. The pool keeps a reference count. `release`
drops this request's references and returns its slot. A block returns to
the free list only when the count hits zero.

Walk a pool of 8 blocks and 4 slots:

- Request A allocates 2 blocks. Free blocks: 6. Free slots: 3.
- Request B allocates 1 block. Free blocks: 5. Free slots: 2.
- B also attaches one of A's blocks. The free count stays 5, and that
  block's count becomes 2.
- Cancel A. A's private block returns. The shared block stays with B.
  Free blocks: 6. Free slots: 3.

A release that ignores the count returns both of A's blocks. The free
count becomes 7, and B's block is also on the free list. The next
`alloc` can hand B's block to someone else. The lab expects free 6 and
free slots 3 after that cancel, and it expects a lone request's cancel to
put the free count back where it started.

> [!WARNING] Freeing a block another request still holds
> The symptom is a later request reading someone else's tokens, or a free
> count that is higher than the number of blocks nobody references. Count
> references. Do not return a block id only because the cancelled request
> listed it.

The lab's scheduler is deliberately smaller than chapter 16's. It admits
every waiting request, runs one new token each, and relies on you for the
release. Plug the same `release` into chapter 16's retire path when you
connect the two.

## Overlap the next schedule with the forward

This section answers how to hide $\hla{S}$ without pretending you can hide
$\hlc{P}$.

Sampling reads the logits, so $\hlc{P}$ cannot start until the forward
returns. The *next* schedule does not need those logits if you are willing
to guess that the current requests will still be running. You can build it
while the GPU is inside $\hlb{F}$.

vLLM's engine source keeps a batch queue so it can "asynchronously schedule
and execute batches," which pipeline parallelism uses to avoid bubbles. The
same file runs the ZeroMQ sockets on threads "to overlap ZMQ socket IO with
GPU." The v1 blog's split, API process versus engine core, is the same idea
one level up: tokenization and streaming run beside the core. The lab
isolates one overlap you can add by hand. It is not a model of their
scheduler.

The simulated clock has three rules:

- The first schedule cannot hide. No forward is running yet.
- Each later schedule runs during the previous forward. If
  $\hla{S} > \hlb{F}$, the extra $\hla{S} - \hlb{F}$ stays on the clock
  after the forward.
- The post stage follows the forward it belongs to.

For $n$ steps, the GPU-busy ticks are $n\hlb{F}$. The idle ticks are the
first schedule, the leftovers, and every post stage:

$$
\hla{S} + (n - 1)\max(0, \hla{S} - \hlb{F}) + n\hlc{P}
$$

The wall is those idle ticks plus the busy ticks. When the schedule fits
inside the forward, the $\max$ term is zero, and the idle fraction
collapses to the following:

$$
\boxed{\frac{\hla{S} + n\hlc{P}}{\hla{S} + n(\hlb{F} + \hlc{P})}}
\quad\text{when }\hla{S} \le \hlb{F}
$$

Compare that with the serial fraction $(\hla{S}+\hlc{P})/(\hla{S}+\hlb{F}+\hlc{P})$. The overlapped form charges $\hla{S}$ once, not once per step. $\hlc{P}$ is still charged $n$ times.

> [!EXAMPLE] Eight steps, schedule 6, forward 10, post 2
> These are ticks, not milliseconds. $\hla{S} \le \hlb{F}$, so the leftover
> is zero.
>
> The serial wall is $8 \times (6 + 10 + 2) = 144$ ticks, of which
> $8 \times 10 = 80$ are GPU-busy. Idle ticks: $144 - 80 = 64$. The idle
> fraction is $64/144 = 4/9$, which is 0.444 to three decimals.
>
> The overlapped wall is $6 + 8 \times (10 + 2) = 102$ ticks, still with
> 80 GPU-busy. Idle ticks: $102 - 80 = 22$. The idle fraction is
> $22/102 = 11/51$, which is 0.216 to three decimals.
>
> `scripts/figures/16a-the-serving-loop-and-api.py` prints both rows. The
> lab's `gpu_idle_fraction` is the overlapped ratio, 0.215686 before
> rounding.

The figure draws the same two bars. Move the schedule and the overlap bar
shrinks only while the schedule fits in the forward. Past that, the
leftover comes back, and overlap helps less.

```viz
16a-schedule-overlap
```

The prediction, and its limit: on a real GPU, a schedule shorter than the
forward disappears from the idle time, and sampling does not. Chapter 0's
42 ms lower bound says a short schedule is a small slice of *this* model's
decode step. The vLLM blog's about 5 ms, for Llama-8B on an H100, says the
same schedule can be the whole step on a faster device. This chapter does
not measure either device. The lab measures the simulated fraction, and it
checks that overlap lowers 0.444444 to 0.215686 for the worked ticks.

> [!DEEPDIVE] Two steps, written out as a timeline
> Take $n = 2$, $\hla{S} = 6$, $\hlb{F} = 10$, $\hlc{P} = 2$.
>
> Serial. Schedule 0 from tick 0 to 6. Forward 0 from 6 to 16. Post 0 from
> 16 to 18. Schedule 1 from 18 to 24. Forward 1 from 24 to 34. Post 1 from
> 34 to 36. Wall 36, busy 20, idle 16.
>
> Overlap. Schedule 0 still occupies 0 to 6. Forward 0 occupies 6 to 16,
> and schedule 1 runs inside it, so schedule 1 adds no wall. Post 0
> occupies 16 to 18. Forward 1 occupies 18 to 28. Post 1 occupies 28 to
> 30. Wall 30, busy 20, idle 10. The formula gives
> $6 + 2 \times 2 = 10$ idle ticks and $10 + 20 = 30$ wall ticks.
>
> The last forward does not hide another schedule. There is no step 2.

One optimism is built into the prefetch. The clock hides the next schedule
whenever any request in the batch still has token budget after this step.
A stop string can finish the request anyway, and the hidden schedule is
then unused. When $\hla{S} \le \hlb{F}$ that unused schedule adds no wall
ticks, so the graded trace, which does not stop early, matches the boxed
fraction exactly.

## The streaming response

This section answers how the safe text becomes bytes on the socket.

Each event the engine returns is a small record: the request id, the text
that became safe on this step, and a finish reason or null. The API process wraps
that record in a chat chunk. The lab's `sse_chunks` is that wrap, without
a socket.

A chunk is a JSON object with the same `id`, `created`, and `model` on
every record, and with `object` equal to `chat.completion.chunk`. The
choice carries a `delta` and a `finish_reason`. Content rides in
`delta.content`. The finish record has an empty delta and `finish_reason`
set to `stop` or `length`.

The fields the lab checks are the following:

| Field | On a content chunk | On the finish chunk | On the usage chunk |
|---|---|---|---|
| `object` | `chat.completion.chunk` | `chat.completion.chunk` | `chat.completion.chunk` |
| `choices` | one choice | one choice, empty delta | an empty array |
| `finish_reason` | null | `stop` or `length` | absent |
| `usage` | null, if usage was requested | null, if usage was requested | prompt, completion, and total |

Usage is optional. When the caller asks for it, the chunk that carries the
numbers has an empty `choices` array, and it is sent before the trailer
`data: [DONE]`. Every earlier chunk includes `usage` as null. The fields
the lab fills are `prompt_tokens`, `completion_tokens`, and `total_tokens`,
which add in the obvious way: prompt plus completion equals total.

The following stream is the lab's shape for `Say ` and `hi`, then a stop,
with a usage total of 5. Records are separated by a blank line:

```text
data: {"id":"chatcmpl-lab","object":"chat.completion.chunk","created":1,"model":"toy","choices":[{"index":0,"delta":{"content":"Say "},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-lab","object":"chat.completion.chunk","created":1,"model":"toy","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-lab","object":"chat.completion.chunk","created":1,"model":"toy","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}

data: {"id":"chatcmpl-lab","object":"chat.completion.chunk","created":1,"model":"toy","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}

data: [DONE]
```

A round trip parses each `data:` line, joins `delta.content`, reads the
finish reason, and reads the one non-null usage object. If the join is
not the text you meant to send, the bug is in the chunk builder, not in
the engine. The lab checks the schema and the round trip separately, so a
stream that is valid JSON and the wrong text fails the second check.

The API process is also where backpressure belongs. If the client reads
slowly, the process stops pulling from the core rather than buffering
without bound. The lab does not model that queue. It does require that a
cancel, which is what you do when the client is gone, frees the pool.

## What goes wrong

These are the failures the lab is built to catch. Each one still returns a
stream, which is why they are easy to miss.

**Sending the stop prefix early.** `EN` goes out on the step that sampled
it. The next token completes `END`, but the client already has `EN`. Hold
the suffix until the next call of `safe_prefix` resolves it.

**Treating the stop buffer as the UTF-8 buffer.** Chapter 3 holds an
incomplete character. This chapter holds an incomplete stop string. Run
the detokenizer first, then the stop hold. Skipping the first brings back
the replacement character. Skipping the second leaks the stop.

**Freeing every block id on the request.** A shared block goes back to
the pool while another request's count is still positive. Decrement the
count. Return the block only at zero. Return the recurrent slot on that
same release, including when the request is cancelled before its first
token.

**Overlapping the sample with the forward.** The logits do not exist yet,
so the post stage cannot move earlier. Overlap the *next* schedule. If
you also drop the post ticks from the clock, `gpu_idle_fraction` comes
out lower than the 22/102 the lab expects.

**Putting the usage numbers on every chunk, or forgetting the trailer.**
The spec puts the numbers on one chunk with empty `choices`, and ends the
stream with `data: [DONE]`. A parser that stops at the finish reason does
not see the usage, and a client that waits for `[DONE]` does not finish.

> [!RECAP]
> - The engine step is add, schedule, forward, sample, detokenize, stream.
>   The forward is the GPU stage. The stages on either side of it are CPU.
> - Hold a proper prefix of a stop string. Emit it when it is no longer a
>   prefix, and do not emit the stop. Flush the hold only when the request
>   ends for length.
> - Cancel and finish both release. A block returns when its reference
>   count hits zero, and the recurrent slot returns with the request.
> - Overlap the next schedule with the current forward. Sampling stays
>   after the logits. On the worked ticks the idle fraction falls from
>   4/9 to 11/51. The lab measures that simulated ratio.
> - A chat stream is `chat.completion.chunk` records, a finish reason, one
>   usage chunk with empty choices, and `data: [DONE]`.

## Check your understanding

> [!QUESTION] Why can the next schedule overlap the forward, while sampling cannot?
> The next schedule only needs to know which requests are still running.
> You can build it during the forward if you assume the current requests
> continue. Sampling reads the logits, and those logits do not exist until
> the forward returns, so the post stage stays after $\hlb{F}$.

> [!QUESTION] The stop string is `END`, and the pieces are `EN` then `D`. What does the client receive after the `EN` token?
> Nothing new. `EN` is a proper prefix of `END`, so it stays in the pending
> buffer. After `D` the stop matches, the stream finishes with reason
> `stop`, and `END` is not in the output.

> [!QUESTION] Two requests share a block. You cancel one. What happens to the free count?
> It rises by the cancelled request's private blocks only. The shared
> block's reference count drops by one and stays positive, so the block
> stays out of the free list. The cancelled request's recurrent slot
> returns.

> [!QUESTION] Chapter 0 puts a decode step near 42 ms, and the vLLM blog puts a Llama-8B step on an H100 near 5 ms. Which one does this lab measure?
> Neither. Both are someone else's figure, on named hardware or as a lower
> bound. The lab runs a simulated clock and reports `gpu_idle_fraction`.
> For schedule 6, forward 10, post 2, and 8 steps, overlap moves that
> fraction from 4/9 to 11/51.

## Lab

> [!TRY]
> Implement `safe_prefix`, `BlockPool`, `EngineCore`, and `sse_chunks` on
> CPU, against a fake model. You pass when the streamed text matches
> offline generation, a stop string that spans two tokens stays unsent
> until it resolves, cancel restores the pool, the chunk stream round-trips,
> and overlapping the schedule lowers `gpu_idle_fraction`.

The harness is CPU-only. The model is `step(ids) -> logits`. There are no
weights and no GPU. The clock's ticks are the numbers in the worked example.

`safe_prefix` is the first check, and it is the only function that check
calls. Given pending text `hiEN` and stop `END`, it returns the emit
string `hi`, the hold `EN`, and no match. A second check strips a completed
`END` and leaves ordinary text alone.

`EngineCore` then has to use that helper. Fifty requests, drawn with seed
16, must stream the same text as cutting the concatenated pieces at the
first stop. A separate check samples `EN` and then `D` with stop `END`,
and requires the `EN` step to emit the empty string. Emitting `EN` on
that step is the failure the check names.

`cancel` on the only live request must put `free` and `free_slots` back to
the pool's initial counts. A second request that shares a block must keep
that block: after the cancel, the free count is 6 on a pool of 8, not 7.

`sse_chunks` must emit `chat.completion.chunk` records, a `finish_reason`
of `stop`, one usage object on a chunk whose `choices` are empty, and the
`data: [DONE]` trailer. A parser joins the deltas and compares them with
the source text.

The load check runs one request for 8 tokens with schedule 6, forward 10,
and post 2. The serial `gpu_idle_fraction` is 0.444444. With overlap it is
0.215686, and the overlapped value has to be the smaller one. The metric
the lab records is the overlapped fraction.

## Further reading

- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the engine-core split, and the Llama-8B on H100 note of about 5 ms.
- [EngineCore in the vLLM API](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/core/) — `step` is documented as "Schedule, execute, and make output."
- [vLLM `EngineCore` source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py) — the batch queue that schedules a later batch while an earlier one executes, and the threads that overlap socket IO with the GPU.
- [SGLang `Engine`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/engine.py) — tokenizer, scheduler, and detokenizer as separate processes, connected with ZeroMQ.
- [TensorRT-LLM Executor API](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html) — `enqueueRequest`, `awaitResponses`, `cancelRequest`, and in-flight batching.
- [OpenAI chat completions](https://developers.openai.com/api/reference/resources/chat) — stop sequences are omitted from the returned text, and the streaming section shows `data: [DONE]`.
- [Chat completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) — `chat.completion.chunk`, `finish_reason`, and the usage chunk.
- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) — lecture 10, Inference, taught by Percy Liang.
