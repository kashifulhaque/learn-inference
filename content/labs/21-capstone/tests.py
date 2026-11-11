"""Checks for the capstone engine: pools, a paged sequence view, the whole loop,
preemption, seeded sampling, and a goodput trace on a simulated clock.

Every random choice uses seed 21 for the model and 7, 13, and 31 for prompts,
sampling, and the trace. The model is the repository's hybrid stack at toy
size: 8 layers (full attention at layers 3 and 7), hidden size 48, vocabulary
128, float32, on the CPU. The reference is a no-cache forward pass over the
whole sequence for every new token.

Plausible wrong implementations, each of which the checks below catch:

* A recompute restart that keeps the victim's old recurrent state. Freed slots
  are handed out without zeroing, so a restarted or newly admitted request
  starts from someone else's state. The full-attention cache is rebuilt from
  scratch, so only the 6 linear layers are wrong and the output drifts without
  an error. Caught by "outputs survive preemption by recompute".
* Prefill chunks run through `attn_impl="sdpa"`. PyTorch's is_causal aligns its
  mask to the top-left corner, so a chunk that follows a cached prefix cannot
  see the prefix. Caught by "chunked prefill and interleaving leave outputs
  unchanged".
* One sampler generator shared by all requests, so a request's draws depend on
  who else is in the batch. Caught by "seeded sampling is per request".
* Preemption frees the blocks but not the state slot. The slots run out.
  Caught by the preemption check and the pools check.

The goodput check is a correctness gate on the simulated clock, not a
performance threshold. The step times are chapter 17a's floors for the 27B
model, not measurements, and the tiny model only supplies the tokens.
"""

import importlib.util
import os
import random
from pathlib import Path

import torch

from lab_common import Checks

SEED_MODEL = 21
SEED_PROMPTS = 7
SEED_SAMPLE = 13
SEED_TRACE = 31
NEW_TOKENS = 10
SLO_TTFT = 1.0   # seconds
SLO_TPOT = 0.15  # seconds
GOODPUT_FACTOR = 2.0


def _load_17a():
    path = Path(__file__).resolve().parent.parent / "17a-performance-model" / "solution.py"
    spec = importlib.util.spec_from_file_location("perf_17a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_generate(model, prompt, n):
    """Greedy decoding with no cache: one full forward pass per new token."""
    ids = torch.tensor([prompt])
    out = []
    with torch.no_grad():
        for _ in range(n):
            token = int(model(ids)[0, -1].argmax())
            out.append(token)
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return out


def make_prompts(count=8):
    rng = random.Random(SEED_PROMPTS)
    return [[rng.randrange(1, 128) for _ in range(rng.randint(6, 30))] for _ in range(count)]


def make_trace(count=64, rate=40.0):
    rng = random.Random(SEED_TRACE)
    now = 0.0
    rows = []
    for _ in range(count):
        now += rng.expovariate(rate)
        rows.append({
            "arrival": now,
            "prompt": [rng.randrange(1, 128) for _ in range(rng.randint(6, 30))],
            "max_new": rng.randint(4, 12),
        })
    return rows


def _first_diff(actual, expected):
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            return f"prompt {i}: expected {e}, got {a}"
    return None


def run(submission):
    from engine.sampling import SamplingParams

    c = Checks()
    if not c.require(submission, "StatePool", "SeqCache", "Engine", "step_seconds",
                     "run_trace", "goodput", "build_tiny_model"):
        return c.finish()

    model = submission.build_tiny_model(SEED_MODEL)
    cfg = model.config
    prompts = make_prompts()
    refs = [reference_generate(model, p, NEW_TOKENS) for p in prompts]

    # 1. The smallest piece: the state pool.
    def check_pool():
        pool = submission.StatePool(cfg, 3, "cpu", torch.float32)
        slot = pool.alloc("a")
        layer = cfg.linear_attention_layers[0]
        pool.states[layer][slot].fill_(5.0)
        pool.convs[layer][slot].fill_(5.0)
        if pool.free_slots != 2:
            return False, f"expected 2 free slots after one alloc, got {pool.free_slots}"
        pool.release("a")
        pool.release("a")
        if pool.free_slots != 3:
            return False, f"expected 3 free slots after release (twice), got {pool.free_slots}"
        again = pool.alloc("b")
        leftover = pool.states[layer][again].abs().sum().item() + pool.convs[layer][again].abs().sum().item()
        return leftover == 0.0, f"expected a zeroed slot on reuse, got sum {leftover}"

    c.check("a freed state slot comes back zeroed", check_pool)

    # 2. The sequence view: paged KV plus one slot reproduces a full forward pass.
    def check_seqcache():
        ids = torch.tensor([prompts[0][:12] + [3, 5, 9]])
        with torch.no_grad():
            full = model(ids)
        paged = submission.PagedKVCache(cfg, 12, 4, device="cpu", dtype=torch.float32)
        pool = submission.StatePool(cfg, 2, "cpu", torch.float32)
        pool.alloc("s")
        paged.allocate("s", 15)
        seq = submission.SeqCache(paged, pool, "s")
        worst = 0.0
        with torch.no_grad():
            for lo, hi in ((0, 5), (5, 11), (11, 12), (12, 13), (13, 14), (14, 15)):
                out = model(ids[:, lo:hi], cache=seq, attn_impl="naive")
                worst = max(worst, (out[0] - full[0, lo:hi]).abs().max().item())
        if seq.length != 15:
            return False, f"expected length 15 after the passes, got {seq.length}"
        return worst < 1e-4, f"expected max abs error under 1e-4, got {worst:.2e}"

    c.check("a paged sequence view reproduces the full forward pass", check_seqcache)

    # 3. The whole engine, greedy, no pressure.
    def check_greedy():
        engine = submission.Engine(model, num_blocks=200)
        got = engine.generate(prompts, NEW_TOKENS)
        diff = _first_diff(got, refs)
        return diff is None, diff or "expected 8 of 8 prompts to match the no-cache reference"

    c.check("greedy output equals the no-cache reference for 8 seeded prompts", check_greedy)

    # 4. Interleaved and chunked. Chunk 5 does not divide any prompt evenly.
    def check_chunked():
        engine = submission.Engine(model, num_blocks=200, chunk_size=5, max_batched_tokens=12)
        got = engine.generate(prompts, NEW_TOKENS)
        diff = _first_diff(got, refs)
        return diff is None, diff or "expected 8 of 8 to match with chunk 5 and a 12-token budget"

    c.check("chunked prefill and interleaving leave outputs unchanged", check_chunked)

    # 5 and 6. Preemption by recompute under a 16-block budget, with an audit
    # after every step, then the pools.
    state = {}

    def check_preempt():
        engine = submission.Engine(model, num_blocks=16, block_size=4, max_batch_size=4,
                                   chunk_size=5, max_batched_tokens=12)
        requests = [engine.add_request(p, NEW_TOKENS) for p in prompts]
        problems = []
        steps = 0
        while engine.has_work and steps < 400:
            engine.step()
            steps += 1
            problems += engine.audit()
        state["engine"] = engine
        if engine.has_work:
            return False, f"expected the run to finish within 400 steps, still working at {steps}"
        if problems:
            return False, f"expected an empty audit after every step, got {problems[0]}"
        if engine.preemptions < 1:
            return False, f"expected at least 1 preemption under 16 blocks, got {engine.preemptions}"
        diff = _first_diff([list(r.output_ids) for r in requests], refs)
        c.metric("preemptions", engine.preemptions)
        return diff is None, diff or f"expected 8 of 8 to match after {engine.preemptions} preemptions"

    c.check("outputs survive preemption by recompute", check_preempt)

    def check_pools():
        engine = state.get("engine")
        if engine is None:
            return False, "expected the preemption run to exist, got none"
        free = engine.paged.free
        ok = (
            engine.free_blocks == 16 and len(set(free)) == len(free) == 16
            and engine.free_slots == 4 and not engine.seqs and not engine.generators
            and not engine.audit()
        )
        return ok, (
            f"expected 16 unique free blocks, 4 free slots, no views, got "
            f"{engine.free_blocks} blocks ({len(set(free))} unique), {engine.free_slots} slots, "
            f"{len(engine.seqs)} views"
        )

    c.check("blocks and state slots all return to the pool", check_pools)

    # 7. Seeded sampling: one generator per request.
    def sampled(engine_kwargs, ids):
        params = [SamplingParams(temperature=0.9, top_k=20, seed=SEED_SAMPLE + i) for i in ids]
        engine = submission.Engine(model, **engine_kwargs)
        return engine.generate([prompts[i] for i in ids], NEW_TOKENS, params), engine

    def check_sampling():
        solo = []
        for i in range(4):
            out, _ = sampled(dict(num_blocks=200), [i])
            solo.append(out[0])
        together, _ = sampled(dict(num_blocks=200, chunk_size=5, max_batched_tokens=12), [0, 1, 2, 3])
        again, _ = sampled(dict(num_blocks=200, chunk_size=5, max_batched_tokens=12), [0, 1, 2, 3])
        squeezed, eng = sampled(dict(num_blocks=16, max_batch_size=4, chunk_size=5,
                                     max_batched_tokens=12), [0, 1, 2, 3])
        if together != again:
            return False, "expected two runs with the same seeds to match, got different outputs"
        if together != solo:
            return False, f"expected batched output equal to solo output, got {_first_diff(together, solo)}"
        if squeezed != solo:
            return False, (f"expected output unchanged after {eng.preemptions} preemptions, "
                           f"got {_first_diff(squeezed, solo)}")
        other = submission.Engine(model, num_blocks=200).generate(
            [prompts[0]], NEW_TOKENS, [SamplingParams(temperature=0.9, top_k=20, seed=SEED_SAMPLE + 99)])
        if other[0] == solo[0]:
            return False, "expected a different seed to change the output, got the same tokens"
        return True, f"expected solo = batched = preempted ({eng.preemptions} preemptions), got the same"

    c.check("seeded sampling is per request and reproducible", check_sampling)

    # 8. The step-time model, wired in.
    tm = _load_17a()
    hw = submission.A100 if hasattr(submission, "A100") else None
    cfg17 = {"kind": "hybrid"}

    def check_step_seconds():
        decode = submission.step_seconds(tm, hw, 8, 8 * 1024, [])
        want_decode = tm.decode_step_time(cfg17, hw, 8, 1024)
        lone = submission.step_seconds(tm, hw, 0, 0, [64])
        want_lone = tm.prefill_time(cfg17, hw, 64)
        mixed = submission.step_seconds(tm, hw, 8, 8 * 1024, [64])
        separate = want_decode + want_lone
        if abs(decode - want_decode) > 1e-9:
            return False, f"expected decode step {want_decode:.9f} s, got {decode:.9f} s"
        if abs(lone - want_lone) > 1e-9:
            return False, f"expected lone chunk {want_lone:.9f} s, got {lone:.9f} s"
        return mixed < separate, (
            f"expected a mixed step under the two passes ({separate:.6f} s), got {mixed:.6f} s"
        )

    c.check("a step costs 17a's floor, with one weight read per step", check_step_seconds)

    # 9. Goodput on a seeded 64-request trace, against a serial floor.
    def check_trace():
        trace = make_trace()
        engine = submission.Engine(model, num_blocks=48, block_size=4, max_batch_size=16,
                                   max_batched_tokens=48, chunk_size=16)
        result = submission.run_trace(engine, trace, tm, hw)
        records = result["records"]
        unfinished = [r for r in records.values() if r["finish"] is None]
        if len(records) != 64 or unfinished:
            return False, f"expected 64 finished requests, got {len(records) - len(unfinished)}"
        if engine.free_blocks != 48 or engine.free_slots != 16 or engine.audit():
            return False, (f"expected 48 blocks and 16 slots back, got {engine.free_blocks} "
                           f"and {engine.free_slots}")
        serial_seconds = sum(
            tm.prefill_time(cfg17, hw, len(row["prompt"]))
            + sum(tm.decode_step_time(cfg17, hw, 1, len(row["prompt"]) + j)
                  for j in range(1, row["max_new"]))
            for row in trace
        )
        tokens = sum(row["max_new"] for row in trace)
        floor = GOODPUT_FACTOR * tokens / serial_seconds
        value = submission.goodput(records, result["makespan"], SLO_TTFT, SLO_TPOT)
        c.metric("goodput_tokens_per_s", round(value, 2))
        c.metric("serial_floor_tokens_per_s", round(floor, 2))
        c.metric("trace_preemptions", result["preemptions"])
        c.metric("trace_steps", result["steps"])
        return value >= floor, (
            f"expected goodput of at least {floor:.2f} tokens/s "
            f"({GOODPUT_FACTOR}x one-at-a-time), got {value:.2f}"
        )

    c.check("goodput on the 64-request trace beats the serial floor", check_trace)

    _real_weights(submission, c)
    return c.finish()


def _real_weights(submission, c):
    """Optional: greedy output against Hugging Face transformers on real weights.

    Runs only with a CUDA device and a local checkpoint. NOT VERIFIED: the
    course's GPU endpoint was paused when this lab was written, so this code
    has never run. On a CPU it records a metric and adds no check.
    """
    if not torch.cuda.is_available():
        c.metric("real_weights_check", "skipped: no CUDA device")
        return
    if not hasattr(submission, "load_hf_weights"):
        c.metric("real_weights_check", "skipped: no load_hf_weights in the submission")
        return
    from engine.config import ModelConfig
    from engine.weights import local_model_path

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
    try:
        path = local_model_path(model_id)
    except Exception as exc:  # noqa: BLE001
        c.metric("real_weights_check", f"skipped: {type(exc).__name__}: {exc}")
        return
    config = ModelConfig.from_path(path)
    if config.hidden_size > 2048:
        c.metric("real_weights_check", f"skipped: {model_id} is too large for float32; set MODEL_ID to a small sibling")
        return

    def check():
        import transformers
        from engine.model import HybridLanguageModel

        tok = transformers.AutoTokenizer.from_pretrained(path)
        try:
            hf = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        except Exception:  # noqa: BLE001
            hf = transformers.AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.float32)
        hf = hf.cuda().eval()
        mine = HybridLanguageModel(config).cuda().eval()
        submission.load_hf_weights(mine, path)
        texts = ["The capital of France is", "def fibonacci(n):", "Once upon a time"]
        ids = [tok(t).input_ids for t in texts]
        engine = submission.Engine(mine, num_blocks=64, block_size=16, chunk_size=8)
        got = engine.generate(ids, 8)
        for text, prompt, tokens in zip(texts, ids, got):
            with torch.no_grad():
                out = hf.generate(torch.tensor([prompt]).cuda(), max_new_tokens=8, do_sample=False)
            expected = out[0, len(prompt):].tolist()
            if tokens != expected:
                return False, f"prompt {text!r}: expected {expected}, got {tokens}"
        return True, f"expected 3 of 3 prompts to match transformers on {model_id}"

    c.check(f"real weights ({model_id}): greedy output equals transformers", check)
