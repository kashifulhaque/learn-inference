"""Numbers for chapter 21.

Every step time is chapter 17a's floor for the 27B hybrid model, applied to the
batches that the tiny engine in content/labs/21-capstone/solution.py actually
scheduled. Nothing here is an A100 measurement, and nothing here ran real
weights. The tokens come from a seeded tiny model on the CPU.

Run from the repository root:

    python3 scripts/figures/21-capstone-the-engine.py
    python3 scripts/figures/21-capstone-the-engine.py --ts   # data for the figure
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LABS = ROOT / "content" / "labs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LABS))
sys.path.insert(0, str(LABS / "21-capstone"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sol = load("capstone_solution", LABS / "21-capstone" / "solution.py")
    tests = load("capstone_tests", LABS / "21-capstone" / "tests.py")
    tm = load("perf_17a", LABS / "17a-performance-model" / "solution.py")
    hw = sol.A100
    cfg17 = {"kind": "hybrid"}
    model = sol.build_tiny_model(tests.SEED_MODEL)
    cfg = model.config

    if "--ts" in sys.argv:
        trace = tests.make_trace()
        engine = sol.Engine(model, num_blocks=48, block_size=4, max_batch_size=16,
                            max_batched_tokens=48, chunk_size=16)
        result = sol.run_trace(engine, trace, tm, hw)
        rows = [[round(r["dt"] * 1000, 3), r["decode"], r["prefill"], r["preempted"],
                 round(r["t"] * 1000, 3)] for r in result["log"]]
        print(json.dumps(rows, separators=(",", ":")))
        return

    print("tiny model:", cfg.num_hidden_layers, "layers;",
          "full attention", cfg.full_attention_layers, "; linear", cfg.linear_attention_layers)
    print("tiny state per sequence:", cfg.recurrent_state_bytes_per_sequence(), "bytes (config arithmetic)")

    print("\n-- step times, 17a floors, 27B hybrid, one A100 --")
    print("weight read alone: %.3f ms" % (tm._weight_bytes("hybrid") / (hw["hbm_gbs"] * 1e9) * 1e3))
    print("decode batch 1, ctx 1:   %.3f ms" % (tm.decode_step_time(cfg17, hw, 1, 1) * 1e3))
    print("decode batch 16, ctx 24: %.3f ms" % (tm.decode_step_time(cfg17, hw, 16, 24) * 1e3))
    print("prefill 64 tokens:       %.3f ms" % (tm.prefill_time(cfg17, hw, 64) * 1e3))
    mixed = sol.step_seconds(tm, hw, 16, 16 * 24, [16])
    apart = tm.decode_step_time(cfg17, hw, 16, 24) + tm.prefill_time(cfg17, hw, 16)
    print("16 decodes + one 16-token chunk, one step: %.3f ms; as two passes: %.3f ms" % (mixed * 1e3, apart * 1e3))

    print("\n-- the trace: 64 requests, seed", tests.SEED_TRACE, "--")
    trace = tests.make_trace()
    engine = sol.Engine(model, num_blocks=48, block_size=4, max_batch_size=16,
                        max_batched_tokens=48, chunk_size=16)
    result = sol.run_trace(engine, trace, tm, hw)
    records = result["records"]
    tokens = sum(r["max_new"] for r in trace)
    serial = sum(
        tm.prefill_time(cfg17, hw, len(r["prompt"]))
        + sum(tm.decode_step_time(cfg17, hw, 1, len(r["prompt"]) + j) for j in range(1, r["max_new"]))
        for r in trace
    )
    value = sol.goodput(records, result["makespan"], tests.SLO_TTFT, tests.SLO_TPOT)
    ttft = sorted(r["first"] - r["arrival"] for r in records.values())
    tpot = sorted((r["finish"] - r["first"]) / (r["tokens"] - 1) for r in records.values())
    print("output tokens:", tokens)
    print("serial, one request at a time: %.3f s, %.2f tokens/s" % (serial, tokens / serial))
    print("floor (%.1fx serial): %.2f tokens/s" % (tests.GOODPUT_FACTOR, tests.GOODPUT_FACTOR * tokens / serial))
    print("engine: makespan %.4f s, %d steps, %d preemptions, goodput %.2f tokens/s"
          % (result["makespan"], result["steps"], result["preemptions"], value))
    print("TTFT median %.3f s, max %.3f s; TPOT median %.4f s, max %.4f s"
          % (ttft[32], ttft[-1], tpot[32], tpot[-1]))
    print("pools after the run: %d of 48 blocks free, %d of 16 slots free"
          % (engine.free_blocks, engine.free_slots))

    print("\n-- the chunk-boundary bug, tiny model, 8 prompts --")
    prompts = tests.make_prompts()
    refs = [tests.reference_generate(model, p, tests.NEW_TOKENS) for p in prompts]
    for impl in ("naive", "sdpa"):
        out = sol.Engine(model, num_blocks=200, chunk_size=5, max_batched_tokens=12,
                         attn_impl=impl).generate(prompts, tests.NEW_TOKENS)
        print("attn_impl=%-5s chunk 5: %d of 8 prompts match the no-cache reference"
              % (impl, sum(o == r for o, r in zip(out, refs))))

    print("\n-- preemption, 16 blocks, batch 4, chunk 5 --")
    e = sol.Engine(model, num_blocks=16, block_size=4, max_batch_size=4, chunk_size=5,
                   max_batched_tokens=12)
    out = e.generate(prompts, tests.NEW_TOKENS)
    print("preemptions %d; %d of 8 match; free blocks %d of 16; free slots %d of 4"
          % (e.preemptions, sum(o == r for o, r in zip(out, refs)), e.free_blocks, e.free_slots))


if __name__ == "__main__":
    main()
