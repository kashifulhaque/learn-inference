"""Step times with a launch floor, for chapter 10 and its lab.

Every number comes from the lab's reference solution, so the chapter and the
lab cannot drift apart. The launch cost is chapter 10's 5 µs. This script does
not measure a device.

Run from the repository root:

    python3 scripts/figures/10-roofline.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "solution", ROOT / "content" / "labs" / "10-roofline" / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

NORM_BYTES = 2 * 5120 * 2
NORMS, WEIGHT_KERNELS, WEIGHT_BYTES = 129, 271, 53.8e9

norm = (float(NORM_BYTES), float(NORM_BYTES))
weight = (WEIGHT_BYTES / WEIGHT_KERNELS,) * 2
step = [norm] * NORMS + [weight] * WEIGHT_KERNELS

eager = sol.step_time_ms(step)
graph = sol.step_time_ms(step, graph=True)
print(f"norm floor            {NORM_BYTES / 1935e9 * 1e9:.1f} ns")
print(f"weight kernel floor   {weight[1] / 1935e9 * 1e3:.4f} ms")
print(f"eager                 {eager:.4f} ms")
print(f"graph                 {graph:.4f} ms")
print(f"saved                 {eager - graph:.4f} ms ({(eager - graph) / eager:.1%} of eager)")
print(f"class                 {sol.classify_step(step)}")
print(f"400 tiny, eager       {sol.step_time_ms([norm] * 400):.4f} ms, "
      f"{sol.classify_step([norm] * 400)}")

mlp = sol.mlp_cost(1, 5120, 17408)
mixed = [norm] * 3 + [mlp]
one_max = max(sol.launch_floor_ms(4), sum(sol.roofline(f, b)["floor_ms"] for f, b in mixed))
print(f"3 tiny + MLP, per kernel {sol.step_time_ms(mixed):.4f} ms, one max {one_max:.4f} ms")
