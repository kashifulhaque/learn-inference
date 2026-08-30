#!/usr/bin/env python3
"""Runs every lab's reference solution against its own tests.

A lab whose solution fails its tests is broken, so this is the guard that runs
before anything is deployed. Labs that need a GPU are skipped on a CPU-only
machine and reported as skipped, not passed.

Usage:
    python3 scripts/check_labs.py [LAB_ID ...]
"""

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "content" / "labs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LABS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    wanted = set(argv)
    passed, failed, skipped = [], [], []

    for lab_dir in sorted(p for p in LABS.iterdir() if (p / "meta.json").is_file()):
        meta = json.loads((lab_dir / "meta.json").read_text())
        lab_id = meta["id"]
        if wanted and lab_id not in wanted:
            continue
        # Some GPU labs check mathematics that runs anywhere; their tests call
        # pick_device and are marked device_agnostic so they still run on CPU.
        if meta.get("needs_gpu") and not has_cuda and not meta.get("device_agnostic"):
            skipped.append(lab_id)
            print(f"SKIP  {lab_id}  (needs a GPU)")
            continue

        os.environ["LI_LAB_ID"] = lab_id
        print(f"\n=== {lab_id} ===")
        try:
            solution = load(f"sol_{lab_id.replace('-', '_')}", lab_dir / "solution.py")
            tests = load(f"tst_{lab_id.replace('-', '_')}", lab_dir / "tests.py")
            outcome = tests.run(solution)
        except Exception:
            traceback.print_exc()
            failed.append(lab_id)
            continue

        (passed if outcome.get("passed") else failed).append(lab_id)

    print("\n" + "=" * 60)
    print(f"passed  {len(passed)}: {', '.join(passed) or '-'}")
    print(f"failed  {len(failed)}: {', '.join(failed) or '-'}")
    print(f"skipped {len(skipped)}: {', '.join(skipped) or '-'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
