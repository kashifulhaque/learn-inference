#!/usr/bin/env python3
"""Synthesises every narration sentence before the renders need it.

A render asks the speech server for one narration block at a time, so the
GPU sees only a few requests at once. This script keeps the server full
instead: it reads the sentences that a `collect` pass recorded, skips those
already in the cache, and sends the rest with many requests in flight. The
renders that follow then find every sentence in the cache.

Usage, from the repository root:

    LECTURE_TTS=collect python3 lectures/build.py --all --frames --out DIR
    LECTURE_TTS_URL=... LECTURE_TTS_TOKEN=... \\
        python3 lectures/prefetch.py --from DIR --concurrency 32
    python3 lectures/build.py --all --quality h --out DIR

`--from` is the build output directory of the `collect` pass; the script
reads every `narration.jsonl` under it. Set `LECTURE_TTS_CACHE` to the same
cache directory that the renders use.

The script needs only the standard library and NumPy; it doesn't load manim.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Load kit.narration without kit/__init__.py, which imports manim.
_pkg = types.ModuleType("kit")
_pkg.__path__ = [str(HERE / "kit")]
sys.modules.setdefault("kit", _pkg)
from kit import narration  # noqa: E402


def collected(root: Path) -> list[str]:
    """Every distinct spoken sentence under `root`, in first-seen order."""
    seen: dict[str, None] = {}
    for path in sorted(root.rglob("narration.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.setdefault(json.loads(line)["spoken"], None)
    return list(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from", dest="src", type=Path, required=True,
                        help="output directory of a LECTURE_TTS=collect pass")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="requests in flight (default 32, the server's batch size)")
    opts = parser.parse_args()

    if narration.mode() != "remote":
        parser.error("unset LECTURE_TTS, or set it to remote; prefetch needs the speech server.")
    voice = narration.Voice()
    texts = collected(opts.src)
    misses = [t for t in texts if not voice._cache_path(t).exists()]
    print(f"{len(texts)} sentences collected, {len(texts) - len(misses)} cached, "
          f"{len(misses)} to synthesise into {voice.cache_dir}", flush=True)

    failed: list[tuple[str, str]] = []
    done = 0
    audio = 0.0
    start = time.monotonic()

    def one(text: str):
        pcm = voice._synthesize(text)
        narration._write_wav(voice._cache_path(text), pcm)
        return len(pcm) / narration.SAMPLE_RATE

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, opts.concurrency)) as pool:
        futures = {pool.submit(one, t): t for t in misses}
        for fut in concurrent.futures.as_completed(futures):
            try:
                audio += fut.result()
                done += 1
            except Exception as err:  # keep going; report every failure at the end
                failed.append((futures[fut], str(err)))
            n = done + len(failed)
            if n % 50 == 0 or n == len(misses):
                wall = time.monotonic() - start
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(misses)} done, {audio / 60:.1f} min of audio "
                      f"in {wall / 60:.1f} min ({audio / max(wall, 1e-9):.1f}x real time), "
                      f"{len(failed)} failed", flush=True)

    for text, err in failed:
        print(f"FAILED: {text[:80]!r}: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
