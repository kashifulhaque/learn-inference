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
import io
import json
import sys
import threading
import time
import types
import urllib.parse
import urllib.request
import wave
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


def request(voice, text: str, fresh: bool) -> tuple[float, dict]:
    """Synthesises `text` into the cache and returns its length and QA headers.

    `fresh` asks the server to skip its own cache and generate again.
    """
    cfg = voice._remote
    body = json.dumps({"model": narration.MODEL_ID, "input": text, "voice": narration.VOICE,
                       "speed": narration.SPEED, "response_format": "wav", "lang": narration.LANG}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "lecture-kit-prefetch/1"}
    if cfg["token"]:
        headers["Authorization"] = f"Bearer {cfg['token']}"
    if fresh:
        headers["X-TTS-No-Cache"] = "1"
    req = urllib.request.Request(f"{cfg['base_url']}{cfg['path']}", data=body, headers=headers)
    last: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data, h = resp.read(), resp.headers
            break
        except Exception as err:  # retried; the last error is raised
            last = err
    else:
        raise RuntimeError(str(last))
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getframerate() != narration.SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise RuntimeError("unexpected WAV layout from the speech server")
        frames = w.getnframes()
        pcm = narration.np.frombuffer(w.readframes(frames), dtype=narration.np.int16)
    narration._write_wav(voice._cache_path(text), pcm)
    qa = {"wer": float(h.get("X-TTS-WER") or 0), "tries": int(h.get("X-TTS-Tries") or 1),
          "flags": h.get("X-TTS-Flags") or "",
          "heard": urllib.parse.unquote(h.get("X-TTS-Transcript") or "")}
    return frames / narration.SAMPLE_RATE, qa


def load_qa(path: Path) -> dict[str, dict]:
    """The latest QA record for each spoken sentence."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["spoken"]] = rec
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from", dest="src", type=Path, required=True,
                        help="output directory of a LECTURE_TTS=collect pass")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="requests in flight (default 32, the server's batch size)")
    parser.add_argument("--redo-wer", type=float, default=None, metavar="X",
                        help="also synthesize again, fresh, every cached sentence whose last "
                             "word error rate was above X (for example 0.15)")
    parser.add_argument("--report", action="store_true",
                        help="list the sentences whose last word error rate is above --redo-wer "
                             "(or 0.15), and synthesize nothing")
    opts = parser.parse_args()

    if narration.mode() != "remote":
        parser.error("unset LECTURE_TTS, or set it to remote; prefetch needs the speech server.")
    voice = narration.Voice()
    qa_path = voice.cache_dir / "qa.jsonl"
    qa = load_qa(qa_path)
    texts = collected(opts.src)
    limit = opts.redo_wer if opts.redo_wer is not None else 0.15
    if opts.report:
        bad = [(qa[t]["wer"], t, qa[t]["heard"]) for t in texts if t in qa and qa[t]["wer"] > limit]
        for wer, text, heard in sorted(bad, reverse=True):
            print(f"{wer:.2f}  said:  {text}\n       heard: {heard}")
        unknown = sum(1 for t in texts if t not in qa)
        print(f"{len(bad)} of {len(texts)} sentences above {limit}; {unknown} have no QA record")
        return 0
    misses = [t for t in texts if not voice._cache_path(t).exists()]
    redo = []
    if opts.redo_wer is not None:
        redo = [t for t in texts if t not in misses and t in qa and qa[t]["wer"] > opts.redo_wer]
    print(f"{len(texts)} sentences collected, {len(texts) - len(misses)} cached, "
          f"{len(misses)} to synthesise and {len(redo)} to redo into {voice.cache_dir}", flush=True)
    fresh = set(redo)
    misses = misses + redo

    failed: list[tuple[str, str]] = []
    done = 0
    audio = 0.0
    start = time.monotonic()

    qa_lock = threading.Lock()

    def one(text: str):
        seconds, record = request(voice, text, fresh=text in fresh)
        with qa_lock, qa_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"spoken": text, **record}, ensure_ascii=False) + "\n")
        return seconds

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
