"""Narration: remote speech synthesis, a cache, and caption cues.

A narration block is split into sentences. Each sentence is synthesised on
its own, so the kit knows exactly when every sentence starts and ends. That
gives captions that match the voice to the frame, and it lets a scene start
an animation on a given sentence.

The voice runs on a remote server with an OpenAI-compatible
`/v1/audio/speech` endpoint (VibeVoice-1.5B for this course). The kit
configures it from the environment:

- `LECTURE_TTS_URL`: the server's base URL. Required on a cache miss.
- `LECTURE_TTS_TOKEN`: a bearer token, if the server needs one. Never
  write it into a file in the repository.
- `LECTURE_TTS_PATH`: the endpoint path. Defaults to `/v1/audio/speech`.
- `LECTURE_TTS_WORKERS`: concurrent requests on a cache miss. Defaults to 4.
- `LECTURE_TTS_CACHE`: the cache directory. Defaults to
  `lectures/.cache/tts/`.
- `LECTURE_TTS=estimate`: no network at all. Each sentence becomes silence
  lasting its word count divided by `estimate_words_per_second` from
  `course.json`, so layout previews run on a laptop. Estimated audio goes to
  its own directory and never into the real cache.

The model, voice, speed, language, and sample rate come from the `voice`
block of `lectures/course.json`, and every cache key includes them, so
changing the voice resynthesises every lecture on its next render.

Caption text and spoken text can differ. Captions keep symbols such as `σ²`
and `KV`; the voice reads them through `PRONUNCIATIONS` below, then the
course's `lectures/pronunciations.json`, or through the `say=` override on
`Lecture.narrate`.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .course import LECTURES
from .course import config as course_config

_VOICE = course_config()["voice"]
MODEL_ID = str(_VOICE["model"])
VOICE = str(_VOICE["voice"])
SPEED = float(_VOICE.get("speed", 1.0))
LANG = str(_VOICE.get("lang", "en"))
SAMPLE_RATE = int(_VOICE.get("sample_rate", 24000))
WORDS_PER_SECOND = float(course_config()["estimate_words_per_second"])

# Silence between sentences inside one narration block, in seconds.
SENTENCE_GAP = 0.3

# Longest caption cue, in characters. Two lines of about 42.
CUE_MAX_CHARS = 84

CACHE_DIR = Path(os.environ.get("LECTURE_TTS_CACHE", LECTURES / ".cache" / "tts"))
ESTIMATE_DIR = Path(os.environ.get("LECTURE_TTS_ESTIMATE_CACHE", LECTURES / ".cache" / "tts-estimate"))
COURSE_PRONUNCIATIONS = LECTURES / "pronunciations.json"


def mode() -> str:
    """Returns `estimate`, `collect`, or `remote`, from the `LECTURE_TTS` variable.

    `collect` behaves like `estimate` and also records every spoken sentence
    in `$LECTURE_OUT/narration.jsonl`, for `prefetch.py` to synthesise in bulk.
    """
    value = os.environ.get("LECTURE_TTS", "remote").strip().lower() or "remote"
    if value not in ("estimate", "collect", "remote"):
        raise ValueError(f"LECTURE_TTS must be 'estimate', 'collect', or 'remote', not {value!r}.")
    return value


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵥ", "0123456789aeoxhklmnpstijv")

# Spoken forms that every course shares. They apply in order to the text sent
# to the synthesiser only; captions keep the original. Patterns are regular
# expressions. VibeVoice is a language-model voice that reads decimals and
# most words correctly, so this list only covers symbols.
PRONUNCIATIONS: list[tuple[str, object]] = [
    # Time units after a number, before μ becomes a word.
    (r"(\d)\s*[µμ]s\b", r"\1 microseconds"),
    (r"(\d)\s*ms\b", r"\1 milliseconds"),
    (r"(\d)\s*ns\b", r"\1 nanoseconds"),
    # Ordinals on a letter: "k-th" would be spelt out as "k t h".
    (r"\b([a-z])-th\b", r"\1'th"),
    # Letters with a combining macron or hat: x̄, ŷ.
    (r"([A-Za-z])̄", r"\1 bar"),
    (r"([A-Za-z])̂", r"\1 hat"),
    (r"χ²", "chi-square"),
    (r"([A-Za-z0-9σμ)])²", r"\1 squared"),
    (r"²", " squared"),
    (r"([A-Za-z0-9)])³", r"\1 cubed"),
    (r"³", " cubed"),
    (r"χ", "kai"),
    (r"σ", "sigma"),
    (r"μ", "mu"),
    (r"θ", "theta"),
    (r"λ", "lambda"),
    (r"α", "alpha"),
    (r"β", "beta"),
    (r"γ", "gamma"),
    (r"δ", "delta"),
    (r"ε", "epsilon"),
    (r"τ", "tau"),
    (r"π", "pi"),
    (r"Φ|φ", "phi"),
    (r"Σ", "the sum of"),
    (r"√\s*([A-Za-z0-9]+)", r"root \1"),
    (r"≈", " about "),
    (r"≤", " at most "),
    (r"≥", " at least "),
    (r"≠", " is not "),
    (r"±", " plus or minus "),
    (r"(\d)\s*×\s*(\d)", r"\1 times \2"),
    (r"(\d)×", r"\1 times"),
    (r"×", " times "),
    (r"÷", " divided by "),
    (r"∞", " infinity "),
    (r"→", " to "),
    (r"←", " from "),
    (r"=", " equals "),
    # A letter, a minus sign, a digit: n − 1, n-1.
    (r"\b([A-Za-z])\s*[−-]\s*(\d+)\b", r"\1 minus \2"),
    (r"−", " minus "),
    # Subscripts read as plain characters after a space: dₕ is "d h".
    (r"[₀-₉ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼᵥ]+", lambda m: " " + m.group(0).translate(_SUBSCRIPTS)),
]


def course_pronunciations(path: Path = COURSE_PRONUNCIATIONS) -> list[tuple[str, str]]:
    """Loads the course's own rules: a JSON list of `[regex, replacement]`.

    They apply after the shared `PRONUNCIATIONS`, in file order. Use Python
    `re` syntax, with `\\1` for groups.
    """
    if not path.exists():
        return []
    rules = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, rule in enumerate(rules):
        if not (isinstance(rule, list) and len(rule) == 2 and all(isinstance(x, str) for x in rule)):
            raise ValueError(f"{path}: rule {i} must be a [regex, replacement] pair of strings.")
        re.compile(rule[0])
        out.append((rule[0], rule[1]))
    return out


RULES = PRONUNCIATIONS + course_pronunciations()

# A sentence ends at . ! or ? followed by a space and a capital letter, a
# digit, or a quotation, so "e.g. the cache" and "0.0228" stay whole.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def split_sentences(text: str) -> list[str]:
    """Splits narration into sentences, collapsing whitespace."""
    text = " ".join(text.split())
    return [s for s in _SENTENCE_END.split(text) if s]


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}


def number_words(n: int) -> str:
    """Spells a whole number the way a narrator says it: 4096 is "four thousand ninety-six"."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + (" " + number_words(rest) if rest else "")
    for size, name in _SCALES:
        if n >= size:
            rest = n % size
            return number_words(n // size) + " " + name + (" " + number_words(rest) if rest else "")
    raise ValueError(n)


def _ordinal_words(n: int) -> str:
    words = number_words(n)
    head, _, last = words.rpartition(" ")
    if "-" in last:
        h2, _, last = last.rpartition("-")
        head, sep = (f"{head} {h2}".strip(), "-")
    else:
        sep = " "
    if last in _ORDINAL:
        last = _ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}" if head else last


def _say_number(m: re.Match) -> str:
    whole, frac, suffix = m.group(1).replace(",", ""), m.group(2), m.group(3)
    if suffix:
        return " " + _ordinal_words(int(whole)) + " "
    out = number_words(int(whole))
    if frac:
        out += " point " + " ".join(_ONES[int(d)] for d in frac)
    return f" {out} "


# A number with optional thousands commas and decimals, or an ordinal such
# as 30th. It must not touch identifiers like H100 or bf16 (a letter before
# the digits), which read better as they are.
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(st|nd|rd|th)?(?![\d])")


def spoken(text: str) -> str:
    """Returns the text the synthesiser reads for a caption sentence.

    Rules apply first; then every number becomes words, because the voice
    misreads long digit strings ("5120" came out as "5,1-serven-ox-ter").
    """
    out = text
    for pattern, repl in RULES:
        out = re.sub(pattern, repl, out)
    out = _NUMBER.sub(_say_number, out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return " ".join(out.split())


def _chunks(sentence: str, limit: int = CUE_MAX_CHARS) -> list[str]:
    """Splits one sentence into balanced caption pieces of at most `limit`.

    The pieces come out close to equal length, so a sentence never ends on
    a one-word cue. A break prefers to follow a comma or a dash.
    """
    if len(sentence) <= limit:
        return [sentence]
    words = sentence.split(" ")
    pieces = -(-len(sentence) // limit)  # ceiling division
    target = len(sentence) / pieces
    out: list[str] = []
    line = ""
    for i, word in enumerate(words):
        candidate = f"{line} {word}".strip()
        if line and len(candidate) > limit:
            out.append(line)
            line = word
            continue
        line = candidate
        left = len(" ".join(words[i + 1 :]))
        if len(out) < pieces - 1 and left and (
            len(line) >= target or (len(line) >= 0.7 * target and line[-1] in ",;:—")
        ):
            out.append(line)
            line = ""
    if line:
        out.append(line)
    return out


@dataclass
class Cue:
    """One caption cue, in seconds from the start of its narration block."""

    start: float
    end: float
    text: str


@dataclass
class Speech:
    """A synthesised narration block."""

    path: Path
    duration: float
    text: str
    sentences: list[Cue] = field(default_factory=list)  # one per sentence
    cues: list[Cue] = field(default_factory=list)  # caption-sized pieces


def _remote_config() -> dict:
    url = os.environ.get("LECTURE_TTS_URL", "").strip()
    return {
        "base_url": url.rstrip("/"),
        "path": os.environ.get("LECTURE_TTS_PATH", "/v1/audio/speech"),
        "token": os.environ.get("LECTURE_TTS_TOKEN", ""),
        "workers": int(os.environ.get("LECTURE_TTS_WORKERS", "4")),
    }


def estimate_seconds(spoken_text: str) -> float:
    """The length estimate mode gives a sentence: words over words per second."""
    words = len(spoken_text.split())
    return max(0.5, words / WORDS_PER_SECOND)


class Voice:
    """Synthesises and caches narration."""

    def __init__(self, cache_dir: Path | None = None):
        self.mode = mode()
        self.silent = self.mode in ("estimate", "collect")
        self._collect = None
        if self.mode == "collect":
            self._collect = Path(os.environ.get("LECTURE_OUT", ".")) / "narration.jsonl"
        if self.silent:
            base = Path(cache_dir) if cache_dir else ESTIMATE_DIR
            self.cache_dir = base / f"{WORDS_PER_SECOND}wps"
        else:
            base = Path(cache_dir) if cache_dir else CACHE_DIR
            self.cache_dir = base / _safe(f"{MODEL_ID}-{VOICE}-{SPEED}-{LANG}-{SAMPLE_RATE}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._remote = _remote_config()
        self.synthesised = 0  # sentences synthesised in this process
        self.cached = 0  # sentences served from the cache

    @staticmethod
    def _key(spoken_text: str) -> str:
        blob = f"{MODEL_ID}|{VOICE}|{SPEED}|{LANG}|{SAMPLE_RATE}|{spoken_text}".encode()
        return hashlib.sha256(blob).hexdigest()[:24]

    def _cache_path(self, spoken_text: str) -> Path:
        return self.cache_dir / f"{self._key(spoken_text)}.wav"

    def _synthesize_remote(self, spoken_text: str) -> np.ndarray:
        cfg = self._remote
        if not cfg["base_url"]:
            raise RuntimeError(
                "Narration isn't cached and LECTURE_TTS_URL is not set. Set LECTURE_TTS_URL (and "
                "LECTURE_TTS_TOKEN) to the speech server, or preview with LECTURE_TTS=estimate."
            )
        url = f"{cfg['base_url']}{cfg['path']}"
        body = json.dumps(
            {
                "model": MODEL_ID,
                "input": spoken_text,
                "voice": VOICE,
                "speed": SPEED,
                "response_format": "wav",
                "lang": LANG,
            }
        ).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "lecture-kit-tts/2"}
        if cfg["token"]:
            headers["Authorization"] = f"Bearer {cfg['token']}"
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        last: Exception | None = None
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = resp.read()
                break
            except urllib.error.HTTPError as err:
                detail = err.read()[:300].decode("utf-8", errors="replace")
                last = RuntimeError(f"Speech server HTTP {err.code} for {len(spoken_text)} chars: {detail}")
                if err.code < 500:
                    raise last from err
            except urllib.error.URLError as err:
                last = RuntimeError(f"Speech server unreachable at {cfg['base_url']}: {err}")
        else:
            assert last is not None
            raise last
        with wave.open(io.BytesIO(data), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise RuntimeError(
                    f"The speech server returned a WAV with {w.getnchannels()} channels and "
                    f"{8 * w.getsampwidth()}-bit samples; the kit expects mono 16-bit."
                )
            rate = w.getframerate()
            if rate != SAMPLE_RATE:
                raise RuntimeError(f"The speech server returned {rate} Hz audio; the kit expects {SAMPLE_RATE} Hz.")
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
        if pcm.size < SAMPLE_RATE // 20:  # under 50 ms is almost certainly broken
            raise RuntimeError("The speech server returned near-empty audio.")
        return pcm

    def _synthesize(self, spoken_text: str) -> np.ndarray:
        if self.silent:
            return np.zeros(int(round(estimate_seconds(spoken_text) * SAMPLE_RATE)), dtype=np.int16)
        return self._synthesize_remote(spoken_text)

    def _fill_cache(self, spoken_texts: list[str]) -> None:
        """Synthesises cache misses; distinct misses run concurrently."""
        misses = list(dict.fromkeys(t for t in spoken_texts if not self._cache_path(t).exists()))
        if not misses:
            return
        workers = 1 if self.silent else max(1, min(64, self._remote["workers"]))

        def one(text: str) -> tuple[str, np.ndarray]:
            return text, self._synthesize(text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for text, pcm in pool.map(one, misses):
                _write_wav(self._cache_path(text), pcm)
                self.synthesised += 1

    def speak(self, text: str, say: str | None = None) -> Speech:
        """Synthesises a narration block and returns its audio and timings.

        Args:
          text: What the captions and speaker notes show.
          say: What the voice reads instead, if the automatic spoken form is
            wrong. It must have the same number of sentences as `text`, so
            the captions stay in step with the voice.
        """
        captions = split_sentences(text)
        if not captions:
            raise ValueError("Narration text is empty.")
        if say is None:
            voiced = [spoken(s) for s in captions]
        else:
            # The override goes through the same rules, so its numbers are
            # spelled out too.
            voiced = [spoken(s) for s in split_sentences(say)]
            if len(voiced) != len(captions):
                raise ValueError(
                    f"`say` has {len(voiced)} sentences but the text has {len(captions)}; "
                    "give them the same number of sentences."
                )
        if self._collect is not None:
            with self._collect.open("a", encoding="utf-8") as fh:
                for t in voiced:
                    fh.write(json.dumps({"spoken": t}, ensure_ascii=False) + "\n")
        existed = {t: self._cache_path(t).exists() for t in set(voiced)}
        self._fill_cache(voiced)
        gap = np.zeros(int(SENTENCE_GAP * SAMPLE_RATE), dtype=np.int16)
        pieces: list[np.ndarray] = []
        sentences: list[Cue] = []
        cues: list[Cue] = []
        cursor = 0
        for i, (caption, voice_text) in enumerate(zip(captions, voiced)):
            if i:
                pieces.append(gap)
                cursor += len(gap)
            pcm = _read_wav(self._cache_path(voice_text))
            if existed[voice_text]:
                self.cached += 1
            else:
                # First use of a fresh synthesis; later duplicates count as hits.
                existed[voice_text] = True
            start = cursor / SAMPLE_RATE
            end = (cursor + len(pcm)) / SAMPLE_RATE
            sentences.append(Cue(start, end, caption))
            # Share the sentence's time among its caption pieces by length.
            parts = _chunks(caption)
            total = sum(len(p) for p in parts)
            t = start
            for part in parts:
                dt = (end - start) * len(part) / total
                cues.append(Cue(t, t + dt, part))
                t += dt
            pieces.append(pcm)
            cursor += len(pcm)
        audio = np.concatenate(pieces)
        whole = self.cache_dir / "blocks" / f"{self._key('|'.join(voiced))}.wav"
        if not whole.exists():
            whole.parent.mkdir(parents=True, exist_ok=True)
            _write_wav(whole, audio)
        return Speech(whole, len(audio) / SAMPLE_RATE, " ".join(captions), sentences, cues)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    tmp.replace(path)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
