#!/usr/bin/env python3
"""Renders lectures, cuts them into slides, and publishes them.

Usage:

    python3 lectures/build.py 02-memory-arithmetic
    python3 lectures/build.py 02-memory-arithmetic --frames
    python3 lectures/build.py 02-memory-arithmetic --quality l --only kv-per-token
    python3 lectures/build.py --all --jobs 6 --out /workspace/learn-inference/build
    python3 lectures/build.py --publish 02-memory-arithmetic --from DIR --dry-run
    python3 lectures/build.py --publish --all --from DIR --upload

A target is a chapter slug with a scene at `lectures/scenes/<slug>.py`.

Rendering, for each target:

1. Computes a content-hash `version` from the scene, the kit, the image
   recipe, `course.json`, `pronunciations.json`, the render quality, and the
   chapter front matter the lecture shows.
2. Skips the target if a finished build of that version exists, unless you
   pass `--force`.
3. Renders the scene with `--save_sections`, in the lecture image or, with
   `LECTURE_NATIVE=1`, directly on the host.
4. Re-encodes the video for the web, checks that every section ends on a
   still frame that matches the saved slide, converts each slide to WebP,
   writes WebVTT captions, and writes `dist/manifest.json`.

Publishing (`--publish`) reads finished builds from `--from DIR`. With
`--upload` it puts every file in the R2 bucket with `cf`; without it, it
assumes another process copied the files. Either way it checks every public
URL for status 200, the content type, and the size, then writes the
course's manifest (`content/lectures/<slug>.json`). `--dry-run` prints what
it would do and touches neither the network nor the repository.

The course's bucket, public URL, and paths come from `lectures/course.json`.
Only the Python standard library is needed on the host, plus Docker for
renders and `cf` and `curl` for publishing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCENES = HERE / "scenes"
KIT = HERE / "kit"
DEFAULT_OUT = HERE / ".build"


def _load_course():
    """Loads kit/course.py by path, without importing the kit (and manim)."""
    spec = importlib.util.spec_from_file_location("lecture_course", KIT / "course.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


course = _load_course()

# Bump when the build's outputs change for the same scene and kit (encoder
# settings, slide format, caption format, manifest shape). It feeds the
# version hash.
PIPELINE = "1"

QUALITY = {
    # flag: (manim args, fps, width, height)
    "h": (["-qh", "--fps", "30"], 30, 1920, 1080),
    "l": (["-ql"], 15, 854, 480),
}

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webp": "image/webp",
    ".vtt": "text/vtt; charset=utf-8",
    ".json": "application/json",
}

# R2 puts through `cf` are limited to 300 MB per object.
MAX_OBJECT_BYTES = 300 * 1024 * 1024

# With LECTURE_NATIVE=1 (a host with the image's toolchain, such as a
# Runpod pod set up by lectures/pod/setup.sh), commands run directly and
# paths stay as they are. Otherwise the repository is mounted at /work and
# an output directory outside it at /out.
NATIVE = os.environ.get("LECTURE_NATIVE") == "1"
WORK = "/work"
OUT_MOUNT = "/out"
CACHE_MOUNT = "/tts-cache"

# Environment passed through to the render. Values travel through the
# process environment (`docker run -e NAME`), never on a command line, so
# the token never lands in a log.
PASS_ENV = (
    "LECTURE_TTS",
    "LECTURE_TTS_URL",
    "LECTURE_TTS_TOKEN",
    "LECTURE_TTS_PATH",
    "LECTURE_TTS_WORKERS",
)

# Files whose bytes feed every version hash.
RECIPE = ("Dockerfile", "fonts.sh", "course.json", "pronunciations.json")


class BuildError(RuntimeError):
    pass


def log(target: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {target}: {message}", flush=True)


def cpu_count() -> int:
    """Cores this process may use: `LECTURE_CPUS`, else affinity and cgroup quota.

    A container often reports the host's core count through `os.cpu_count()`,
    so the CPU affinity mask and a cgroup CPU quota take precedence.
    """
    if os.environ.get("LECTURE_CPUS"):
        return max(1, int(os.environ["LECTURE_CPUS"]))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # cgroup v2, then v1
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        try:
            quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip()
            period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
        except OSError:
            quota, period = "max", "0"
    if quota not in ("max", "-1") and period.isdigit() and int(period) > 0:
        n = min(n, max(1, -(-int(quota) // int(period))))
    return max(1, n)


def tts_mode() -> str:
    return (os.environ.get("LECTURE_TTS", "remote").strip().lower() or "remote")


# Targets and versions ---------------------------------------------------------


class Target:
    def __init__(self, slug: str):
        self.slug = slug
        self.name = slug
        self.scene = SCENES / f"{slug}.py"

    @property
    def manifest(self) -> Path:
        return course.manifest_file(self.slug)


def resolve_targets(names: list[str], everything: bool) -> list[Target]:
    if everything:
        return [Target(p.stem) for p in sorted(SCENES.glob("*.py")) if not p.stem.startswith("_")]
    targets = []
    for name in names:
        slug = Path(name.rstrip("/")).name
        slug = slug[:-3] if slug.endswith(".py") else slug
        target = Target(slug)
        if not target.scene.exists():
            raise BuildError(f"{name}: no scene at {target.scene.relative_to(REPO)}.")
        if not course.chapter_file(slug).exists():
            raise BuildError(f"{name}: no chapter at {course.chapter_file(slug).relative_to(REPO)}.")
        targets.append(target)
    return targets


def version_of(target: Target, quality: str) -> str:
    """A 12-hex-digit hash of everything that decides what the video shows."""
    h = hashlib.sha256()
    h.update(f"pipeline={PIPELINE}\nquality={quality}\n".encode())
    files = sorted(KIT.rglob("*.py")) + [HERE / f for f in RECIPE] + [target.scene]
    for path in files:
        if not path.exists():
            continue
        h.update(f"--- {path.relative_to(HERE).as_posix()}\n".encode())
        h.update(path.read_bytes())
    # The title card and the closing card show front matter, including the
    # next chapter's title and summary.
    slugs = course.chapter_slugs()
    shown = [target.slug]
    i = slugs.index(target.slug)
    if i + 1 < len(slugs):
        shown.append(slugs[i + 1])
    for slug in shown:
        h.update(json.dumps(course.shown_fields(slug), sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:12]


def work_dir(out: Path, target: Target, version: str, preview: bool, estimate: bool) -> Path:
    suffix = "-preview" if preview else ("-estimate" if estimate else "")
    return out / target.slug / f"{version}{suffix}"


# Docker -------------------------------------------------------------------------


def image_tag() -> str:
    h = hashlib.sha256()
    for name in ("Dockerfile", "fonts.sh"):
        h.update((HERE / name).read_bytes())
    return f"{course.config()['course']}-lectures:{h.hexdigest()[:10]}"


def ensure_image() -> str:
    if NATIVE:
        return "native"
    if shutil.which("docker") is None:
        raise BuildError("docker is not on PATH. Install Docker Desktop, or set LECTURE_NATIVE=1 on a set-up pod.")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise BuildError("The Docker daemon is not running. Start it with `open -a Docker` and retry.")
    tag = image_tag()
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
        print(f"Building the lecture image {tag} (first run only) ...", flush=True)
        platform = "linux/arm64" if _arm() else "linux/amd64"
        result = subprocess.run(["docker", "build", "--platform", platform, "-t", tag, str(HERE)])
        if result.returncode != 0:
            raise BuildError("docker build failed.")
    return tag


def _arm() -> bool:
    import platform

    return platform.machine().lower() in ("arm64", "aarch64")


class Runner:
    """Runs a command in the lecture image, or natively, with path mapping."""

    def __init__(self, tag: str, out: Path):
        self.tag = tag
        self.out = out.resolve()
        cache = os.environ.get("LECTURE_TTS_CACHE")
        self.cache = Path(cache).resolve() if cache else None

    def path(self, p: Path) -> str:
        p = Path(p).resolve()
        if NATIVE:
            return str(p)
        for root, mount in ((REPO, WORK), (self.out, OUT_MOUNT)):
            try:
                return f"{mount}/{p.relative_to(root).as_posix()}".rstrip("/")
            except ValueError:
                continue
        raise BuildError(f"{p} is neither in the repository nor in the output directory.")

    def run(self, args: list[str], env: dict[str, str], log_path: Path) -> None:
        passed = {k: os.environ[k] for k in PASS_ENV if os.environ.get(k)}
        if NATIVE:
            cmd = list(args)
            run_env = {**os.environ, "PYTHONPATH": str(HERE), **env}
            if self.cache:
                run_env["LECTURE_TTS_CACHE"] = str(self.cache)
        else:
            cmd = ["docker", "run", "--rm", "-v", f"{REPO}:{WORK}", "-w", WORK]
            if not _inside(self.out, REPO):
                self.out.mkdir(parents=True, exist_ok=True)
                cmd += ["-v", f"{self.out}:{OUT_MOUNT}"]
            all_env = {**passed, **env}
            if self.cache:
                if _inside(self.cache, REPO):
                    all_env["LECTURE_TTS_CACHE"] = self.path(self.cache)
                else:
                    self.cache.mkdir(parents=True, exist_ok=True)
                    cmd += ["-v", f"{self.cache}:{CACHE_MOUNT}"]
                    all_env["LECTURE_TTS_CACHE"] = CACHE_MOUNT
            # `-e NAME` copies the value from this process's environment, so
            # values (the token among them) never appear in the command line.
            for k in sorted(all_env):
                cmd += ["-e", k]
            cmd += [self.tag, *args]
            run_env = {**os.environ, **all_env}
        with log_path.open("a") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env=run_env)
        if result.returncode != 0:
            tail = log_path.read_text(errors="replace").splitlines()[-40:]
            raise BuildError(
                f"{' '.join(args[:2])} failed (exit {result.returncode}). Last lines of {log_path}:\n" + "\n".join(tail)
            )


def _inside(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# Rendering ----------------------------------------------------------------------


def build_target(target: Target, opts: argparse.Namespace, runner: Runner) -> dict:
    version = version_of(target, opts.quality)
    preview = opts.frames or bool(opts.only)
    estimate = tts_mode() in ("estimate", "collect")
    work = work_dir(opts.out, target, version, preview, estimate)
    dist = work / "dist"
    done = work / "done.json"
    if done.exists() and not opts.force and not preview:
        log(target.name, f"version {version} already built in {_show(work)}; skipped (pass --force to rebuild)")
        return {"target": target.name, "version": version, "skipped": True, **json.loads(done.read_text())}
    if work.exists():
        shutil.rmtree(work)
    (work / "render").mkdir(parents=True)
    log_path = work / "build.log"
    manim_args, fps, _width, height = QUALITY[opts.quality]
    env = {
        "LECTURE_OUT": runner.path(work / "render"),
        "LECTURE_LAYOUT": "warn" if preview else "strict",
        "OMP_NUM_THREADS": str(opts.omp_threads),
        "LECTURE_FFMPEG_THREADS": str(opts.ffmpeg_threads),
    }
    if opts.only:
        env["LECTURE_ONLY"] = opts.only
    if os.environ.get("LECTURE_ENCODER"):
        env["LECTURE_ENCODER"] = os.environ["LECTURE_ENCODER"]
    args = ["manim", "render", *manim_args, "--disable_caching", "--media_dir", runner.path(work / "media")]
    args.append("--dry_run" if opts.frames else "--save_sections")
    args += [runner.path(target.scene), "Lecture"]
    what = "frames only" if opts.frames else "video"
    log(target.name, f"rendering version {version} at {height}p ({what}{', estimated narration' if estimate else ''})")
    t0 = time.time()
    runner.run(args, env, log_path)
    render_seconds = time.time() - t0
    log(target.name, f"rendered in {render_seconds:.0f} s; post-processing")
    t1 = time.time()
    runner.run(
        ["python", runner.path(Path(__file__)), "--_postprocess", runner.path(work),
         "--_fps", str(fps), "--_mode", "frames" if opts.frames else "video"],
        env,
        log_path,
    )
    result = json.loads((work / "result.json").read_text())
    result["render_seconds"] = round(render_seconds, 1)
    result["post_seconds"] = round(time.time() - t1, 1)
    result["version"] = version
    result["quality"] = opts.quality
    anchors = course.section_anchors(target.slug)
    bad = [s["book_section"] for s in result["sections"] if s.get("book_section") and s["book_section"] not in anchors]
    if bad:
        raise BuildError(
            f"book_section ids not found as ## headings of {_show(course.chapter_file(target.slug))}: "
            f"{', '.join(bad)}. Known ids: {', '.join(anchors)}"
        )
    for w in result.get("warnings", []):
        log(target.name, f"warning: {w}")
    if not opts.frames:
        manifest = make_manifest(target.slug, version, result, anchors)
        (dist / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not preview:
        done.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    duration = result["duration"]
    log(target.name, f"{duration / 60:.1f} min, {len(result['sections'])} slides")
    if preview:
        log(target.name, f"preview frames in {_show(work / 'render' / 'frames')}; contact sheet {_show(work / 'contact.png')}")
        if not opts.frames:
            log(target.name, f"preview video {_show(dist / 'lecture.mp4')}")
        return {"target": target.name, "version": version, "preview": True, **result}
    if render_seconds and duration:
        log(target.name, f"{render_seconds / max(duration / 60, 1e-6):.0f} s of render per minute of video")
    log(target.name, f"built {_show(dist)}" + (" (estimated narration; can't be published)" if estimate else ""))
    return {"target": target.name, "version": version, **result}


def _show(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def make_manifest(slug: str, version: str, result: dict, anchors: dict[str, str]) -> dict:
    """The manifest the site reads; every path in it is relative to `base_url`."""
    sections = []
    for s in result["sections"]:
        entry = {
            "id": s["id"],
            "title": s["title"],
            "start": s["start"],
            "end": s["end"],
            "slide": s["slide"],
            "notes": s["notes"],
        }
        if s.get("book_section"):
            entry["book_section"] = s["book_section"]
            entry["book_anchor"] = anchors[s["book_section"]]
        sections.append(entry)
    voice = course.config()["voice"]
    tts = result.get("tts") or {}
    return {
        "course": course.config()["course"],
        "slug": slug,
        "title": result["title"],
        "version": version,
        "base_url": course.base_url(slug, version),
        "video": {
            "src": "lecture.mp4",
            "duration": result["duration"],
            "width": result["width"],
            "height": result["height"],
            "poster": "poster.webp",
        },
        "captions": "captions.vtt",
        "narration": {"mode": tts.get("mode", "remote"), "model": voice["model"], "voice": voice["voice"]},
        "sections": sections,
    }


# Publishing -----------------------------------------------------------------------


class Publisher:
    """Publishes finished builds: optional upload, URL checks, manifest write.

    With `dry_run`, every action is printed instead of run: no `cf`, no
    `curl`, and no file written. The checks on the local build still run.
    """

    def __init__(self, upload: bool, dry_run: bool, force: bool, printer=print, verify: bool = True):
        self.upload = upload
        self.verify = verify
        self.dry_run = dry_run
        self.force = force
        self.say = printer

    def find_build(self, root: Path, target: Target, version: str | None) -> tuple[Path, str]:
        want = version or version_of(target, "h")
        work = root / target.slug / want
        if (work / "done.json").exists():
            return work, want
        found = sorted(
            p.name for p in (root / target.slug).glob("*") if (p / "done.json").exists()
        ) if (root / target.slug).exists() else []
        hint = f" Finished builds there: {', '.join(found)}." if found else " There are no finished builds there."
        raise BuildError(
            f"no finished build of version {want} under {root / target.slug}.{hint} "
            "Render the current scene at --quality h, or pass --version to publish another one."
        )

    def check_build(self, work: Path, target: Target, version: str) -> tuple[dict, list[Path]]:
        if work.name != version:
            raise BuildError(f"{work} is a preview or estimate build; only final builds can be published.")
        result = json.loads((work / "done.json").read_text())
        mode = (result.get("tts") or {}).get("mode", "remote")
        if mode != "remote":
            raise BuildError(f"{work} was rendered with {mode} narration; render it with the speech server first.")
        if result.get("quality", "h") != "h" or (result.get("width"), result.get("height")) != (1920, 1080):
            raise BuildError(f"{work} isn't a 1080p build; publish only --quality h renders.")
        dist = work / "dist"
        manifest_path = dist / "manifest.json"
        if not manifest_path.exists():
            raise BuildError(f"{manifest_path} is missing.")
        manifest = json.loads(manifest_path.read_text())
        expected = course.base_url(target.slug, version)
        if manifest.get("version") != version or manifest.get("slug") != target.slug:
            raise BuildError(f"{manifest_path} is for {manifest.get('slug')} {manifest.get('version')}, not {target.slug} {version}.")
        if manifest.get("base_url") != expected:
            raise BuildError(
                f"{manifest_path} has base_url {manifest.get('base_url')}, but course.json gives {expected}. "
                "Rebuild after changing public_base or key_prefix."
            )
        if (manifest.get("narration") or {}).get("mode", "remote") != "remote":
            raise BuildError(f"{manifest_path} describes estimated narration.")
        needed = [manifest["video"]["src"], manifest["video"]["poster"], manifest["captions"]]
        needed += [s["slide"] for s in manifest["sections"]]
        missing = [rel for rel in needed if not (dist / rel).is_file()]
        if missing:
            raise BuildError(f"{dist} lacks {', '.join(missing)}.")
        files = sorted(p for p in dist.rglob("*") if p.is_file())
        for p in files:
            if p.suffix not in CONTENT_TYPES:
                raise BuildError(f"No content type for {p.relative_to(dist)}; remove it from {dist}.")
            if p.stat().st_size > MAX_OBJECT_BYTES:
                raise BuildError(f"{p.relative_to(dist)} is {p.stat().st_size / 1e6:.0f} MB; R2 puts through cf allow 300 MB.")
        # The video and then the manifest go last, so a partial upload never
        # leaves a manifest that points at missing files.
        files.sort(key=lambda p: (p.suffix == ".json", p.suffix == ".mp4"))
        return manifest, files

    def publish(self, root: Path, target: Target, version: str | None = None) -> dict:
        work, version = self.find_build(root, target, version)
        manifest, files = self.check_build(work, target, version)
        dest = target.manifest
        if dest.exists() and not self.force:
            try:
                if json.loads(dest.read_text()).get("version") == version:
                    self.say(f"{target.name}: {_show(dest)} already points at {version}; skipped")
                    return {"target": target.name, "version": version, "skipped": True}
            except json.JSONDecodeError:
                pass
        dist = work / "dist"
        prefix = course.key_prefix(target.slug, version)
        bucket = course.config()["bucket"]
        if self.upload:
            if not self.dry_run and shutil.which("cf") is None:
                raise BuildError("The `cf` CLI is not on PATH; it's needed for --upload.")
            for path in files:
                rel = path.relative_to(dist).as_posix()
                self.put(bucket, f"{prefix}/{rel}", path, CONTENT_TYPES[path.suffix], course.base_url(target.slug, version) + rel)
        checks = [
            (course.base_url(target.slug, version) + p.relative_to(dist).as_posix(),
             CONTENT_TYPES[p.suffix].split(";")[0], p.stat().st_size)
            for p in files
        ]
        if not self.verify:
            self.say(f"{target.name}: skipped the URL checks (--no-verify); run --publish again without it "
                     "once the public domain serves the bucket")
        elif self.dry_run:
            for url, ctype, size in checks:
                self.say(f"would check: {url} -> 200, {ctype}, {size} bytes")
        else:
            check_urls(checks)
        text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        if self.dry_run:
            self.say(f"would write {_show(dest)}:\n{text}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(dest)
            self.say(f"{target.name}: published {course.base_url(target.slug, version)} and wrote {_show(dest)}")
        return {"target": target.name, "version": version, "files": len(files), "manifest": manifest}

    def put(self, bucket: str, key: str, path: Path, ctype: str, url: str) -> None:
        """Uploads one object, unless the same bytes are already public.

        Keys are content-hashed, so an object that exists with the same size
        and type is this file. One that differs is never overwritten.
        """
        cmd = ["cf", "r2", "objects", "put", key, "--bucket-name", bucket, "--file", str(path),
               "--content-type", ctype, "--quiet"]
        if self.dry_run:
            self.say("would run: " + " ".join(_quote(c) for c in cmd))
            return
        size = path.stat().st_size
        state = head(url)
        if state is not None:
            status, got_type, got_len = state
            if status == 200:
                if got_type == ctype.split(";")[0] and got_len in (None, size):
                    self.say(f"  {key}: already uploaded")
                    return
                raise BuildError(
                    f"{url} already exists with type {got_type} and {got_len} bytes, not {ctype} and {size}; "
                    "refusing to overwrite a published object."
                )
        with tempfile.TemporaryDirectory() as scratch:
            for attempt in range(3):
                # Run from a scratch directory so cf leaves no state in the repo.
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=scratch)
                if res.returncode == 0:
                    return
                time.sleep(2 * (attempt + 1))
        raise BuildError(f"Upload of {key} failed:\n{res.stdout}\n{res.stderr}")


def _quote(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:@%+=-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def head(url: str) -> tuple[int, str | None, int | None] | None:
    """Returns (status, content type, length) for a HEAD request, or None on a network error."""
    res = subprocess.run(["curl", "-sSI", "--max-time", "30", url], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    text = res.stdout.lower()
    blocks = [b for b in re.split(r"\r?\n\r?\n", text) if b.strip()]
    last = blocks[-1] if blocks else ""
    status = re.search(r"^http/\S+ (\d+)", last, re.M)
    got_type = re.search(r"^content-type:\s*([^\r\n;]+)", last, re.M)
    got_len = re.search(r"^content-length:\s*(\d+)", last, re.M)
    if not status:
        return None
    return (
        int(status.group(1)),
        got_type.group(1).strip() if got_type else None,
        int(got_len.group(1)) if got_len else None,
    )


def check_urls(checks: list[tuple[str, str, int]]) -> None:
    """Checks a lecture's public URLs concurrently; reports the first failure in list order."""
    if not checks:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(checks))) as pool:
        futures = [pool.submit(check_url, url, ctype, size) for url, ctype, size in checks]
        for fut in futures:
            fut.result()


def check_url(url: str, ctype: str, size: int) -> None:
    state = None
    for attempt in range(4):
        state = head(url)
        if state is not None and state[0] == 200:
            _, got_type, got_len = state
            if got_type != ctype:
                raise BuildError(f"{url} has content type {got_type or 'none'}, expected {ctype}.")
            if got_len is not None and got_len != size:
                raise BuildError(f"{url} is {got_len} bytes, expected {size}.")
            return
        time.sleep(2 * (attempt + 1))
    raise BuildError(f"{url} did not return 200 (got {state[0] if state else 'no response'}).")


# Post-processing (runs inside the image, or natively) -----------------------------


def _vtt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _frame_at(video: Path, t: float, width: int, height: int):
    import numpy as np

    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.4f}", "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    if len(out) != width * height * 3:
        raise BuildError(f"Could not read a frame of {video.name} at {t:.3f} s.")
    return np.frombuffer(out, dtype=np.uint8).reshape(height, width, 3).astype(np.int16)


def contact_sheet(frames: list[Path], out: Path, cols: int = 3, width: int = 640) -> None:
    """Tiles the slide frames into one image, for a quick visual review."""
    from PIL import Image, ImageDraw

    thumbs = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        thumbs.append(im.resize((width, round(im.height * width / im.width)), Image.LANCZOS))
    h = thumbs[0].height
    rows = -(-len(thumbs) // cols)
    gap = 12
    sheet = Image.new("RGB", (cols * width + (cols + 1) * gap, rows * (h + 28) + (rows + 1) * gap), (220, 216, 206))
    draw = ImageDraw.Draw(sheet)
    for i, (f, im) in enumerate(zip(frames, thumbs)):
        x = gap + (i % cols) * (width + gap)
        y = gap + (i // cols) * (h + 28 + gap)
        draw.text((x, y), f.stem, fill=(28, 27, 24))
        sheet.paste(im, (x, y + 22))
    sheet.save(out)


def _moving_blocks(a, b, block: int = 8, threshold: float = 12.0) -> int:
    """Counts 8x8 blocks whose mean brightness differs by more than `threshold` levels.

    Averaging over blocks cancels the codec's keyframe noise (a few levels)
    but keeps real motion, which moves whole glyphs and shapes.
    """
    import numpy as np

    h, w = a.shape[0] // block * block, a.shape[1] // block * block

    def pool(x):
        return x[:h, :w].mean(axis=2).reshape(h // block, block, w // block, block).mean(axis=(1, 3))

    return int((np.abs(pool(a) - pool(b)) > threshold).sum())


def _has_encoder(name: str) -> bool:
    res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return re.search(rf"^\s*V\S*\s+{re.escape(name)}\b", res.stdout, re.M) is not None


def video_codec_args(fps: int) -> list[str]:
    """Encoder arguments: libx264 by default, NVENC when asked for and present.

    `LECTURE_ENCODER=h264_nvenc` selects NVIDIA's encoder on a GPU pod. If
    FFmpeg lacks it, the build falls back to libx264 and says so.
    """
    gop = ["-g", str(2 * fps), "-keyint_min", str(fps)]
    common = ["-profile:v", "high", "-pix_fmt", "yuv420p", *gop]
    wanted = os.environ.get("LECTURE_ENCODER", "libx264").strip() or "libx264"
    if wanted == "h264_nvenc":
        if _has_encoder("h264_nvenc"):
            return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "21", "-b:v", "0",
                    "-bf", "2", "-forced-idr", "1", *common]
        print("warning: LECTURE_ENCODER=h264_nvenc, but FFmpeg has no h264_nvenc; using libx264.", flush=True)
    elif wanted != "libx264":
        raise BuildError(f"LECTURE_ENCODER must be libx264 or h264_nvenc, not {wanted!r}.")
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "animation", "-level", "4.1",
            "-sc_threshold", "0", *common]


def postprocess(work: Path, fps: int, mode: str) -> None:
    """Turns a render into web assets."""
    import numpy as np
    from PIL import Image

    meta = json.loads((work / "render" / "meta.json").read_text())
    dist = work / "dist"
    (dist / "slides").mkdir(parents=True, exist_ok=True)
    width, height = meta["width"], meta["height"]
    warnings: list[str] = []
    for s in meta["sections"]:
        warnings.extend(s.get("warnings", []))
    if not meta["sections"]:
        raise BuildError("The scene made no slides; call self.slide(...) at least once.")

    # Slides: the kit's lossless final frame of each section, as WebP.
    slides = []
    for s in meta["sections"]:
        png = work / "render" / s["frame"]
        name = f"slides/{s['index']:02d}-{s['id']}.webp"
        img = Image.open(png).convert("RGB")
        if img.size != (width, height):
            raise BuildError(f"{png.name} is {img.size}, expected {width}x{height}.")
        img.save(dist / name, "WEBP", quality=90, method=6)
        slides.append(name)
    Image.open(work / "render" / meta["sections"][0]["frame"]).convert("RGB").save(
        dist / "poster.webp", "WEBP", quality=90, method=6
    )
    contact_sheet([work / "render" / s["frame"] for s in meta["sections"]], work / "contact.png")

    sections = []
    if mode == "frames":
        for s, slide in zip(meta["sections"], slides):
            sections.append({**_section_fields(s), "start": round(s["start"], 3), "end": round(s["end"], 3),
                             "slide": slide})
        _write_result(work, meta, sections, meta["duration"], warnings)
        return

    videos = list((work / "media" / "videos").rglob("Lecture.mp4"))
    if len(videos) != 1:
        raise BuildError(f"Expected one Lecture.mp4 in the render, found {len(videos)}.")
    source = videos[0]
    index_files = list(source.parent.glob("sections/Lecture.json"))
    if not index_files:
        raise BuildError("No sections index; the render needs --save_sections.")
    index = json.loads(index_files[0].read_text())

    # Section boundaries in the full video, from manim's own section files.
    # With --only, skipped sections have no video file and no frames.
    partial = bool(meta.get("only"))
    names = [e["name"] for e in index]
    ids = [s["id"] for s in meta["sections"]]
    expected = [i for i in ids if not partial or i in meta["only"]]
    if names != expected:
        raise BuildError(
            "Manim's sections don't match the kit's slides. Play nothing before the first "
            f"self.slide() and never call next_section() yourself.\nmanim: {names}\nkit:   {expected}"
        )
    by_name = {e["name"]: int(e["nb_frames"]) for e in index}
    frames = [by_name.get(i, 0) for i in ids]
    total = sum(frames)
    duration = total / fps
    cursor = 0
    for s, n, slide in zip(meta["sections"], frames, slides):
        start, end = cursor / fps, (cursor + n) / fps
        cursor += n
        if not partial and (abs(start - s["start"]) > 1.5 / fps or abs(end - s["end"]) > 1.5 / fps):
            raise BuildError(
                f"Section {s['id']!r}: video says {start:.3f}-{end:.3f} s, the kit recorded "
                f"{s['start']:.3f}-{s['end']:.3f} s."
            )
        sections.append({**_section_fields(s), "start": round(start, 3), "end": round(end, 3), "slide": slide})

    # The settled-frame rule, checked on the video itself: the last second of
    # every section is still, and the kit's saved slide is that frame.
    for s, sec, n in zip(meta["sections"], sections, frames):
        if n == 0:
            continue
        end = sec["end"]
        last = _frame_at(source, end - 1.5 / fps, width, height)
        earlier = _frame_at(source, end - 1.0 + 0.5 / fps, width, height)
        moving = _moving_blocks(last, earlier)
        if moving > 2:
            raise BuildError(
                f"Section {s['id']!r} is still moving in its last second ({moving} changed 8x8 blocks). "
                "End every section on a still frame."
            )
        png = np.asarray(Image.open(work / "render" / s["frame"]).convert("RGB")).astype(np.int16)
        # Chroma subsampling blurs text edges a little, hence the higher bar.
        drift = _moving_blocks(last, png, threshold=24)
        if drift > 2:
            raise BuildError(
                f"Section {s['id']!r}: the saved slide differs from the video's final frame "
                f"({drift} changed 8x8 blocks)."
            )

    # Web encode: H.264 High, yuv420p, AAC at -16 LUFS, a keyframe every 2 s
    # so seeking to a section start is quick, moov atom first.
    out = dist / "lecture.mp4"
    threads = os.environ.get("LECTURE_FFMPEG_THREADS", "0")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-threads", threads, "-i", str(source),
         "-map", "0:v:0", "-map", "0:a:0",
         *video_codec_args(fps), "-threads", threads,
         "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         # Speech loudness for the web (EBU R128 style), then pad the audio
         # to the video's length.
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,apad",
         "-t", f"{duration:.3f}",
         "-movflags", "+faststart", str(out)],
        check=True,
    )
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height",
         "-show_entries", "format=duration", "-of", "json", str(out)],
        capture_output=True, check=True, text=True).stdout)
    kinds = {st["codec_type"]: st for st in probe["streams"]}
    if "video" not in kinds or "audio" not in kinds:
        raise BuildError("The encoded video lacks a video or an audio stream.")
    if abs(float(probe["format"]["duration"]) - duration) > 0.25:
        raise BuildError(f"Encoded duration {probe['format']['duration']} s, expected {duration:.3f} s.")

    # Captions. In a partial preview, skipped sections have no video, so each
    # kept section's cues shift back to where that section starts in the video.
    lines = ["WEBVTT", ""]
    n = 0
    for s, sec, frames_in in zip(meta["sections"], sections, frames):
        if frames_in == 0:
            continue
        shift = s["start"] - sec["start"]
        for cue in s["cues"]:
            n += 1
            start, end = cue["start"] - shift, min(cue["end"] - shift, duration)
            lines += [str(n), f"{_vtt_time(start)} --> {_vtt_time(end)}", cue["text"], ""]
    (dist / "captions.vtt").write_text("\n".join(lines), encoding="utf-8")
    _write_result(work, meta, sections, duration, warnings)


def _section_fields(s: dict) -> dict:
    return {
        "id": s["id"],
        "title": s["title"],
        "notes": " ".join(s["notes"]),
        "book_section": s.get("book_section"),
    }


def _write_result(work: Path, meta: dict, sections: list[dict], duration: float, warnings: list[str]) -> None:
    result = {
        "title": meta["title"],
        "duration": round(duration, 3),
        "width": meta["width"],
        "height": meta["height"],
        "tts": meta.get("tts"),
        "sections": sections,
        "warnings": warnings,
    }
    (work / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


# CLI ------------------------------------------------------------------------------


def default_jobs() -> int:
    """Concurrent renders: half the cores natively, one under Docker Desktop.

    Manim's Cairo renderer draws on one core per scene, so a pod runs one
    scene per two cores and leaves the rest to the encoders. Docker Desktop
    on a laptop shares a small VM, so it renders one lecture at a time.
    """
    return max(1, cpu_count() // 2) if NATIVE else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("targets", nargs="*", help="chapter slugs with a scene in lectures/scenes/")
    parser.add_argument("--all", action="store_true", help="every scene under lectures/scenes/")
    parser.add_argument("--quality", choices=sorted(QUALITY), default="h", help="h: 1080p30 (default); l: 480p15")
    parser.add_argument("--jobs", type=int, default=None,
                        help="targets to render at once (default: half the cores natively, 1 under Docker)")
    parser.add_argument("--omp-threads", type=int, default=None,
                        help="OMP_NUM_THREADS for each render (default: cores / jobs)")
    parser.add_argument("--force", action="store_true", help="rebuild or republish even if the version is current")
    parser.add_argument("--frames", action="store_true",
                        help="render only each slide's final frame (fast layout check; no video)")
    parser.add_argument("--only", default="", help="comma-separated slide ids to render; the rest are skipped")
    parser.add_argument("--out", type=Path, default=None, help="build directory (default: lectures/.build)")
    parser.add_argument("--allow-estimate", action="store_true",
                        help="allow a 1080p full render with LECTURE_TTS=estimate (never publishable)")
    parser.add_argument("--publish", action="store_true",
                        help="publish finished builds from --from instead of rendering")
    parser.add_argument("--from", dest="from_dir", type=Path, default=None,
                        help="with --publish: the build directory to publish from (default: --out)")
    parser.add_argument("--upload", action="store_true", help="with --publish: put the files in R2 with cf first")
    parser.add_argument("--dry-run", action="store_true", help="with --publish: print the actions, change nothing")
    parser.add_argument("--no-verify", action="store_true",
                        help="with --publish: skip the public URL checks, for a bucket whose domain isn't attached yet")
    parser.add_argument("--version", dest="version", default=None,
                        help="with --publish and one target: publish this version instead of the current one")
    parser.add_argument("--_postprocess", help=argparse.SUPPRESS)
    parser.add_argument("--_fps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_mode", default="video", help=argparse.SUPPRESS)
    opts = parser.parse_args(argv)
    opts.parser = parser
    return opts


def main(argv: list[str]) -> int:
    opts = parse_args(argv)
    parser = opts.parser
    if opts._postprocess:
        try:
            postprocess(Path(opts._postprocess), opts._fps, opts._mode)
        except BuildError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        return 0

    opts.out = (opts.out or DEFAULT_OUT).resolve()
    if not opts.publish and (opts.upload or opts.dry_run or opts.from_dir or opts.version or opts.no_verify):
        parser.error("--upload, --dry-run, --no-verify, --from, and --version go with --publish.")
    if opts.publish and (opts.frames or opts.only):
        parser.error("--frames and --only make previews; they can't be published.")
    try:
        targets = resolve_targets(opts.targets, opts.all)
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if not targets:
        parser.error("name at least one target, or pass --all.")

    if opts.publish:
        return run_publish(opts, targets)

    preview = opts.frames or bool(opts.only)
    if tts_mode() in ("estimate", "collect") and not preview and opts.quality == "h" and not opts.allow_estimate:
        print("error: LECTURE_TTS=estimate makes silent placeholder narration. Use it with --frames, --only, "
              "or --quality l, or pass --allow-estimate for a 1080p test render (it can't be published).",
              file=sys.stderr)
        return 1
    opts.jobs = max(1, min(opts.jobs or default_jobs(), len(targets)))
    opts.ffmpeg_threads = max(1, cpu_count() // opts.jobs)
    opts.omp_threads = max(1, opts.omp_threads or opts.ffmpeg_threads)
    try:
        tag = ensure_image()
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    runner = Runner(tag, opts.out)
    print(f"{len(targets)} target(s), {opts.jobs} at a time, {opts.ffmpeg_threads} encoder threads and "
          f"OMP_NUM_THREADS={opts.omp_threads} each; output in {_show(opts.out)}", flush=True)

    failures: list[tuple[str, str]] = []
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=opts.jobs) as pool:
        futures = {pool.submit(build_target, t, opts, runner): t for t in targets}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                results.append(fut.result())
            except Exception as err:  # report every target, then fail
                failures.append((t.name, str(err)))
                log(t.name, f"FAILED: {err}")
    print()
    for r in sorted(results, key=lambda r: r["target"]):
        state = "skipped" if r.get("skipped") else ("preview" if r.get("preview") else "ok")
        extra = ""
        if "duration" in r:
            extra = f"  {r['duration'] / 60:.1f} min, {len(r.get('sections', []))} slides"
            if r.get("render_seconds") and not r.get("skipped"):
                extra += f", render {r['render_seconds']:.0f} s"
            if r.get("warnings"):
                extra += f", {len(r['warnings'])} warning(s)"
        print(f"  {state:8} {r['target']}  {r['version']}{extra}")
    for name, err in failures:
        print(f"  FAILED   {name}: {err.splitlines()[0]}")
    return 1 if failures else 0


def run_publish(opts: argparse.Namespace, targets: list[Target]) -> int:
    root = (opts.from_dir or opts.out).resolve()
    if opts.version and len(targets) != 1:
        opts.parser.error("--version needs exactly one target.")
    if not root.is_dir():
        print(f"error: {root} isn't a directory.", file=sys.stderr)
        return 1
    publisher = Publisher(upload=opts.upload, dry_run=opts.dry_run, force=opts.force, verify=not opts.no_verify)
    failures = []
    for t in targets:
        try:
            publisher.publish(root, t, opts.version)
        except BuildError as err:
            failures.append((t.name, str(err)))
            print(f"{t.name}: FAILED: {err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
