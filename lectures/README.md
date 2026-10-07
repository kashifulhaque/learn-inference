# Lectures

Each chapter can have one narrated video lecture, rendered with Manim. The
site plays the video on the chapter page with captions. Each slide's final
frame is a still image, with the slide's narration as its notes, so the same
build also gives a slide deck.

This directory holds the pipeline. It's course-agnostic: everything specific
to learn-inference lives in `course.json` and `pronunciations.json`, so you
can copy the directory to another course and edit those two files.

| Path | Contents |
|---|---|
| `course.json` | The course id and name, the chapter and manifest paths, the R2 bucket and public URL, the accent colour, the fonts, and the voice. |
| `pronunciations.json` | The course's spoken forms for jargon, such as `KV` and `GiB`. |
| `Dockerfile`, `fonts.sh` | The local render image: Manim Community 0.21, TeX Live, FFmpeg, WebP tools, and the site's fonts. |
| `kit/` | The Python package every scene imports: theme, slides, narration, code panels, diagrams, and plots. |
| `scenes/<slug>.py` | One `Lecture` scene per chapter. |
| `build.py` | Renders, cuts slides, writes captions and manifests, and publishes. |
| `pod/` | Scripts that set up a Runpod pod and render on it. |
| `tests/` | Tests for the host side of `build.py` and the narration rules. |
| `AUTHORING.md` | How to write a scene. Read it before you write one. |

Each published lecture has a manifest at `content/lectures/<slug>.json`,
which is committed.

## Requirements

- Python 3.10 or later on the host. `build.py` uses only the standard
  library.
- To render on a Mac: Docker Desktop, running. Start it with `open -a Docker`.
  Renders run in the image, so you don't need LaTeX, Manim, or FFmpeg on the
  host. The image is native `arm64` on Apple silicon.
- To render with real narration: the speech server's URL and token (see
  [Narration](#narration)).
- To publish: `curl`, and for `--upload`, the `cf` CLI signed in to the
  Cloudflare account.

## Narration

The voice is VibeVoice-1.5B on a remote server with an OpenAI-compatible
`/v1/audio/speech` endpoint. The kit sends one request per sentence with the
model, voice, speed, and language from `course.json`, asks for a WAV file,
and expects 24 kHz mono 16-bit audio back. Set the server in the environment:

```bash
export LECTURE_TTS_URL=https://tts.example.com
export LECTURE_TTS_TOKEN=TOKEN
```

Replace `TOKEN` with the server's bearer token. Never write the token into a
file in the repository; `build.py` passes it to the render through the
environment, so it doesn't appear in build logs.

Synthesised sentences are cached in `lectures/.cache/tts/`, keyed by the
model, voice, speed, language, sample rate, and spoken text. A re-render
synthesises only the sentences that changed. To use another directory, set
`LECTURE_TTS_CACHE`.

### Estimate mode

To preview without the speech server, set `LECTURE_TTS=estimate`. Each
sentence becomes silence that lasts its word count divided by 2.6 seconds
(`estimate_words_per_second` in `course.json`), which is close to the voice's
pace. Layout, timing, and captions behave as they do with real audio, and no
request leaves the machine.

Estimated audio goes to `lectures/.cache/tts-estimate/`, never into the real
cache. A full render in estimate mode lands in a `<version>-estimate`
directory and records the mode in its manifest. `build.py` refuses a 1080p
full render in estimate mode unless you pass `--allow-estimate`, and
`--publish` always refuses estimated builds.

### Synthesize narration in bulk

A render asks the speech server for one narration block at a time, so the
GPU sees only a few requests at once. To keep the GPU busy, synthesize every
sentence before you render:

1. Record every spoken sentence with a fast frames-only pass:

   ```bash
   LECTURE_TTS=collect python3 lectures/build.py --all --frames --jobs 6 --out COLLECT_DIR
   ```

1. Send the sentences that aren't cached yet, with many requests in flight:

   ```bash
   python3 lectures/prefetch.py --from COLLECT_DIR --concurrency 32 --redo-wer 0.15
   ```

1. Render as usual. Every sentence is a cache hit, so the render doesn't wait
   on the server.

Replace `COLLECT_DIR` with any scratch directory. The speech server checks
each sentence with speech recognition and returns its word error rate.
`prefetch.py` records that rate in `qa.jsonl` in the cache, and `--redo-wer`
synthesizes again, from scratch, every sentence above the threshold. To list
the worst sentences with what the recognizer heard, pass `--report`. Many
reported mismatches are spelling differences, such as "pre-fill" for
"prefill", rather than misread speech.

The spoken form spells every number as words, because the voice misreads long
digit strings. Captions keep the digits.

## Render a lecture

To render one lecture at 1080p, run the following command from the
repository root:

```bash
python3 lectures/build.py 02-memory-arithmetic
```

The output lands in `lectures/.build/<slug>/<version>/dist/`: `lecture.mp4`,
`poster.webp`, `captions.vtt`, `slides/NN-<id>.webp`, and `manifest.json`.

The build takes the following options:

| Option | Effect |
|---|---|
| `--all` | Renders every scene under `lectures/scenes/`. |
| `--quality {h,l}` | `h` renders 1920x1080 at 30 fps (the default); `l` renders 854x480 at 15 fps for previews. |
| `--frames` | Renders only each slide's final frame, for a fast layout check. |
| `--only ID,ID` | Renders only the named slides. |
| `--jobs N` | Renders `N` lectures at once. The default is half the cores with `LECTURE_NATIVE=1` and 1 under Docker. |
| `--omp-threads N` | Sets `OMP_NUM_THREADS` for each render. The default is cores divided by jobs. |
| `--out DIR` | Writes builds under `DIR` instead of `lectures/.build`. |
| `--force` | Rebuilds even if the version is already built. |
| `--allow-estimate` | Allows a 1080p full render with `LECTURE_TTS=estimate`. The result can't be published. |

`--frames` and `--only` make previews in a `<version>-preview` directory.

For each lecture, the build does the following:

1. Computes the `version`: 12 hex digits of a SHA-256 over the scene, every
   file in `kit/`, `Dockerfile`, `fonts.sh`, `course.json`,
   `pronunciations.json`, the render quality, the `PIPELINE` constant in
   `build.py`, and the front matter that the title and closing slides show.
1. Skips the lecture if a finished build of that version exists. To rebuild
   it, pass `--force`.
1. Renders the scene with `--save_sections`, and checks that the last second
   of every section is still and matches the saved slide.
1. Encodes the web video (H.264 High, yuv420p, AAC 128 kb/s, a keyframe every
   2 seconds, `+faststart`), converts the slides to WebP at quality 90, writes
   WebVTT captions, and writes `dist/manifest.json`.

Manim draws each scene on one core, so the build runs several lectures at
once and gives each one's FFmpeg encoder `cores / jobs` threads. The core
count respects the container's CPU affinity and cgroup quota; to override
it, set `LECTURE_CPUS`. To use
NVIDIA's encoder on a GPU pod, set `LECTURE_ENCODER=h264_nvenc`; if FFmpeg
lacks it, the build falls back to libx264 and prints a warning.

The build exits with a non-zero status if any lecture fails, and it reports
every failure before it exits.

## Render on a Runpod pod

Rendering the whole course runs on a Runpod pod. The Mac copies the
repository to the pod, the pod renders, and the Mac copies the builds back and
publishes them. The pod never uploads to R2.

1. Copy the repository to the pod's volume:

   ```bash
   rsync -az --delete --exclude .git --exclude node_modules \
     --exclude lectures/.build --exclude lectures/.cache \
     ./ POD:/workspace/learn-inference/
   ```

   Replace `POD` with the pod's SSH destination.

1. On the pod, install the toolchain. The script is safe to re-run:

   ```bash
   bash /workspace/learn-inference/lectures/pod/setup.sh
   ```

1. On the pod, with `LECTURE_TTS_URL` and `LECTURE_TTS_TOKEN` set in its
   environment, render:

   ```bash
   bash /workspace/learn-inference/lectures/pod/render.sh --all
   ```

   Builds go to `/workspace/learn-inference/build`, the narration cache to
   `/workspace/cache/learn-inference/tts`, and logs to
   `/workspace/logs/learn-inference/`.

1. Copy the builds back to the Mac:

   ```bash
   rsync -az POD:/workspace/learn-inference/build/ ~/lecture-builds/learn-inference/
   ```

## Publish

Publishing reads finished 1080p builds from a build directory, checks them,
and writes the manifests. To see what it would do without changing anything,
run the following command:

```bash
python3 lectures/build.py --publish 02-memory-arithmetic \
  --from ~/lecture-builds/learn-inference --dry-run
```

`--dry-run` prints every upload command, every URL check, and the manifest
it would write, and it touches neither the network nor the repository.

To upload with `cf` and publish, run the following command:

```bash
python3 lectures/build.py --publish --all --from ~/lecture-builds/learn-inference --upload
```

If another process already copied the files to the bucket, omit `--upload`.
If the bucket has no public domain yet, add `--no-verify` to upload and write
the manifests without the URL checks. Run `--publish` again without it, with
`--force`, once the domain serves the bucket.
For each lecture, `--publish` does the following:

1. Finds the build whose version matches the current scene. To publish
   another finished version, pass `--version VERSION`.
1. Refuses builds that aren't 1080p, that used estimated narration, whose
   `base_url` doesn't match `course.json`, or that lack a file the manifest
   names.
1. With `--upload`, puts every file at
   `lectures/<slug>/<version>/<file>` in the `learn-inference-lectures`
   bucket, with its content type. The slides go first, then the video, then
   `manifest.json`. A key that's already public with the same size and type is
   skipped; one that differs stops the publish, because the build never
   overwrites or deletes objects.
1. Checks that every public URL returns 200 with the right content type and
   size.
1. Writes `content/lectures/<slug>.json`.

Keys contain the version, so a new version never breaks a page that still
points at the old one. The public base URL is `public_base` in `course.json`.

## The manifest

The manifest has the following shape:

```json
{
  "course": "learn-inference",
  "slug": "02-memory-arithmetic",
  "title": "Memory arithmetic",
  "version": "8cb8f12fad21",
  "base_url": "https://learn-inference-media.ifkash.dev/lectures/02-memory-arithmetic/8cb8f12fad21/",
  "video": { "src": "lecture.mp4", "duration": 69.267, "width": 1920, "height": 1080, "poster": "poster.webp" },
  "captions": "captions.vtt",
  "narration": { "mode": "remote", "model": "vibevoice-1.5b", "voice": "en-Alice_woman" },
  "sections": [
    { "id": "kv-per-token", "title": "One token's keys and values",
      "start": 10.233, "end": 32.867, "slide": "slides/02-kv-per-token.webp",
      "notes": "Narration for this slide.",
      "book_section": "s-kv-cache", "book_anchor": "s-4" }
  ]
}
```

Paths in the manifest are relative to `base_url`. Sections are contiguous:
the first starts at 0 and the last ends at `video.duration`.

`book_section` names the chapter's `##` section that a slide teaches, as a
slug of the heading text, so it survives sections being added. `book_anchor`
is the site's positional anchor for the same section (`s-1` for the first
`##` heading), which `rehypeSectionize` in
`frontend/src/lib/markdownPlugins.ts` assigns. Both are present only when the
slide names a section.

## Add a lecture

1. Write `lectures/scenes/<slug>.py`, following `AUTHORING.md`.
1. Check the layout with `LECTURE_TTS=estimate` and `--frames`, and fix every
   warning.
1. Watch the densest slides with `--quality l --only ...`.
1. Render at 1080p with the speech server, publish, and commit the scene and
   the manifest.

## Use the pipeline in another course

1. Copy `lectures/` without `.build/`, `.cache/`, `scenes/`, and
   `content/lectures/`.
1. Edit `course.json`: `course`, `name`, `chapter_path` (for example,
   `content/lessons/{slug}.md`), `manifest_path`, `bucket`, `public_base`,
   `accent`, and `fonts`. The fonts must be installed by `fonts.sh`.
1. Replace `pronunciations.json` with that course's jargon.
1. Add `lectures/.build/` and `lectures/.cache/` to `.gitignore`.
1. Run the tests:

   ```bash
   python3 -m unittest discover -s lectures/tests
   ```

   The tests use one chapter slug from this course; change `SLUG` in
   `tests/test_build.py`.

## Caches and tests

`lectures/.cache/` holds synthesised narration and `lectures/.build/` holds
renders. Git ignores both, and you can delete either at any time.

To run the tests, run the following command from the repository root:

```bash
python3 -m unittest discover -s lectures/tests -v
```

## Troubleshooting

- **`The Docker daemon is not running`**: start Docker Desktop and wait until
  `docker info` succeeds.
- **`Narration isn't cached and LECTURE_TTS_URL is not set`**: set the speech
  server, or preview with `LECTURE_TTS=estimate`.
- **`Slide 'ID' leaves the safe area`**: a mobject sits outside the frame's
  safe margins. The message names it and its extent. Use `self.place()`, or
  make the content smaller.
- **`matches 2 sentences of this narration`**: a `t.wait_until()` fragment is
  ambiguous. Use a longer fragment.
- **`book_section ids not found`**: the id doesn't match a `##` heading of the
  chapter. The message lists the valid ids.
- **`is still moving in its last second`**: something animates at the end of
  a section, usually an updater added outside the kit. End the slide on a
  still frame.
- **`Line N set as X glyphs, expected Y`**: a code line holds a character the
  code font lacks. Replace the character.
- **A render fails with a LaTeX error**: the build log
  (`lectures/.build/.../build.log`) holds the full message. In `kit.prose`,
  escape `%`, `&`, `#`, and `_`.
