# learn-inference

A web app that teaches you to build an LLM inference engine, one chapter and one
lab at a time. You write the kernels, the cache, the scheduler, and the server.
Each lab runs on a real A100 through [RunPod](https://runpod.io), and streams its
output back to the browser.

The target model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## What is in here

| Path | Contents |
|---|---|
| `content/chapters/` | 40 chapters, in Markdown with YAML front matter. |
| `content/labs/` | 39 labs: a starter file, a test harness, and a worked solution. |
| `engine/` | The reference engine the labs check against, and the chapters read. |
| `gpu/` | The RunPod worker and the lab runner it uses. |
| `backend/` | FastAPI: auth, content, progress, and the run stream. |
| `frontend/` | React and Vite: reader, editor, dashboard, and compute panel. |

## The curriculum

**Part 1 — Ground truth.** What you are building, the notation and background
the rest of the course assumes, the model on disk, memory arithmetic, and inference-aware scaling laws and
model design.

**Part 2 — A forward pass.** Tokens and embeddings, RMSNorm, rotary embeddings,
the gated delta rule, grouped-query attention, the KV-economy variants (MLA, sliding windows,
attention sinks, and eviction), assembling and validating the stack.

**Part 3 — Making it fast.** The KV cache, the roofline, CUDA graphs and launch overhead, profiling and performance debugging, sampling, constrained decoding.

**Part 4 — Kernels.** Your first CUDA kernel, fusion in Triton, skinny decode
GEMMs, FlashAttention, split-KV decode attention, paged attention, and prefix
caching.

**Part 5 — Serving.** Continuous batching, the serving loop, scheduling policy and routing, benchmarks that mean something, a model of step time and goodput, and disaggregated prefill and decode.

**Part 6 — Scaling.** Quantization on Ampere, int4 weights and KV quantization, sparsity and pruning,
multi-LoRA serving, tensor parallelism, mixture-of-experts serving and expert parallelism, pipeline
schedules and ring attention, speculative decoding, and tree speculation with speculation under load.

## Why this model is interesting

Qwen3.8-27B is not a stack of identical transformer blocks. Of its 64 layers,
only every fourth is full attention; the other 48 use gated delta linear
attention, which keeps a fixed-size recurrent state instead of a growing cache.

| | Layers | State per sequence | Cost per token |
|---|---|---|---|
| Full attention, grouped-query | 16 | Grows with context | 64 KiB total |
| Gated delta linear attention | 48 | 147.8 MiB, fixed | 0 |

Past about 788 tokens of context the hybrid uses less memory than an
all-full-attention model would, and the gap widens without limit. At 32k context
it needs 2.14 GiB per sequence against 8.0 GiB.

The weights are 53.8 GB in bfloat16, so the labs need an **A100 80GB**. A 40GB
part cannot hold the model at all.

## Run it locally

You need Python 3.12 or later and Node 20 or later.

1. Copy the environment file and fill it in:

   ```bash
   cp .env.example .env
   ```

   Set `SSO_PROXY_SECRET` to the same high-entropy value configured in Caddy:

   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   If an Authelia username already has progress under another display name,
   preserve it with `SSO_USERS=ifkash:enderboi,ananya:michu`.

2. Install and start the backend:

   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
   .venv/bin/uvicorn backend.app.main:app --reload --port 8000
   ```

3. In another terminal, start the frontend:

   ```bash
   cd frontend && npm install && npm run dev
   ```

Open http://localhost:5173. The Vite dev server proxies `/api` to the backend.

The development frontend has no local login bypass: to use private pages, serve
it behind a trusted reverse proxy that sets the SSO headers after Authelia
authorization, with the same `SSO_PROXY_SECRET` configured in the backend.

## Set up the GPU

Labs run on a RunPod serverless endpoint.

1. Build the worker image and push it. Run this from the repository root, so the
   build context includes `engine/` and `content/`:

   ```bash
   docker build -f gpu/runpod_worker/Dockerfile -t REGISTRY/learn-inference-worker:latest .
   ```

   Replace `REGISTRY` with your container registry. Then push it:

   ```bash
   docker push REGISTRY/learn-inference-worker:latest
   ```

   Pushing to GitHub Container Registry happens on its own: see
   [the publish workflow](.github/workflows/publish-runpod-worker.yml).

2. In the RunPod console, create a serverless endpoint from that image, on an
   **A100 80GB** worker.

3. Attach a network volume mounted at `/models`, so the weight cache survives a
   worker being recycled.

4. Set `HF_TOKEN` on the endpoint, so the worker can download weights.

5. Put `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in `.env`.

For more detail, see [the worker's README](gpu/runpod_worker/README.md).

The endpoint costs nothing while it is idle, which is why the compute panel
never offers to delete it.

## The compute panel

`/compute` in the app shows what is running on RunPod, and stops it. The
header carries the same count on every page, so a container nobody remembers
starting is hard to miss.

| Section | What it shows | What you can do |
|---|---|---|
| RunPod | The serverless endpoint's workers and job queue, any GPU pods, the account balance, and spend over the last day and week. | Purge the job queue. Cancel a job. Set always-on workers back to zero. Stop a pod. |
| Storage | RunPod network volumes, with the weight cache marked. | Open a network volume in the console. |
| Unfinished runs | Runs this app started and never saw finish, which is what a closed browser tab leaves behind. | Cancel the run, and the provider job with it. |

The panel never deletes anything. The serverless endpoint costs nothing while
it is idle, so it stays deployed; the volumes hold a cache that takes an hour to
refill.

RunPod has no file API for network volumes—a network volume is only readable
from a machine that mounts it—so the panel links to the console instead of
listing files.

## Deploy with Docker Compose

```bash
docker compose pull && docker compose up -d
```

The compose file runs the prebuilt image
`ghcr.io/kashifulhaque/learn-inference:latest`. The host never builds from
source and holds no copy of this repository's code. Put a `.env` file next to
the compose file; see [.env.example](.env.example).

The container joins the external `edge` network and answers on port 8000, which
is where a reverse proxy should send traffic. See
[Caddyfile.example](Caddyfile.example) for a Caddy block that keeps server-sent
events unbuffered, which the lab output stream needs.

The deployment this repository targets runs at
[qwen.ifkash.dev](https://qwen.ifkash.dev).

The compose file also publishes `127.0.0.1:8087`, so the app stays reachable
through an SSH tunnel if the proxy or its certificate is ever the problem:

```bash
ssh -L 8087:127.0.0.1:8087 ifkash@vm.ifkash.dev
```

### How a release reaches the server

To release a change, merge it to `main`. The `build-and-deploy` workflow
(`.github/workflows/deploy.yml`) then does the following:

1. Builds the image on GitHub Actions, including the PDF book, and pushes it to
   GHCR as `latest` and as `sha-COMMIT`.
2. Connects to the server and runs `deploy learn-inference`, which pulls the new
   image and restarts the container.

What runs is therefore an image built from a commit you can name. `/api/health`
reports the commit it was built from:

```bash
curl -s https://qwen.ifkash.dev/api/health
```

The output is similar to the following:

```json
{"ok":true,"chapters":29,"commit":"a1b2c3d","book_pages":498}
```

The server's `.env` is untracked and stays where it is; the database lives in a
Docker volume, untouched by a deploy.

## The PDF book

The sidebar's **Course PDF** link downloads the whole course as one book: a
cover, a contents page, every chapter with its objectives and sections, every
lab's brief, starter file, and hints, and the reference solutions in an
appendix. The PDF has a bookmark for every chapter and section, running heads,
and live links between chapters, labs, and solutions.

`frontend/scripts/build-book.mjs` renders the print layout in
`frontend/src/book/` with the site's own Markdown pipeline, then prints it with
headless Chrome until the page numbers stop moving. The build fails on maths
that KaTeX can't render and on anything wider than the page. To build the book
locally, run the following command:

```bash
node frontend/scripts/build-book.mjs --force
```

The PDF lands in `book/`, which is untracked. The script finds Chrome through
`CHROME_PATH` or in the usual install locations, and without `--force` it skips
the build when no chapter, lab, or layout file has changed.

The Docker image builds the book in its own `book` stage and copies only the
PDF and its manifest into `/app/book`, so every image carries the PDF that
matches its chapters. Chromium and the build's dependencies stay out of the
final image. The manifest's `commit` is `unknown` in the image, because the
build context has no `.git` directory. A checkout without a built book shows no
download link. The download requires an authenticated Authelia identity, like
the chapters.

## Access

The site uses the central Authelia portal at https://auth.ifkash.dev; it has no
site-specific password or local login session. When a visitor has no trusted
identity, the app sends them to the portal with their full current URL as the
encoded `rd` return destination.

The deployment must route the site through Caddy `forward_auth` to Authelia,
remove any client-provided `Remote-User` and `X-SSO-Proxy` headers, and set
those headers only after Authelia authorizes the request. Set
`SSO_PROXY_SECRET` in the app environment and in Caddy's trusted-header
configuration to the same private value. Requests without both the correct
secret and a non-empty `Remote-User` cannot access private API routes or the
PDF. Never expose the app port directly to untrusted clients.

`SSO_USERS` optionally maps Authelia usernames to existing progress, drafts,
and run-history identities, for example `ifkash:enderboi,ananya:michu`.
Unmapped Authelia usernames use their own username as the data identity.

Labs execute code you type. That code runs on the RunPod worker, never on
the web host, but it does run with your Hugging Face token in its environment.
Do not widen access beyond people you would give that token to.

## Measured on the target hardware

Every lab's worked solution has been run on the A100 the course targets. Some of
the numbers contradict what the textbook byte counts predict, and the chapters
say so rather than rounding toward the tidy answer.

RunPod allocates whichever 80GB A100 is free, and the two variants do
not have the same memory bandwidth: the SXM4 module is rated at 2039 GB/s and
the PCIe card at 1935. Runs land on either, so timings move a little between
them. The numbers below were taken on the PCIe card; chapter 0's lab prints
which one you got.

| Measurement | Result |
|---|---|
| GPU | A100 80GB, compute capability 8.0, 108 SMs |
| Device-to-device copy | 1275 GB/s, against a rating of 1935 |
| Hand-written CUDA vector add | 1304 GB/s — the same, because there is nothing to beat |
| Coalesced against strided access | 6.9x |
| Triton RMSNorm | 945 GB/s, 74% of the copy ceiling |
| Fused RMSNorm and residual, past L2 | 1.10x |
| Fused RMSNorm and residual, within L2 | 0.92x — the saved traffic never left cache |
| Fused SwiGLU | 1.52x |
| Triton FlashAttention against the vendor kernel | 0.63x at 8k tokens |
| FlashAttention peak memory against naive | 27.5x less |
| Chunked delta rule against the sequential loop | agrees to 2e-6 |

The fusion result is the interesting one. The byte count predicts a 20% saving
either way, and measuring across sizes shows the saving only appears once the
intermediate stops fitting in the A100's 40 MB L2. Chapter 13 works through it.

## Check the labs

Every lab's worked solution must pass that lab's own tests:

```bash
python3 scripts/check_labs.py
```

Labs that need CUDA are skipped on a machine without a GPU and reported as
skipped. To run those, use a GPU host, or open the lab in the app.

### Performance thresholds

By default a lab's tests grade correctness only, and the metrics in `meta.json`
are informational. A lab can also grade a speed claim by adding an optional
`perf_threshold` block to its `meta.json`:

```json
"metrics": ["swiglu_speedup", "achieved_gbs"],
"perf_threshold": {
  "swiglu_speedup": {"min": 1.4},
  "achieved_gbs": {"min": 900}
},
"perf_tolerance": 0.05
```

Each entry names a metric from `metrics` and sets exactly one of `min` or `max`.
Because RunPod can give you either A100 variant (see
[Measured on the target hardware](#measured-on-the-target-hardware)), timings
differ by a few percent, so the check applies a tolerance, 0.05 by default:
`min` passes when the metric is at least `min * (1 - tolerance)`, and `max`
passes when it is at most `max * (1 + tolerance)`. A metric that is missing or
not a number, for example because the lab skipped it on a CPU, shows as a
skipped line and does not fail, so CPU runs and CI stay green.
`scripts/check_labs.py` validates the block and enforces it for the reference
solution, and the GPU worker enforces it for learner submissions.

## Write a chapter

Chapters use a few reading aids on top of Markdown: callouts such as
`> [!KEY]`, collapsible deep dives and self-check questions, `==highlights==`,
and colour macros for maths symbols. For the syntax and the shape every chapter
follows, see [the chapter style guide](content/STYLE.md).

To check each chapter's callouts and outline, run the following command:

```bash
python3 scripts/check_chapters.py
```

## Check the maths

The chapters write their maths as LaTeX, which the reader renders with KaTeX.
KaTeX supports a subset of LaTeX and fails silently in red where it does not, so
check every expression against the same build the site uses:

```bash
node scripts/check_math.mjs
```

The script reports the file, the line, and the parse error for anything that
does not render. Pass file paths to check only those.

## License

MIT. See [LICENSE](LICENSE).
