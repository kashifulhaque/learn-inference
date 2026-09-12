# learn-inference

A web app that teaches you to build an LLM inference engine, one chapter and one
lab at a time. You write the kernels, the cache, the scheduler, and the server.
Each lab runs on a real A100 through [RunPod](https://runpod.io), and streams its
output back to the browser. [Modal](https://modal.com) runs the same labs, and
the lab pane can switch a single run to it.

The target model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## What is in here

| Path | Contents |
|---|---|
| `content/chapters/` | 22 chapters, in Markdown with YAML front matter. |
| `content/labs/` | 21 labs: a starter file, a test harness, and a worked solution. |
| `engine/` | The reference engine the labs check against, and the chapters read. |
| `gpu/` | The RunPod worker, the Modal app, and the lab runner both share. |
| `backend/` | FastAPI: auth, content, progress, and the run stream. |
| `frontend/` | React and Vite: reader, editor, dashboard, and compute panel. |

## The curriculum

**Part 1 — Ground truth.** What you are building, the notation and background
the rest of the course assumes, the model on disk, memory arithmetic.

**Part 2 — A forward pass.** Tokens and embeddings, RMSNorm, rotary embeddings,
the gated delta rule, grouped-query attention, assembling and validating the
stack.

**Part 3 — Making it fast.** The KV cache, the roofline, sampling.

**Part 4 — Kernels.** Your first CUDA kernel, fusion in Triton, FlashAttention,
paged attention.

**Part 5 — Serving.** Continuous batching, benchmarks that mean something.

**Part 6 — Scaling.** Quantization on Ampere, tensor parallelism, speculative
decoding.

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

   Set `SESSION_SECRET` to a random string:

   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

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

## Set up the GPU

RunPod is the default provider. Modal runs the same labs and is the alternative;
the lab pane switches a single run either way, and `GPU_PROVIDER` in `.env` sets
which one a run starts on.

### RunPod

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

### Modal

1. Install the CLI and authenticate:

   ```bash
   pip install modal && modal token new
   ```

2. Create a Modal secret named `huggingface` holding `HF_TOKEN`, so the
   functions can download weights.

3. Deploy the app:

   ```bash
   modal deploy gpu/modal_app.py
   ```

4. Optional: warm the weight cache. Every lab in the course is self-contained
   and none of them need the 27B weights, so skip this unless you want to work
   with the real model. It is a 54 GB download:

   ```bash
   modal run gpu/modal_app.py::download_model
   ```

5. Confirm the GPU is what you asked for:

   ```bash
   modal run gpu/modal_app.py
   ```

Put `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`, and set
`GPU_PROVIDER=modal` to start runs there.

The lab code and the engine reach Modal as a mount, so after editing either one
you must redeploy. Redeploying alone is not always enough: a warm container
keeps serving the previous mount until it scales down, which takes up to five
minutes. To pick up a change immediately, stop the app first:

```bash
modal app stop learn-inference --yes && modal deploy gpu/modal_app.py
```

A lab that fails on code you know you fixed is almost always this. The RunPod
worker has no equivalent trap, because the image carries the code and a new
image is a new deploy.

## The compute panel

`/compute` in the app shows what is running on both providers, and stops it. The
header carries the same count on every page, so a container nobody remembers
starting is hard to miss.

| Section | What it shows | What you can do |
|---|---|---|
| RunPod | The serverless endpoint's workers and job queue, any GPU pods, the account balance, and spend over the last day and week. | Purge the job queue. Cancel a job. Set always-on workers back to zero. Stop a pod. |
| Modal | Running containers and ephemeral apps, in every environment on the workspace. Whether the lab app is deployed. | Stop a container. Stop an app left behind by `modal run`. |
| Storage | Both providers' volumes, with the weight cache marked. | Browse a Modal volume's files. Open a RunPod network volume in the console. |
| Unfinished runs | Runs this app started and never saw finish, which is what a closed browser tab leaves behind. | Cancel the run, and the provider job with it. |

The panel never deletes anything. The serverless endpoint costs nothing while
it is idle, so it stays deployed; the volumes hold a cache that takes an hour to
refill.

RunPod has no file API for network volumes — a network volume is only readable
from a machine that mounts it — so that one links to the console instead of
listing files.

## Deploy with Docker Compose

```bash
docker compose up -d --build
```

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

For the deployment this repository targets, push first, then deploy:

```bash
git push && ./scripts/deploy.sh
```

The server holds a clone of this repository, so a deploy is a commit the server
fetches, not a directory someone copied over. The script refuses to run with a
dirty tree or an unpushed commit, resets the server's checkout to the commit
being released, rebuilds, and waits for the health check. `/api/health` reports
the commit it is running:

```bash
curl -s https://qwen.ifkash.dev/api/health
```

The output is similar to the following:

```json
{"ok":true,"chapters":22,"commit":"a1b2c3d"}
```

The server's `.env` is untracked and stays where it is; the database lives in a
Docker volume, untouched by either step. To convert a server that has no
checkout yet, or to set up a new one, run `./scripts/deploy.sh --init` once. The
clone pulls over HTTPS, so the server needs no deploy key.

## Access

The app is password protected for two people. Set `APP_PASSWORD` in `.env`.
Whoever signs in picks a display name, which scopes their progress, drafts, and
run history. The session is a signed, HTTP-only cookie.

Labs execute code you type. That code runs in the provider's sandbox, never on
the web host, but it does run with your Hugging Face token in its environment.
Do not widen access beyond people you would give that token to.

## Measured on the target hardware

Every lab's worked solution has been run on the A100 the course targets. Some of
the numbers contradict what the textbook byte counts predict, and the chapters
say so rather than rounding toward the tidy answer.

Both providers allocate whichever 80GB A100 is free, and the two variants do
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
