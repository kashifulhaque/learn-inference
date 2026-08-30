# learn-inference

A web app that teaches you to build an LLM inference engine, one chapter and one
lab at a time. You write the kernels, the cache, the scheduler, and the server.
Each lab runs on a real A100 through [Modal](https://modal.com), and streams its
output back to the browser.

The target model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## What is in here

| Path | Contents |
|---|---|
| `content/chapters/` | 21 chapters, in Markdown with YAML front matter. |
| `content/labs/` | 21 labs: a starter file, a test harness, and a worked solution. |
| `engine/` | The reference engine the labs check against, and the chapters read. |
| `gpu/` | The Modal app, the RunPod worker, and the lab runner both share. |
| `backend/` | FastAPI: auth, content, progress, and the run stream. |
| `frontend/` | React and Vite: reader, editor, and dashboard. |

## The curriculum

**Part 1 — Ground truth.** What you are building, the model on disk, memory
arithmetic.

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

Modal is the preferred provider. RunPod is the fallback for when Modal credits
run out.

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

Put `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`.

The lab code and the engine reach the GPU as a Modal mount, so after editing
either one you must redeploy. Redeploying alone is not always enough: a warm
container keeps serving the previous mount until it scales down, which takes up
to five minutes. To pick up a change immediately, stop the app first:

```bash
modal app stop learn-inference --yes && modal deploy gpu/modal_app.py
```

A lab that fails on code you know you fixed is almost always this.

### RunPod

For the fallback path, see [the worker's README](gpu/runpod_worker/README.md).
Switch a single run with the provider picker in the lab pane, or set
`GPU_PROVIDER=runpod` to make it the default. When Modal reports that it is out
of credit, the app says so and offers the switch.

## Deploy with Docker Compose

```bash
docker compose up -d --build
```

The container joins the external `edge` network and answers on port 8000, which
is where a reverse proxy should send traffic. See
[Caddyfile.example](Caddyfile.example) for a Caddy block that keeps server-sent
events unbuffered, which the lab output stream needs.

The compose file also publishes `127.0.0.1:8087` so the app is reachable through
an SSH tunnel before its DNS record exists:

```bash
ssh -L 8087:127.0.0.1:8087 ifkash@vm.ifkash.dev
```

Remove that `ports` entry once the proxy is in front.

For the deployment this repository targets:

```bash
./scripts/deploy.sh
```

The script syncs the repository to the VM, keeps the existing `.env`, rebuilds,
and waits for the health check.

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

| Measurement | Result |
|---|---|
| GPU Modal serves | A100 80GB PCIe, compute capability 8.0, 108 SMs |
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

## License

MIT. See [LICENSE](LICENSE).
