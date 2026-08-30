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

4. Warm the weight cache once. This is a 54 GB download and takes a while:

   ```bash
   modal run gpu/modal_app.py::download_model
   ```

5. Confirm the GPU is what you asked for:

   ```bash
   modal run gpu/modal_app.py
   ```

Put `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in `.env`.

### RunPod

For the fallback path, see [the worker's README](gpu/runpod_worker/README.md).
Switch a single run with the provider picker in the lab pane, or set
`GPU_PROVIDER=runpod` to make it the default. When Modal reports that it is out
of credit, the app says so and offers the switch.

## Deploy with Docker Compose

```bash
docker compose up -d --build
```

The app listens on `127.0.0.1:8087`. Put a reverse proxy in front of it; see
[Caddyfile.example](Caddyfile.example) for a configuration that keeps
server-sent events unbuffered, which the lab output stream needs.

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

## Check the labs

Every lab's worked solution must pass that lab's own tests:

```bash
python3 scripts/check_labs.py
```

Labs that need CUDA are skipped on a machine without a GPU and reported as
skipped. To run those, use a GPU host, or open the lab in the app.

## License

MIT. See [LICENSE](LICENSE).
