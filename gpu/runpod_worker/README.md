# RunPod worker

RunPod is the course's default GPU provider. This directory holds the serverless
worker that executes labs: `handler.py` receives a lab id and the reader's code,
runs it through the shared lab runner in `gpu/lab_runner.py`, and streams events
back. Modal's function in `gpu/modal_app.py` exposes the same contract.

## Build and push

Run the build from the repository root, so the context includes `engine/` and
`content/`:

```bash
docker build -f gpu/runpod_worker/Dockerfile -t REGISTRY/learn-inference-worker:latest .
```

Replace `REGISTRY` with your container registry. Then push it:

```bash
docker push REGISTRY/learn-inference-worker:latest
```

[The publish workflow](../../.github/workflows/publish-runpod-worker.yml) builds
and pushes the same image to GitHub Container Registry, so you only need to
build by hand when you want to test an image before it lands on `main`.

Unlike Modal, the worker carries the engine and the labs inside the image. A new
image is the only way a code change reaches the GPU, and there is no warm mount
to invalidate.

## Create the endpoint

1. In the RunPod console, create a serverless endpoint from the pushed image.
2. Choose an **A100 80GB** worker. The 27B model in bfloat16 is 53.8 GB, so a
   40GB part cannot hold it.
3. Attach a network volume mounted at `/models`, so weights persist when a
   worker is recycled.
4. Set the `HF_TOKEN` environment variable on the endpoint.
5. Leave the minimum worker count at zero. An idle endpoint costs nothing, and
   the compute panel can put always-on workers back to zero if one gets set.
6. Copy the endpoint id into `RUNPOD_ENDPOINT_ID` in the app's `.env`, and set
   `RUNPOD_API_KEY`.

`GPU_PROVIDER=runpod` is the default, so labs start here. To send a single run
to Modal instead, use the provider picker in the lab pane.

## What the first run costs you

A cold worker pulls the image and imports PyTorch before it runs anything, which
takes a minute or two. Later runs on a warm worker start in seconds. The compute
panel shows the endpoint's worker states, so a run that seems stuck is usually a
worker still initializing, or one RunPod has throttled for lack of capacity.
