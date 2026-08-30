# RunPod fallback worker

Modal is the preferred provider. Use this only when Modal credits run out.

## Build and push

Run from the repository root so the build context includes `engine/` and
`content/`:

```bash
docker build -f gpu/runpod_worker/Dockerfile -t <registry>/learn-inference-worker:latest .
docker push <registry>/learn-inference-worker:latest
```

## Create the endpoint

1. In the RunPod console, create a serverless endpoint from the pushed image.
2. Choose an **A100 80GB** worker.
3. Attach a network volume mounted at `/models` so weights persist between
   workers.
4. Set the `HF_TOKEN` environment variable on the endpoint.
5. Copy the endpoint id into `RUNPOD_ENDPOINT_ID` in the app's `.env`, and set
   `RUNPOD_API_KEY`.

Switch a run to RunPod from the provider picker in the lab pane, or set
`GPU_PROVIDER=runpod` to make it the default.
