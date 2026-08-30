# RunPod serverless worker. Mirrors the Modal image so labs behave identically.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    LI_LABS_DIR=/repo/content/labs \
    LI_ENGINE_DIR=/repo

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==2.1.3 \
        safetensors==0.6.2 \
        transformers==4.57.1 \
        tokenizers==0.22.1 \
        "huggingface_hub[hf_transfer]==0.35.3" \
        triton==3.4.0 \
        einops==0.8.1 \
        sentencepiece==0.2.1 \
        runpod==1.7.7

WORKDIR /repo
COPY engine /repo/engine
COPY content/labs /repo/content/labs
COPY gpu/harness.py /repo/gpu/harness.py
COPY gpu/lab_runner.py /repo/gpu/lab_runner.py
COPY gpu/runpod_worker/handler.py /repo/gpu/handler.py

CMD ["python", "-u", "/repo/gpu/handler.py"]
