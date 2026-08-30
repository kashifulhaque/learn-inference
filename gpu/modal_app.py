"""Modal app that executes labs on an A100.

Deploy with:

    modal deploy gpu/modal_app.py

Then warm the weight cache once (it is a ~54 GB download):

    modal run gpu/modal_app.py::download_model
"""

import os
from pathlib import Path
from typing import Any, Iterator

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "learn-inference")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.8-27B")
SMALL_MODEL_ID = os.environ.get("SMALL_MODEL_ID", "Qwen/Qwen3-0.6B")
GPU = os.environ.get("MODAL_GPU", "A100-80GB")

REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

# Weights are big and immutable, so they live in a Volume that survives across
# containers. /models is the HF cache root inside every function.
models_volume = modal.Volume.from_name("learn-inference-models", create_if_missing=True)
MODELS_PATH = "/models"

# A CUDA *devel* base, not a runtime one. Chapter 12 compiles CUDA C++ at run
# time with torch.utils.cpp_extension.load_inline, which needs nvcc and the CUDA
# headers. The PyTorch wheels bundle the runtime libraries but not the compiler.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.8.0",
        "numpy==2.1.3",
        "safetensors==0.6.2",
        "transformers==4.57.1",
        "tokenizers==0.22.1",
        "huggingface_hub[hf_transfer]==0.35.3",
        "triton==3.4.0",
        "einops==0.8.1",
        "sentencepiece==0.2.1",
    )
    .env(
        {
            "HF_HOME": MODELS_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "LI_LABS_DIR": "/repo/content/labs",
            "LI_ENGINE_DIR": "/repo",
            "MODEL_ID": MODEL_ID,
            "SMALL_MODEL_ID": SMALL_MODEL_ID,
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            # Ampere is compute capability 8.0. Naming it keeps nvcc from
            # building for every architecture it knows, which is slow.
            "TORCH_CUDA_ARCH_LIST": "8.0",
        }
    )
    .add_local_dir(REPO_ROOT / "engine", "/repo/engine")
    .add_local_dir(REPO_ROOT / "content" / "labs", "/repo/content/labs")
    .add_local_file(REPO_ROOT / "gpu" / "harness.py", "/repo/gpu/harness.py")
    .add_local_file(REPO_ROOT / "gpu" / "lab_runner.py", "/repo/gpu/lab_runner.py")
)

hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


def _stream(lab_id: str, code: str, timeout: int) -> Iterator[dict[str, Any]]:
    import sys

    sys.path.insert(0, "/repo/gpu")
    from lab_runner import run_lab_stream

    yield from run_lab_stream(lab_id=lab_id, code=code, timeout=timeout)


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODELS_PATH: models_volume},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=300,
)
def run_lab(
    lab_id: str, code: str, gpu: str = GPU, timeout: int = 900
) -> Iterator[dict[str, Any]]:
    """Run a GPU lab and stream its output."""
    yield from _stream(lab_id, code, timeout)


@app.function(
    image=image,
    volumes={MODELS_PATH: models_volume},
    secrets=[hf_secret],
    timeout=1800,
    cpu=4,
    memory=8192,
)
def run_lab_cpu(
    lab_id: str, code: str, gpu: str = "cpu", timeout: int = 600
) -> Iterator[dict[str, Any]]:
    """Run a lab that needs no GPU. Cheaper, and it starts faster."""
    yield from _stream(lab_id, code, timeout)


@app.function(
    image=image,
    volumes={MODELS_PATH: models_volume},
    secrets=[hf_secret],
    timeout=7200,
    cpu=8,
    memory=16384,
)
def download_model(model_id: str = MODEL_ID, allow_patterns: list[str] | None = None) -> str:
    """Pull weights into the Volume so labs do not each pay the download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        cache_dir=MODELS_PATH,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    models_volume.commit()
    print(f"Downloaded {model_id} to {path}")
    return path


@app.function(image=image, gpu=GPU, volumes={MODELS_PATH: models_volume}, timeout=600)
def gpu_report() -> dict[str, Any]:
    """A quick sanity check that the GPU is what we asked for."""
    import subprocess

    import torch

    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1024**3, 1),
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "nvidia_smi": smi,
    }


@app.local_entrypoint()
def main() -> None:
    print(gpu_report.remote())
