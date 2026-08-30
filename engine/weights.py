"""Finding and loading safetensors weights.

A 27B model ships as 18 shards plus an index that maps each tensor name to its
shard. `safetensors` memory-maps those files, so opening one costs nothing and
you pay only for the tensors you actually read. That is what makes it possible
to load layer by layer instead of building a 50 GB CPU copy first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open


def local_model_path(model_id: str | None = None) -> Path:
    """Where the weights live inside a GPU container.

    Labs get a warm Hugging Face cache under $HF_HOME, populated once by
    `modal run gpu/modal_app.py::download_model`.
    """
    model_id = model_id or os.environ.get("MODEL_ID", "Qwen/Qwen3.8-27B")
    override = os.environ.get("LI_MODEL_DIR")
    if override:
        return Path(override)

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    for root in (hf_home / "hub", hf_home):
        candidate = root / f"models--{model_id.replace('/', '--')}"
        snapshots = candidate / "snapshots"
        if snapshots.is_dir():
            versions = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
            if versions:
                return versions[-1]
    raise FileNotFoundError(
        f"No local copy of {model_id}. Run "
        f"`modal run gpu/modal_app.py::download_model --model-id {model_id}` first."
    )


class WeightIndex:
    """Reads model.safetensors.index.json and serves tensors by name."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        index_file = self.dir / "model.safetensors.index.json"
        if index_file.is_file():
            self.map: dict[str, str] = json.loads(index_file.read_text())["weight_map"]
        else:
            single = "model.safetensors"
            with safe_open(self.dir / single, framework="pt") as f:
                self.map = {name: single for name in f.keys()}
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.dir / shard, framework="pt", device="cpu")
        return self._handles[shard]

    def __contains__(self, name: str) -> bool:
        return name in self.map

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.map if n.startswith(prefix))

    def get(self, name: str, device: str = "cpu", dtype: torch.dtype | None = None) -> torch.Tensor:
        shard = self.map[name]
        tensor = self._handle(shard).get_tensor(name)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    def layer_tensors(self, layer_idx: int) -> Iterator[tuple[str, str]]:
        """Yield (suffix, full name) for one decoder layer."""
        prefix = f"model.language_model.layers.{layer_idx}."
        alt = f"model.layers.{layer_idx}."
        for name in self.map:
            for candidate in (prefix, alt):
                if name.startswith(candidate):
                    yield name[len(candidate):], name

    def total_bytes(self) -> int:
        return sum(
            (self.dir / shard).stat().st_size
            for shard in {s for s in self.map.values()}
        )


def describe(model_dir: str | Path, limit: int = 40) -> list[dict[str, object]]:
    """List tensors with their shapes and dtypes. Handy for the first lab."""
    index = WeightIndex(model_dir)
    rows = []
    for name in index.names()[:limit]:
        with safe_open(index.dir / index.map[name], framework="pt") as f:
            slice_ = f.get_slice(name)
            rows.append(
                {"name": name, "shape": list(slice_.get_shape()), "dtype": slice_.get_dtype()}
            )
    return rows
