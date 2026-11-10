"""Lab 18c — multi-LoRA serving, reference solution."""

from __future__ import annotations

import torch

MLP_SHAPES = ((5120, 17408), (5120, 17408), (17408, 5120))
LAYERS = 64
PCIE_GBS = 32.0


def lora_forward(x, W, A, B, alpha):
    r = A.shape[1]
    return x @ W + (x @ A) @ B * (alpha / r)


def merge_lora(W, A, B, alpha):
    r = A.shape[1]
    return W + (A @ B) * (alpha / r)


def bgmv(x, A_stack, B_stack, adapter_ids, alpha):
    """Gathered LoRA delta: one gather per token, then two batched products."""
    r = A_stack.shape[2]
    ids = adapter_ids.to(torch.int64)
    live = ids >= 0
    safe = torch.where(live, ids, torch.zeros_like(ids))
    a = A_stack[safe]  # (T, d_in, r)
    b = B_stack[safe]  # (T, r, d_out)
    low = torch.bmm(x.unsqueeze(1), a)  # (T, 1, r)
    out = torch.bmm(low, b).squeeze(1) * (alpha / r)
    return out * live.unsqueeze(1).to(out.dtype)


def segment_by_adapter(adapter_ids):
    """Return (order, segments). `order` is a stable argsort of the ids and
    `segments` lists (adapter, start, end) over the sorted positions, skipping -1."""
    ids = adapter_ids.to(torch.int64)
    order = torch.argsort(ids, stable=True)
    sorted_ids = ids[order].tolist()
    segments = []
    start = 0
    for i in range(1, len(sorted_ids) + 1):
        if i == len(sorted_ids) or sorted_ids[i] != sorted_ids[start]:
            if sorted_ids[start] >= 0:
                segments.append((sorted_ids[start], start, i))
            start = i
    return order, segments


def sgmv(x, A_stack, B_stack, adapter_ids, alpha):
    """Segmented LoRA delta: sort by adapter, one pair of matmuls per segment."""
    r = A_stack.shape[2]
    order, segments = segment_by_adapter(adapter_ids)
    xs = x[order]
    out_sorted = torch.zeros(x.shape[0], B_stack.shape[2], dtype=x.dtype)
    for adapter, start, end in segments:
        out_sorted[start:end] = (xs[start:end] @ A_stack[adapter]) @ B_stack[adapter] * (alpha / r)
    out = torch.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def lora_cost(d_in, d_out, rank, bytes_per=2):
    base_flops = 2 * d_in * d_out
    lora_flops = 2 * rank * (d_in + d_out)
    base_bytes = bytes_per * d_in * d_out
    lora_bytes = bytes_per * rank * (d_in + d_out)
    return {
        "base_flops": base_flops,
        "lora_flops": lora_flops,
        "base_bytes": base_bytes,
        "lora_bytes": lora_bytes,
        "flops_ratio": lora_flops / base_flops,
        "bytes_ratio": lora_bytes / base_bytes,
    }


def adapter_bytes(rank, shapes=MLP_SHAPES, layers=LAYERS, bytes_per=2):
    return layers * sum(bytes_per * rank * (d_in + d_out) for d_in, d_out in shapes)


def adapter_load_ms(nbytes, link_gbs=PCIE_GBS):
    return nbytes / (link_gbs * 1e9) * 1e3


def simulate_adapter_pool(trace, sizes, capacity_bytes, policy, link_gbs=PCIE_GBS):
    resident = {}  # adapter -> [load_seq, last_use]
    count = {}
    used = 0
    seq = 0
    hits = misses = evictions = loaded = 0
    for step, batch in enumerate(trace, start=1):
        distinct = list(dict.fromkeys(batch))
        pinned = set(distinct)
        for a in distinct:
            count[a] = count.get(a, 0) + 1
        for a in distinct:
            if a in resident:
                hits += 1
                resident[a][1] = step
                continue
            misses += 1
            size = sizes[a]
            while used + size > capacity_bytes:
                cands = [k for k in resident if k not in pinned]
                if not cands:
                    raise ValueError("batch does not fit in the pool")
                if policy == "fifo":
                    key = lambda k: (resident[k][0],)
                elif policy == "lru":
                    key = lambda k: (resident[k][1], resident[k][0])
                elif policy == "freq":
                    key = lambda k: (count[k], resident[k][1], resident[k][0])
                else:
                    raise ValueError(f"unknown policy {policy!r}")
                victim = min(cands, key=key)
                used -= sizes[victim]
                del resident[victim]
                evictions += 1
            seq += 1
            resident[a] = [seq, step]
            used += size
            loaded += size
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "bytes_loaded": loaded,
        "load_ms": loaded / (link_gbs * 1e9) * 1e3,
    }
