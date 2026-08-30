import math

import torch


class PagedCache:
    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int,
                 head_dim: int, device="cpu", dtype=torch.float32):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_blocks = torch.zeros(shape, device=device, dtype=dtype)
        self.v_blocks = torch.zeros(shape, device=device, dtype=dtype)
        self.free = list(range(num_blocks))
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    @property
    def free_blocks(self) -> int:
        return len(self.free)

    def blocks_needed(self, seq_id: str, extra_tokens: int) -> int:
        length = self.lengths.get(seq_id, 0)
        have = len(self.tables.get(seq_id, [])) * self.block_size
        return max(0, math.ceil((length + extra_tokens - have) / self.block_size))

    def allocate(self, seq_id: str, extra_tokens: int) -> list[int]:
        table = self.tables.setdefault(seq_id, [])
        self.lengths.setdefault(seq_id, 0)
        need = self.blocks_needed(seq_id, extra_tokens)
        if need > len(self.free):
            raise RuntimeError(
                f"{seq_id} needs {need} blocks, {len(self.free)} free."
            )
        for _ in range(need):
            table.append(self.free.pop())
        self.lengths[seq_id] += extra_tokens
        return table

    def release(self, seq_id: str) -> None:
        table = self.tables.pop(seq_id, None)
        self.lengths.pop(seq_id, None)
        if table:
            self.free.extend(table)

    def slot_indices(self, seq_id: str, positions: torch.Tensor) -> torch.Tensor:
        blocks = torch.tensor(self.tables[seq_id], device=positions.device,
                              dtype=torch.long)
        return blocks[positions // self.block_size] * self.block_size + (
            positions % self.block_size
        )

    def write(self, seq_id: str, positions: torch.Tensor, k: torch.Tensor,
              v: torch.Tensor) -> None:
        slots = self.slot_indices(seq_id, positions)
        self.k_blocks.view(-1, self.num_kv_heads, self.head_dim)[slots] = k
        self.v_blocks.view(-1, self.num_kv_heads, self.head_dim)[slots] = v

    def gather(self, seq_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(self.lengths[seq_id], device=self.device)
        slots = self.slot_indices(seq_id, positions)
        kb = self.k_blocks.view(-1, self.num_kv_heads, self.head_dim)
        vb = self.v_blocks.view(-1, self.num_kv_heads, self.head_dim)
        return kb[slots], vb[slots]


def paged_decode_attention(cache: "PagedCache", seq_ids: list[str],
                           q: torch.Tensor, scale: float | None = None
                           ) -> torch.Tensor:
    heads, head_dim = q.shape[1], q.shape[2]
    scale = scale or head_dim**-0.5
    group = heads // cache.num_kv_heads

    outputs = []
    for index, seq_id in enumerate(seq_ids):
        keys, values = cache.gather(seq_id)                  # (tokens, kv_heads, dim)
        keys = keys.repeat_interleave(group, dim=1).float()  # (tokens, heads, dim)
        values = values.repeat_interleave(group, dim=1).float()

        scores = torch.einsum("hd,thd->ht", q[index].float(), keys) * scale
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", weights, values))
    return torch.stack(outputs).to(q.dtype)


def memory_waste(lengths: list[int], max_seq_len: int, block_size: int
                 ) -> tuple[float, float]:
    used = sum(lengths)
    contiguous_reserved = len(lengths) * max_seq_len
    paged_reserved = sum(math.ceil(n / block_size) * block_size for n in lengths)
    return (
        1.0 - used / contiguous_reserved,
        1.0 - used / paged_reserved,
    )
