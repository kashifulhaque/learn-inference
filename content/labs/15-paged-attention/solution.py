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
        self.refs = [0] * num_blocks
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    @property
    def free_blocks(self) -> int:
        return len(self.free)

    def blocks_needed(self, seq_id: str, extra_tokens: int) -> int:
        length = self.lengths.get(seq_id, 0)
        have = len(self.tables.get(seq_id, [])) * self.block_size
        return max(0, math.ceil((length + extra_tokens - have) / self.block_size))

    def refcount(self, block: int) -> int:
        return self.refs[block]

    def _needs_cow(self, seq_id: str) -> bool:
        """True when the next token would write into a shared, partly full tail."""
        table = self.tables.get(seq_id)
        if not table or self.lengths[seq_id] % self.block_size == 0:
            return False
        return self.refs[table[-1]] > 1

    def allocate(self, seq_id: str, extra_tokens: int) -> list[int]:
        table = self.tables.setdefault(seq_id, [])
        self.lengths.setdefault(seq_id, 0)
        need = self.blocks_needed(seq_id, extra_tokens)
        cow = extra_tokens > 0 and self._needs_cow(seq_id)
        if need + int(cow) > len(self.free):
            raise RuntimeError(
                f"{seq_id} needs {need + int(cow)} blocks, {len(self.free)} free."
            )
        if cow:
            self.copy_on_write(seq_id)
        for _ in range(need):
            block = self.free.pop()
            self.refs[block] = 1
            table.append(block)
        self.lengths[seq_id] += extra_tokens
        return table

    def fork(self, seq_id: str, new_seq_id: str) -> list[int]:
        """Share every block of `seq_id` with `new_seq_id`. No block is copied."""
        if new_seq_id in self.tables:
            raise ValueError(f"{new_seq_id} already exists.")
        table = list(self.tables[seq_id])
        for block in table:
            self.refs[block] += 1
        self.tables[new_seq_id] = table
        self.lengths[new_seq_id] = self.lengths[seq_id]
        return table

    def copy_on_write(self, seq_id: str) -> int:
        """Give `seq_id` a private tail block. A tail with one owner stays put."""
        table = self.tables[seq_id]
        old = table[-1]
        if self.refs[old] <= 1:
            return old
        if not self.free:
            raise RuntimeError(f"{seq_id} needs 1 block to copy its tail, 0 free.")
        new = self.free.pop()
        self.k_blocks[new] = self.k_blocks[old]
        self.v_blocks[new] = self.v_blocks[old]
        self.refs[old] -= 1
        self.refs[new] = 1
        table[-1] = new
        return new

    def append_token(self, seq_id: str, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one token. k, v: (num_kv_heads, head_dim)."""
        position = self.lengths.get(seq_id, 0)
        self.allocate(seq_id, 1)
        self.write(seq_id, torch.tensor([position], device=k.device), k[None], v[None])

    def release(self, seq_id: str) -> None:
        table = self.tables.pop(seq_id, None)
        self.lengths.pop(seq_id, None)
        for block in table or []:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                self.free.append(block)

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


def preemption_cost(num_tokens: int, kv_bytes_per_token: int, block_size: int,
                    num_params: int, state_bytes: int = 0) -> dict[str, int]:
    """Bytes moved and FLOPs spent to preempt one sequence, by swap or recompute.

    Swap copies whole blocks, so the KV bytes round up to a block boundary. It
    moves them out to host memory and back, plus any recurrent state.
    Recompute moves nothing and redoes the prefill, about 2 * params * tokens.
    """
    blocks = -(-num_tokens // block_size)
    one_way = blocks * block_size * kv_bytes_per_token + state_bytes
    return {
        "swap_bytes_out": one_way,
        "swap_bytes_in": one_way,
        "swap_bytes_total": 2 * one_way,
        "recompute_bytes": 0,
        "recompute_flops": 2 * num_params * num_tokens,
    }
