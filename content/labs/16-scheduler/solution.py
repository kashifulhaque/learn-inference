from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    id: str
    prompt_len: int
    max_new_tokens: int
    prefilled: int = 0
    generated: int = 0
    finished: bool = False

    @property
    def needs_prefill(self) -> bool:
        return self.prefilled < self.prompt_len


@dataclass
class Batch:
    prefill: list = field(default_factory=list)
    decode: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def token_count(self) -> int:
        return sum(chunk for _, chunk in self.prefill) + len(self.decode)


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = num_blocks
        self.held: dict[str, int] = {}
        self.lengths: dict[str, int] = {}

    def blocks_needed(self, seq_id: str, extra: int) -> int:
        length = self.lengths.get(seq_id, 0)
        have = self.held.get(seq_id, 0) * self.block_size
        return max(0, -(-(length + extra - have) // self.block_size))

    def allocate(self, seq_id: str, extra: int) -> bool:
        need = self.blocks_needed(seq_id, extra)
        if need > self.free:
            return False
        self.free -= need
        self.held[seq_id] = self.held.get(seq_id, 0) + need
        self.lengths[seq_id] = self.lengths.get(seq_id, 0) + extra
        return True

    def release(self, seq_id: str) -> None:
        self.free += self.held.pop(seq_id, 0)
        self.lengths.pop(seq_id, None)


class Scheduler:
    def __init__(self, pool: BlockPool, max_batch_size: int = 32,
                 max_batched_tokens: int = 2048, chunk_size: int = 512,
                 watermark: float = 0.02):
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.max_batched_tokens = max_batched_tokens
        self.chunk_size = chunk_size
        self.watermark = watermark

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.preemptions = 0

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _reserve(self, request: Request, tokens: int) -> bool:
        reserve = int(self.pool.num_blocks * self.watermark)
        need = self.pool.blocks_needed(request.id, tokens)
        if need > max(0, self.pool.free - reserve):
            return False
        return self.pool.allocate(request.id, tokens)

    def preempt(self) -> None:
        if not self.running:
            return
        victim = self.running.pop()
        self.pool.release(victim.id)
        victim.prefilled = 0
        victim.generated = 0
        self.waiting.appendleft(victim)
        self.preemptions += 1

    def step(self) -> Batch:
        # Nothing schedulable while sequences are resident means the cache is
        # full. Preempt and retry rather than returning an empty batch and
        # deadlocking.
        for _ in range(self.max_batch_size + 1):
            batch = self._schedule()
            if not batch.empty or not self.running:
                return batch
            self.preempt()
        return batch

    def _schedule(self) -> Batch:
        batch = Batch()
        budget = self.max_batched_tokens

        # 1 and 2: retire what is done, then schedule decodes.
        still_running = []
        for request in self.running:
            if request.finished:
                self.pool.release(request.id)
                self.finished.append(request)
                continue
            if request.needs_prefill:
                still_running.append(request)
                continue
            if len(batch.decode) >= self.max_batch_size or budget < 1:
                still_running.append(request)
                continue
            if not self._reserve(request, 1):
                self.running = still_running
                self.preempt()
                still_running = self.running
                still_running.append(request)
                continue
            batch.decode.append(request)
            budget -= 1
            still_running.append(request)
        self.running = still_running

        # 3: continue prefilling what is already admitted.
        for request in list(self.running):
            if budget <= 0:
                break
            if not request.needs_prefill:
                continue
            chunk = min(self.chunk_size, budget, request.prompt_len - request.prefilled)
            if chunk > 0 and self._reserve(request, chunk):
                batch.prefill.append((request, chunk))
                budget -= chunk

        # 4: admit from the head of the queue.
        while self.waiting and budget > 0 and len(self.running) < self.max_batch_size:
            candidate = self.waiting[0]
            chunk = min(self.chunk_size, budget, candidate.prompt_len)
            if not self._reserve(candidate, chunk):
                break
            self.waiting.popleft()
            self.running.append(candidate)
            batch.prefill.append((candidate, chunk))
            budget -= chunk

        return batch

    def commit_prefill(self, request: Request, tokens: int) -> None:
        request.prefilled += tokens

    def commit_token(self, request: Request) -> None:
        request.generated += 1
        if request.generated >= request.max_new_tokens:
            request.finished = True
