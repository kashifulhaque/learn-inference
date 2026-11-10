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
    arrival: float = 0.0   # when the request reaches the server, in ms
    priority: int = 0      # higher is admitted first; ties go first come, first served

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

    def _head_index(self) -> int:
        return max(range(len(self.waiting)),
                   key=lambda i: (self.waiting[i].priority, -i))

    def preempt(self) -> None:
        if not self.running:
            return
        # The lowest priority goes first. Among equals, the newest goes first.
        index = max(range(len(self.running)),
                    key=lambda i: (-self.running[i].priority, i))
        victim = self.running.pop(index)
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

        # 4: admit from the head of the queue. The head is the waiting request
        # with the highest priority, and the earliest position among equals.
        while self.waiting and budget > 0 and len(self.running) < self.max_batch_size:
            index = self._head_index()
            candidate = self.waiting[index]
            chunk = min(self.chunk_size, budget, candidate.prompt_len)
            if not self._reserve(candidate, chunk):
                break
            del self.waiting[index]
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


def percentile(values, q):
    """Linear-interpolation percentile, the definition in chapter 17."""
    x = sorted(values)
    r = q / 100 * (len(x) - 1)
    i = int(r)
    if i + 1 >= len(x):
        return x[-1]
    return x[i] + (r - i) * (x[i + 1] - x[i])


def replay(scheduler, requests, overhead=2.0, per_token=0.125):
    """Replay an arrival trace on a simulated clock and report latency.

    A step that carries `n` tokens takes `overhead + per_token * n` ms. The
    clock starts at 0. Before each step, every request whose `arrival` is at or
    before the clock joins the scheduler. When nothing is waiting or running,
    the clock jumps to the next arrival. A token's time is the clock at the end
    of the step that produced it, and a request's first token is the one its
    last prefill chunk produces. The trace must not trigger preemption.
    """
    pending = sorted(requests, key=lambda r: r.arrival)
    clock = 0.0
    token_times = {r.id: [] for r in requests}
    order = []
    while pending or scheduler.has_work:
        while pending and pending[0].arrival <= clock:
            scheduler.add(pending.pop(0))
        if not scheduler.has_work:
            clock = pending[0].arrival
            continue
        batch = scheduler.step()
        if batch.empty:
            # The step that retires the last finished request returns an empty
            # batch. Loop again to jump to the next arrival, if there is one.
            if scheduler.has_work:
                break
            continue
        clock += overhead + per_token * batch.token_count
        for request, chunk in batch.prefill:
            if request.id not in order:
                order.append(request.id)
            scheduler.commit_prefill(request, chunk)
            if not request.needs_prefill:
                scheduler.commit_token(request)
                token_times[request.id].append(clock)
        for request in batch.decode:
            scheduler.commit_token(request)
            token_times[request.id].append(clock)

    ttft, itl, gaps = {}, {}, {}
    for request in requests:
        times = token_times[request.id]
        ttft[request.id] = times[0] - request.arrival
        gaps[request.id] = [b - a for a, b in zip(times, times[1:])]
        if len(times) >= 2:
            itl[request.id] = (times[-1] - times[0]) / (len(times) - 1)
    pooled = [g for gs in gaps.values() for g in gs]
    return {
        "ttft": ttft,
        "itl": itl,
        "gaps": gaps,
        "order": order,
        "makespan": clock,
        "p99_ttft": percentile(list(ttft.values()), 99),
        "p99_itl": percentile(pooled, 99),
        "max_gap": max(pooled),
    }
