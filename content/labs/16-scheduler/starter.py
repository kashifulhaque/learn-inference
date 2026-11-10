"""Lab 16 — continuous batching.

Implement `step`. It runs before every forward pass and decides what that pass
contains, under a token budget and a block budget. Then implement `percentile`
and `replay`, which replay an arrival trace on a simulated clock and report
time to first token (TTFT) and inter-token latency (ITL).
"""

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
    prefill: list = field(default_factory=list)   # (request, chunk_size) pairs
    decode: list = field(default_factory=list)    # requests

    @property
    def empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def token_count(self) -> int:
        return sum(chunk for _, chunk in self.prefill) + len(self.decode)


class BlockPool:
    """A stand-in for the paged cache, counting blocks only."""

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

    def step(self) -> Batch:
        """Choose the work for the next forward pass.

        Do these in order:

        1. Retire finished requests, releasing their blocks.
        2. Schedule decodes for running requests that have finished prefilling,
           within the batch size and token budget. When a decode cannot get a
           block, preempt the newest running request and try again later.
        3. Continue prefilling running requests, in chunks, with the remaining
           budget.
        4. Admit waiting requests from the head of the queue while budget and
           blocks allow. The head is the waiting request with the highest
           `priority`, and the earliest position among equals. Stop at the
           first one that does not fit.

        Keep a watermark of free blocks in reserve so the next step still has
        room to make a decision.
        """
        # TODO
        raise NotImplementedError

    def commit_prefill(self, request: Request, tokens: int) -> None:
        request.prefilled += tokens

    def commit_token(self, request: Request) -> None:
        request.generated += 1
        if request.generated >= request.max_new_tokens:
            request.finished = True

    def preempt(self) -> None:
        """Evict one running request and return it to the queue front.

        The victim is the running request with the lowest priority. Among
        equals, it is the newest one.
        """
        # TODO
        raise NotImplementedError


def percentile(values, q):
    """Return the q-th percentile of `values`, with linear interpolation.

    Sort the values as x_0 <= ... <= x_(n-1). Take the rank r = q/100 * (n-1),
    split it into an index i = floor(r) and a fraction phi = r - i, and return
    x_i + phi * (x_(i+1) - x_i). For q = 100, return the largest value.
    """
    # TODO
    raise NotImplementedError


def replay(scheduler, requests, overhead=2.0, per_token=0.125):
    """Replay an arrival trace on a simulated clock and report latency.

    A step that carries `n` tokens takes `overhead + per_token * n` ms. The
    clock starts at 0. Before each step, every request whose `arrival` is at or
    before the clock joins the scheduler. When nothing is waiting or running,
    the clock jumps to the next arrival. A step that retires the last finished
    request returns an empty batch and takes no time. A token's time is the clock at the end
    of the step that produced it. A request's first token is the one its last
    prefill chunk produces, so call `commit_token` once when a chunk finishes
    the prompt. The trace must not trigger preemption.

    Return a dict with these keys:

    - "ttft": {id: first token time - ARRIVAL}. Measure from arrival, not from
      admission, so queueing counts.
    - "itl": {id: (last token time - first token time) / (n - 1)}, for
      requests with n >= 2 tokens. A request that emits n tokens has n - 1
      gaps.
    - "gaps": {id: [gap between each pair of consecutive tokens]}.
    - "order": request ids in the order they first appear in a prefill chunk.
    - "makespan": the clock when the last step ends.
    - "p99_ttft": the 99th percentile of the per-request TTFTs.
    - "p99_itl": the 99th percentile of every gap from every request, pooled.
    - "max_gap": the largest single gap.
    """
    # TODO
    raise NotImplementedError
