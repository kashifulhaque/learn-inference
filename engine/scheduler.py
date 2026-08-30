"""Continuous batching.

Static batching waits for a whole batch to finish before starting the next one,
so one 2000-token generation holds up nineteen 20-token ones. Continuous
batching runs a scheduling step per decode iteration: finished sequences leave
the batch immediately and waiting ones join in their place.

The scheduler's job each step is to answer one question — which sequences run
now — under two constraints: a token budget, because prefill is compute bound
and a huge prefill starves decode; and a block budget, because the paged cache
is finite. Chunked prefill splits a long prompt across several steps so it can
ride along with decodes instead of blocking them.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .cache import OutOfBlocks, PagedKVCache


class State(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


_ids = itertools.count(1)


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int = 128
    id: str = field(default_factory=lambda: f"req-{next(_ids)}")
    state: State = State.WAITING
    output_ids: list[int] = field(default_factory=list)
    prefilled: int = 0
    arrived_at: float = field(default_factory=time.monotonic)
    first_token_at: float | None = None
    finished_at: float | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + len(self.output_ids)

    @property
    def needs_prefill(self) -> bool:
        return self.prefilled < self.prompt_len

    @property
    def done(self) -> bool:
        return len(self.output_ids) >= self.max_new_tokens or self.state is State.FINISHED

    def ttft(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.arrived_at

    def metrics(self) -> dict[str, float | int | None]:
        end = self.finished_at or time.monotonic()
        generated = len(self.output_ids)
        decode_span = end - (self.first_token_at or end)
        return {
            "id": self.id,
            "prompt_tokens": self.prompt_len,
            "generated_tokens": generated,
            "ttft_s": self.ttft(),
            "latency_s": end - self.arrived_at,
            "inter_token_ms": (decode_span / max(generated - 1, 1)) * 1000
            if generated > 1
            else None,
        }


@dataclass
class Batch:
    """What to run in one forward pass."""

    prefill: list[tuple[Request, int]] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.prefill and not self.decode

    @property
    def token_count(self) -> int:
        return sum(chunk for _, chunk in self.prefill) + len(self.decode)


class Scheduler:
    def __init__(
        self,
        cache: PagedKVCache | None = None,
        max_batch_size: int = 32,
        max_batched_tokens: int = 4096,
        chunk_size: int = 1024,
        watermark: float = 0.02,
    ) -> None:
        self.cache = cache
        self.max_batch_size = max_batch_size
        self.max_batched_tokens = max_batched_tokens
        self.chunk_size = chunk_size
        self.watermark = watermark

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    # --- queue management ---------------------------------------------------

    def add(self, request: Request) -> Request:
        self.waiting.append(request)
        return request

    def add_all(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self.add(request)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # --- one scheduling step ------------------------------------------------

    def step(self) -> Batch:
        """Pick the work for the next forward pass.

        Decodes go first. A running sequence has already paid for its prefill
        and is holding cache blocks, so finishing it frees memory; admitting new
        work ahead of it does the opposite.

        When nothing can be scheduled while sequences are still resident, the
        cache is full. Preempt and retry rather than returning an empty batch,
        which would deadlock the loop.
        """
        for _ in range(self.max_batch_size + 1):
            batch = self._schedule()
            if not batch.empty or not self.running:
                return batch
            self._preempt(self.running)
        return batch

    def _schedule(self) -> Batch:
        batch = Batch()
        budget = self.max_batched_tokens

        still_running: list[Request] = []
        for request in self.running:
            if request.done:
                self._finish(request)
                continue
            if request.needs_prefill:
                still_running.append(request)
                continue
            if len(batch.decode) >= self.max_batch_size or budget < 1:
                still_running.append(request)
                continue
            if not self._reserve(request, 1):
                # No block for the next token. Preempt the newest sequence and
                # let it restart later rather than deadlocking.
                self._preempt(still_running)
                still_running.append(request)
                continue
            batch.decode.append(request)
            budget -= 1
            still_running.append(request)
        self.running = still_running

        # Then prefill, in chunks, with whatever budget is left.
        for request in list(self.running):
            if budget <= 0:
                break
            if not request.needs_prefill:
                continue
            chunk = min(self.chunk_size, budget, request.prompt_len - request.prefilled)
            if chunk > 0 and self._reserve(request, chunk):
                batch.prefill.append((request, chunk))
                budget -= chunk

        while self.waiting and budget > 0 and len(self.running) < self.max_batch_size:
            candidate = self.waiting[0]
            chunk = min(self.chunk_size, budget, candidate.prompt_len)
            if not self._reserve(candidate, chunk):
                break
            self.waiting.popleft()
            candidate.state = State.RUNNING
            self.running.append(candidate)
            batch.prefill.append((candidate, chunk))
            budget -= chunk

        return batch

    # --- results ------------------------------------------------------------

    def commit_prefill(self, request: Request, tokens: int) -> None:
        request.prefilled += tokens

    def commit_token(self, request: Request, token_id: int, eos: set[int]) -> None:
        if request.first_token_at is None:
            request.first_token_at = time.monotonic()
        request.output_ids.append(token_id)
        if token_id in eos or len(request.output_ids) >= request.max_new_tokens:
            request.state = State.FINISHED

    # --- internals ----------------------------------------------------------

    def _reserve(self, request: Request, tokens: int) -> bool:
        if self.cache is None:
            return True
        try:
            needed = self.cache.blocks_needed(request.id, tokens)
            reserve = int(self.cache.num_blocks * self.watermark)
            if needed > max(0, self.cache.free_blocks - reserve):
                return False
            self.cache.allocate(request.id, tokens)
            return True
        except OutOfBlocks:
            return False

    def _preempt(self, running: list[Request]) -> None:
        """Evict the most recently admitted sequence and requeue it.

        Newest-first keeps the sequences closest to finishing, which drains the
        queue faster than evicting the oldest.
        """
        if not running:
            return
        victim = running.pop()
        if self.cache is not None:
            self.cache.release(victim.id)
        victim.state = State.PREEMPTED
        victim.prefilled = 0
        victim.output_ids.clear()
        self.waiting.appendleft(victim)

    def _finish(self, request: Request) -> None:
        request.state = State.FINISHED
        request.finished_at = time.monotonic()
        if self.cache is not None:
            self.cache.release(request.id)
        self.finished.append(request)

    def stats(self) -> dict[str, float | int]:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
            "cache_utilisation": round(self.cache.utilisation(), 3) if self.cache else 0.0,
        }
