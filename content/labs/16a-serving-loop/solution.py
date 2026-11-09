"""Reference solution for the serving-loop lab."""


def safe_prefix(text: str, stops: list[str]) -> tuple[str, str, str | None]:
    """Split pending text into (emit, hold, matched_stop)."""
    stops = [stop for stop in stops if stop]
    earliest = None
    matched = None
    for stop in stops:
        idx = text.find(stop)
        if idx == -1:
            continue
        longer = matched is not None and len(stop) > len(matched)
        if earliest is None or idx < earliest or (idx == earliest and longer):
            earliest = idx
            matched = stop
    if matched is not None:
        return text[:earliest], "", matched

    hold = 0
    for stop in stops:
        limit = min(len(text), len(stop) - 1)
        for size in range(limit, 0, -1):
            if text.endswith(stop[:size]):
                if size > hold:
                    hold = size
                break
    if hold:
        return text[:-hold], text[-hold:], None
    return text, "", None


class Request:
    """One generation. The engine fills `generated`, `pending`, and `finished`."""

    def __init__(self, id: str, max_new_tokens: int, stop: list[str] | None = None,
                 n_blocks: int = 1) -> None:
        self.id = id
        self.max_new_tokens = max_new_tokens
        self.stop = list(stop or [])
        self.n_blocks = n_blocks
        self.generated = 0
        self.pending = ""
        self.finished = False


class BlockPool:
    """Blocks with refcounts, plus one recurrent-state slot per request."""

    def __init__(self, num_blocks: int, num_slots: int) -> None:
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.free = num_blocks
        self.free_slots = num_slots
        self._free_blocks = list(range(num_blocks))
        self._free_slot_ids = list(range(num_slots))
        self._held: dict[str, list[int]] = {}
        self._refs: dict[int, int] = {}
        self._slots: dict[str, int] = {}

    def alloc(self, req_id: str, n_blocks: int) -> list[int]:
        if self.free < n_blocks or self.free_slots < 1:
            raise RuntimeError(
                f"need {n_blocks} blocks and 1 slot, have {self.free} and {self.free_slots}"
            )
        blocks = [self._free_blocks.pop() for _ in range(n_blocks)]
        self.free -= n_blocks
        for block_id in blocks:
            self._refs[block_id] = self._refs.get(block_id, 0) + 1
        self._held[req_id] = list(blocks)
        self._slots[req_id] = self._free_slot_ids.pop()
        self.free_slots -= 1
        return blocks

    def attach(self, req_id: str, block_id: int) -> None:
        self._refs[block_id] = self._refs.get(block_id, 0) + 1
        self._held.setdefault(req_id, []).append(block_id)

    def held(self, req_id: str) -> list[int]:
        return list(self._held.get(req_id, []))

    def release(self, req_id: str) -> None:
        for block_id in self._held.pop(req_id, []):
            self._refs[block_id] -= 1
            if self._refs[block_id] == 0:
                del self._refs[block_id]
                self._free_blocks.append(block_id)
                self.free += 1
        slot = self._slots.pop(req_id, None)
        if slot is not None:
            self._free_slot_ids.append(slot)
            self.free_slots += 1


class EngineCore:
    """Serial or overlapped engine loop over a fake model."""

    def __init__(self, model, pool: BlockPool, pieces: list[str], *,
                 schedule_ticks: int = 0, forward_ticks: int = 1,
                 post_ticks: int = 0, overlap: bool = False) -> None:
        self.model = model
        self.pool = pool
        self.pieces = pieces
        self.schedule_ticks = schedule_ticks
        self.forward_ticks = forward_ticks
        self.post_ticks = post_ticks
        self.overlap = overlap
        self.clock = 0
        self.gpu_busy = 0
        self._prefetched = False
        self._waiting: list[Request] = []
        self._running: list[Request] = []

    @property
    def gpu_idle_fraction(self) -> float:
        if self.clock == 0:
            return 0.0
        return (self.clock - self.gpu_busy) / self.clock

    def add(self, request: Request) -> None:
        self.pool.alloc(request.id, request.n_blocks)
        request.generated = 0
        request.pending = ""
        request.finished = False
        self._waiting.append(request)

    def cancel(self, req_id: str) -> None:
        self._waiting = [req for req in self._waiting if req.id != req_id]
        self._running = [req for req in self._running if req.id != req_id]
        self.pool.release(req_id)

    def _tick(self, more_after: bool) -> None:
        if not self._prefetched:
            self.clock += self.schedule_ticks
        self._prefetched = False
        self.clock += self.forward_ticks
        self.gpu_busy += self.forward_ticks
        if self.overlap and more_after:
            self.clock += max(0, self.schedule_ticks - self.forward_ticks)
            self._prefetched = True
        self.clock += self.post_ticks

    def step(self) -> list[dict]:
        if self._waiting:
            self._running.extend(self._waiting)
            self._waiting = []
        batch = [
            req for req in self._running
            if not req.finished and req.generated < req.max_new_tokens
        ]
        if not batch:
            return []
        more_after = any(req.generated + 1 < req.max_new_tokens for req in batch)
        self._tick(more_after)
        rows = self.model.step([req.id for req in batch])
        events = []
        finished = []
        for req, row in zip(batch, rows):
            token = max(range(len(row)), key=lambda i: (row[i], -i))
            req.generated += 1
            req.pending += self.pieces[token]
            emit, hold, matched = safe_prefix(req.pending, req.stop)
            reason = None
            if matched is not None:
                reason = "stop"
                req.pending = ""
                req.finished = True
            elif req.generated >= req.max_new_tokens:
                reason = "length"
                emit = emit + hold
                req.pending = ""
                req.finished = True
            else:
                req.pending = hold
            if emit or reason:
                events.append({
                    "id": req.id,
                    "text": emit,
                    "finish_reason": reason,
                })
            if req.finished:
                finished.append(req)
        for req in finished:
            self._running.remove(req)
            self.pool.release(req.id)
        return events


def sse_chunks(events: list[dict]) -> str:
    """Format an OpenAI-style chat.completion.chunk stream."""
    usage = None
    for event in events:
        if event.get("usage") is not None:
            usage = event["usage"]

    payloads = []
    for event in events:
        content = event.get("content") or ""
        finish = event.get("finish_reason")
        if content and finish:
            payloads.append(_chunk(content, None, usage))
            payloads.append(_chunk("", finish, usage))
        elif finish:
            payloads.append(_chunk("", finish, usage))
        else:
            payloads.append(_chunk(content, None, usage))

    if usage is not None:
        for payload in payloads:
            payload["usage"] = None
        payloads.append({
            "id": "chatcmpl-lab",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "toy",
            "choices": [],
            "usage": usage,
        })

    lines = ["data: " + _dump(payload) for payload in payloads]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def _chunk(content: str, finish, usage_marker) -> dict:
    delta = {"content": content} if content else {}
    payload = {
        "id": "chatcmpl-lab",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "toy",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    }
    if usage_marker is not None:
        payload["usage"] = None
    return payload


def _dump(payload: dict) -> str:
    import json
    return json.dumps(payload, separators=(",", ":"))
