"""Lab 16a — the serving loop and a streaming API.

The model is fake. `model.step(ids)` takes a list of request ids and returns
one logit row per id. You sample with argmax (lowest index on a tie). Piece
`pieces[token_id]` is the text of that token. There is no real tokenizer here:
chapter 3's detokenizer already turns ids into text. You hold stop strings on
top of that text.

`schedule_ticks`, `forward_ticks`, and `post_ticks` are a simulated clock, not
a GPU. The GPU is busy only during `forward_ticks`.
"""


def safe_prefix(text: str, stops: list[str]) -> tuple[str, str, str | None]:
    """Split pending text into (emit, hold, matched_stop).

    `matched_stop` is the stop string that occurs in `text`, chosen as the
    earliest match and, on a tie, the longest one. `emit` is the text before
    that match. The stop string itself is not part of `emit`, and `hold` is
    empty.

    When no stop occurs, `hold` is the longest suffix of `text` that is a
    proper prefix of some stop. `emit` is the rest. A suffix that is only a
    prefix of a stop stays unsent until a later call resolves it.

    Empty strings in `stops` are ignored. With no stops, return
    `(text, "", None)`.
    """
    # TODO
    raise NotImplementedError


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
    """A block pool and a pool of recurrent-state slots.

    Each live request holds one slot. A block can be referenced by more than
    one request. It returns to the free list only when its last reference is
    released.
    """

    def __init__(self, num_blocks: int, num_slots: int) -> None:
        self.num_blocks = num_blocks
        self.num_slots = num_slots
        self.free = num_blocks
        self.free_slots = num_slots
        # TODO: free lists, per-request holdings, and refcounts.
        raise NotImplementedError

    def alloc(self, req_id: str, n_blocks: int) -> list[int]:
        """Reserve `n_blocks` blocks and one slot for `req_id`.

        Returns the new block ids. Lowers `free` by `n_blocks` and
        `free_slots` by 1.
        """
        # TODO
        raise NotImplementedError

    def attach(self, req_id: str, block_id: int) -> None:
        """Give `req_id` another reference to a block that is already held.

        Does not change `free` or `free_slots`.
        """
        # TODO
        raise NotImplementedError

    def held(self, req_id: str) -> list[int]:
        """The block ids `req_id` currently references."""
        # TODO
        raise NotImplementedError

    def release(self, req_id: str) -> None:
        """Drop every reference `req_id` holds, and return its slot.

        A block goes back to the free list only when no request still holds
        it. Releasing an unknown id does nothing.
        """
        # TODO
        raise NotImplementedError


class EngineCore:
    """Add a request, step a fake model, stream text, and cancel.

    `step` returns a list of `{"id", "text", "finish_reason"}`. `text` is only
    the newly safe prefix. `finish_reason` is `"stop"`, `"length"`, or None.
    On `"length"`, flush any held prefix. On `"stop"`, do not emit the stop
    string. Finishing or `cancel` returns that request's blocks and its slot.
    """

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
        # TODO
        raise NotImplementedError

    @property
    def gpu_idle_fraction(self) -> float:
        """`(clock - gpu_busy) / clock`, or 0 before the first step."""
        # TODO
        raise NotImplementedError

    def add(self, request: Request) -> None:
        """Allocate the request's blocks and slot, then queue it."""
        # TODO
        raise NotImplementedError

    def cancel(self, req_id: str) -> None:
        """Drop the request and return its blocks and slot to the pool."""
        # TODO
        raise NotImplementedError

    def step(self) -> list[dict]:
        """Schedule, forward, sample, detokenize, and stream one step.

        Call `model.step` with the batch's request ids, in order. One call
        samples one new token for each id.
        """
        # TODO
        raise NotImplementedError


def sse_chunks(events: list[dict]) -> str:
    """Format events as an OpenAI `chat.completion.chunk` server-sent stream.

    Each event may include `content`, `finish_reason`, and `usage`. A content
    chunk has `choices[0].delta.content` and `finish_reason` null. A finish
    chunk has an empty delta and the finish reason. When any event carries
    `usage`, every chunk includes `usage`, the earlier ones null, and one
    extra chunk before the trailer has `choices: []` and that usage object.

    Chunks share `id` `"chatcmpl-lab"`, `created` 1, `model` `"toy"`, and
    `object` `"chat.completion.chunk"`. Separate records with a blank line.
    End with `data: [DONE]`.
    """
    # TODO
    raise NotImplementedError
