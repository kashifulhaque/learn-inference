"""Lab 15a — a prefix cache over token blocks.

No weights and no GPU. Tokens are integers. A block is `block_size` tokens.

`block_hash` names a full block. `PrefixCache` is the radix tree. The two
agree on how long a shared prefix is, and they disagree when a block is
hashed without its parent.

`plan_state_checkpoints` spends a byte budget on recurrent-state snapshots.
`schedule_prefill` charges only the tokens the cache did not already hold.
"""

import hashlib

# Chapter 2: 3,227,648 bytes in one linear layer, times 48 layers.
STATE_BYTES = 154_927_104


def block_hash(parent: int | None, tokens: tuple[int, ...]) -> int:
    """Hash of the parent hash and this block's token ids.

    `parent` is None on the first block. The preimage is ASCII: the word
    none, or the decimal parent, then `|`, then the token ids joined by
    commas. The result is the first 8 bytes of SHA-256, as an unsigned
    integer.
    """
    # TODO
    raise NotImplementedError


class _Node:
    def __init__(self) -> None:
        self.children: dict[int, "_Edge"] = {}
        self.ref = 0
        self.last_used = 0


class _Edge:
    def __init__(self, tokens: tuple[int, ...], child: _Node, block_id: int) -> None:
        self.tokens = tokens
        self.child = child
        self.block_id = block_id


class PrefixCache:
    """A radix tree of token blocks, with reference counts and leaf eviction.

    `match` does not take a reference. `insert` does. `release` drops the
    one `insert` took. `evict` removes leaves whose reference count is zero,
    least recently used first.
    """

    def __init__(self, block_size: int, capacity: int) -> None:
        self.block_size = block_size
        self.capacity = capacity
        self.root = _Node()
        self.clock = 0
        self._next_id = 1
        self.seqs: dict[int, list[_Edge]] = {}

    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        """Return `(block_ids, n_cached)` for the longest page-aligned hit.

        A hit ends on a block boundary. Tokens inside a partial edge do not
        count.
        """
        # TODO
        raise NotImplementedError

    def insert(self, tokens: list[int]) -> int:
        """Add the sequence, splitting an edge where it diverges.

        Returns a sequence id. Bumps the reference count on every node the
        sequence holds. Copies a partial tail instead of extending it in
        place while another sequence still ends on that tail.
        """
        # TODO
        raise NotImplementedError

    def release(self, seq_id: int) -> None:
        """Drop the reference this sequence holds."""
        # TODO
        raise NotImplementedError

    def evict(self, n: int) -> int:
        """Evict up to `n` leaves with reference count zero. Returns how many."""
        # TODO
        raise NotImplementedError

    def materialize(self, seq_id: int) -> list[int]:
        """Tokens stored for `seq_id`, read from the blocks it holds."""
        # TODO
        raise NotImplementedError


def plan_state_checkpoints(prompt_lens: list[int], budget_bytes: int) -> list[int]:
    """Checkpoint positions that maximise saved prefill tokens.

    One snapshot costs `STATE_BYTES`. A prompt of length L uses the greatest
    chosen position p with p <= L, and contributes p saved tokens. Positions
    are chosen from the prompt lengths themselves. On a tie, return the
    lexicographically smallest list.
    """
    # TODO
    raise NotImplementedError


def schedule_prefill(prompts: list[list[int]], token_budget: int,
                     cache: PrefixCache) -> dict:
    """Admit prompts in order, charging only tokens `cache.match` missed.

    Insert each admitted prompt before looking at the next one. Stop at the
    first prompt whose uncached tail does not fit. Return a dict with
    `admitted`, `charged`, and `prefill_tokens_saved`.
    """
    # TODO
    raise NotImplementedError

