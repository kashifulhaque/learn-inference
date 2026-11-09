"""Reference solution for the prefix-cache lab."""

from __future__ import annotations

import hashlib
from itertools import combinations

# Chapter 2: 3,227,648 bytes in one linear layer, times 48 layers.
STATE_BYTES = 154_927_104


def block_hash(parent: int | None, tokens: tuple[int, ...]) -> int:
    """Hash of the parent hash and this block's token ids.

    The preimage is ASCII: `none` or the decimal parent, then `|`, then the
    token ids joined by commas. The result is the first 8 bytes of SHA-256.
    """
    head = "none" if parent is None else str(int(parent))
    body = head + "|" + ",".join(str(int(token)) for token in tokens)
    digest = hashlib.sha256(body.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


class _Node:
    def __init__(self) -> None:
        self.children: dict[int, _Edge] = {}
        self.ref = 0
        self.last_used = 0


class _Edge:
    def __init__(self, tokens: tuple[int, ...], child: _Node, block_id: int) -> None:
        self.tokens = tokens
        self.child = child
        self.block_id = block_id


class PrefixCache:
    """Radix tree over token blocks. Edges are at most one block long."""

    def __init__(self, block_size: int, capacity: int) -> None:
        self.block_size = block_size
        self.capacity = capacity
        self.root = _Node()
        self.clock = 0
        self._next_id = 1
        self._next_seq = 1
        self.seqs: dict[int, list[_Edge]] = {}

    def _touch(self, node: _Node) -> None:
        self.clock += 1
        node.last_used = self.clock

    def _new_edge(self, tokens: tuple[int, ...], child: _Node) -> _Edge:
        edge = _Edge(tokens, child, self._next_id)
        self._next_id += 1
        return edge

    def _live_edges(self) -> int:
        count = 0

        def walk(node: _Node) -> None:
            nonlocal count
            for edge in node.children.values():
                count += 1
                walk(edge.child)

        walk(self.root)
        return count

    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        """Longest hit whose length is a multiple of the block size."""
        rest = tuple(tokens)
        node = self.root
        taken: list[_Edge] = []
        pos = 0
        while pos < len(rest):
            edge = node.children.get(rest[pos])
            if edge is None:
                break
            n = len(edge.tokens)
            if tuple(rest[pos:pos + n]) != edge.tokens:
                break
            taken.append(edge)
            self._touch(edge.child)
            node = edge.child
            pos += n
        best_ids: list[int] = []
        best_len = 0
        acc = 0
        running: list[int] = []
        for edge in taken:
            acc += len(edge.tokens)
            running.append(edge.block_id)
            if acc % self.block_size == 0:
                best_ids = list(running)
                best_len = acc
        return best_ids, best_len

    def _split(self, parent: _Node, edge: _Edge, at: int) -> None:
        """Break `edge` into a shared prefix and the previous suffix.

        The edge object is not rewritten. Holders of that object move to the
        two new edges, whose tokens concatenate to the old ones.
        """
        prefix_tokens = edge.tokens[:at]
        suffix_tokens = edge.tokens[at:]
        mid = _Node()
        mid.last_used = edge.child.last_used
        suffix = self._new_edge(suffix_tokens, edge.child)
        prefix = self._new_edge(prefix_tokens, mid)
        mid.children[suffix.tokens[0]] = suffix
        del parent.children[edge.tokens[0]]
        parent.children[prefix.tokens[0]] = prefix
        for edges in self.seqs.values():
            if edge in edges:
                index = edges.index(edge)
                edges[index:index + 1] = [prefix, suffix]
                mid.ref += 1

    def _add_chain(self, node: _Node, tokens: tuple[int, ...]) -> list[_Edge]:
        created: list[_Edge] = []
        pos = 0
        while pos < len(tokens):
            while self._live_edges() >= self.capacity:
                if self.evict(1) == 0:
                    break
            chunk = tokens[pos:pos + self.block_size]
            child = _Node()
            edge = self._new_edge(chunk, child)
            node.children[chunk[0]] = edge
            created.append(edge)
            node = child
            pos += len(chunk)
        return created

    def insert(self, tokens: list[int]) -> int:
        rest = tuple(tokens)
        node = self.root
        pos = 0
        held: list[_Edge] = []
        while pos < len(rest):
            edge = node.children.get(rest[pos])
            if edge is None:
                held.extend(self._add_chain(node, rest[pos:]))
                break
            limit = min(len(edge.tokens), len(rest) - pos)
            common = 0
            while common < limit and edge.tokens[common] == rest[pos + common]:
                common += 1
            if common < len(edge.tokens):
                self._split(node, edge, common)
                continue
            held.append(edge)
            node = edge.child
            pos += len(edge.tokens)
        seq_id = self._next_seq
        self._next_seq += 1
        self.seqs[seq_id] = held
        for edge in held:
            edge.child.ref += 1
            self._touch(edge.child)
        return seq_id

    def release(self, seq_id: int) -> None:
        for edge in self.seqs.pop(seq_id):
            edge.child.ref -= 1

    def evict(self, n: int) -> int:
        removed = 0
        for _ in range(n):
            leaves = self._leaves()
            if not leaves:
                break
            leaves.sort(key=lambda item: item[0].last_used)
            _node, parent, edge = leaves[0]
            del parent.children[edge.tokens[0]]
            removed += 1
        return removed

    def _leaves(self) -> list[tuple[_Node, _Node, _Edge]]:
        found: list[tuple[_Node, _Node, _Edge]] = []

        def walk(node: _Node) -> None:
            for edge in node.children.values():
                child = edge.child
                if child.children:
                    walk(child)
                elif child.ref == 0:
                    found.append((child, node, edge))

        walk(self.root)
        return found

    def materialize(self, seq_id: int) -> list[int]:
        out: list[int] = []
        for edge in self.seqs[seq_id]:
            out.extend(edge.tokens)
        return out


def _saved(positions: tuple[int, ...] | list[int], lengths: list[int]) -> int:
    total = 0
    for length in lengths:
        best = 0
        for position in positions:
            if 0 < position <= length and position > best:
                best = position
        total += best
    return total


def plan_state_checkpoints(prompt_lens: list[int], budget_bytes: int) -> list[int]:
    """Positions that maximise saved tokens, lex-smallest on a tie."""
    slots = budget_bytes // STATE_BYTES
    candidates = sorted({length for length in prompt_lens if length > 0})
    if slots <= 0 or not candidates:
        return []
    best: list[int] = []
    best_score = -1
    limit = min(slots, len(candidates))
    for size in range(0, limit + 1):
        for combo in combinations(candidates, size):
            score = _saved(combo, prompt_lens)
            chosen = list(combo)
            if score > best_score or (score == best_score and chosen < best):
                best_score = score
                best = chosen
    return best


def schedule_prefill(prompts: list[list[int]], token_budget: int,
                     cache: PrefixCache) -> dict:
    """Charge the uncached tail of each prompt against `token_budget`."""
    admitted = 0
    charged = 0
    saved = 0
    for prompt in prompts:
        _ids, n_cached = cache.match(prompt)
        need = len(prompt) - n_cached
        if charged + need > token_budget:
            break
        cache.insert(prompt)
        charged += need
        saved += n_cached
        admitted += 1
    return {
        "admitted": admitted,
        "charged": charged,
        "prefill_tokens_saved": saved,
    }
