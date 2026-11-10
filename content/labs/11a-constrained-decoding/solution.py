"""Reference solution for lab 11a."""
import torch


# ---------------------------------------------------------------------------
# Provided. Do not edit this part: the tests call it.
#
# A regex compiler for a small subset, and a toy vocabulary. The compiler
# builds a byte-level DFA in four steps: parse to a tree, build a Thompson
# NFA, run the subset construction, then trim every state that cannot reach
# an accepting state. After the trim, DEAD (-1) means "no continuation can
# still match", not just "no edge here".
#
# Supported syntax: literals, [abc], [a-z], [^"] (a complement taken within
# printable ASCII, 0x20 to 0x7E), \d, \w, escaped punctuation, ., groups,
# alternation, *, +, ?, {n}, and {m,n}. A non-ASCII literal compiles to its
# UTF-8 bytes in sequence.
# ---------------------------------------------------------------------------

DEAD = -1
_PRINTABLE = frozenset(range(0x20, 0x7F))
_DIGITS = frozenset(range(ord("0"), ord("9") + 1))
_WORD = _DIGITS | frozenset(range(ord("a"), ord("z") + 1)) | \
    frozenset(range(ord("A"), ord("Z") + 1)) | frozenset([ord("_")])


class DFA:
    """A deterministic automaton over bytes. State ids are 0..num_states-1."""

    def __init__(self, trans, accept, start=0):
        self.trans = trans          # list of {byte: next_state}
        self.accept = accept        # list of bool
        self.start = start

    @property
    def num_states(self):
        return len(self.trans)

    def step(self, state, byte):
        """The state after one byte, or DEAD."""
        if state == DEAD:
            return DEAD
        return self.trans[state].get(byte, DEAD)

    def accepting(self, state):
        return state != DEAD and self.accept[state]

    def out_bytes(self, state):
        """The bytes that have an edge out of `state`, sorted."""
        return sorted(self.trans[state]) if state != DEAD else []


class _Parser:
    def __init__(self, text):
        self.s, self.i = text, 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def take(self):
        c = self.s[self.i]
        self.i += 1
        return c

    def alt(self):
        branches = [self.cat()]
        while self.peek() == "|":
            self.i += 1
            branches.append(self.cat())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def cat(self):
        items = []
        while self.peek() not in (None, "|", ")"):
            items.append(self.repeat())
        return ("cat", items)

    def repeat(self):
        node = self.atom()
        while self.peek() in ("*", "+", "?", "{"):
            c = self.take()
            if c == "*":
                node = ("star", node)
            elif c == "+":
                node = ("cat", [node, ("star", node)])
            elif c == "?":
                node = ("alt", [node, ("cat", [])])
            else:
                body = ""
                while self.peek() != "}":
                    body += self.take()
                self.take()
                lo, _, hi = body.partition(",")
                lo_n = int(lo)
                hi_n = int(hi) if hi else lo_n
                node = ("cat", [node] * lo_n + [("alt", [node, ("cat", [])])] * (hi_n - lo_n))
        return node

    def escape(self):
        c = self.take()
        if c == "d":
            return _DIGITS
        if c == "w":
            return _WORD
        if c == "n":
            return frozenset([10])
        if c == "t":
            return frozenset([9])
        return frozenset([ord(c)])

    def atom(self):
        c = self.take()
        if c == "(":
            node = self.alt()
            assert self.take() == ")", "unbalanced parenthesis"
            return node
        if c == "[":
            negate = self.peek() == "^"
            if negate:
                self.i += 1
            chars = set()
            while self.peek() != "]":
                if self.peek() == "\\":
                    self.i += 1
                    chars |= self.escape()
                    continue
                a = self.take()
                if self.peek() == "-" and self.s[self.i + 1] != "]":
                    self.i += 1
                    b = self.take()
                    chars |= set(range(ord(a), ord(b) + 1))
                else:
                    chars.add(ord(a))
            self.take()
            return ("set", frozenset(_PRINTABLE - chars) if negate else frozenset(chars))
        if c == "\\":
            return ("set", self.escape())
        if c == ".":
            return ("set", _PRINTABLE)
        return ("cat", [("set", frozenset([b])) for b in c.encode("utf-8")])


def compile_regex(pattern: str) -> DFA:
    """Compile `pattern` to a trimmed byte-level DFA that matches it entirely."""
    parser = _Parser(pattern)
    tree = parser.alt()
    assert parser.i == len(pattern), f"unparsed input at {parser.i}"
    eps, edges = [], []

    def new():
        eps.append([])
        edges.append([])
        return len(eps) - 1

    def build(node):
        start, end = new(), new()
        kind = node[0]
        if kind == "set":
            edges[start].append((node[1], end))
        elif kind == "cat":
            cur = start
            for child in node[1]:
                s, e = build(child)
                eps[cur].append(s)
                cur = e
            eps[cur].append(end)
        elif kind == "alt":
            for child in node[1]:
                s, e = build(child)
                eps[start].append(s)
                eps[e].append(end)
        else:  # star
            s, e = build(node[1])
            eps[start] += [s, end]
            eps[e] += [s, end]
        return start, end

    nfa_start, nfa_end = build(tree)

    def closure(states):
        seen, stack = set(states), list(states)
        while stack:
            for t in eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    first = closure([nfa_start])
    index, order, trans = {first: 0}, [first], []
    while len(trans) < len(order):
        current = order[len(trans)]
        moves = {}
        for s in current:
            for label, t in edges[s]:
                for b in label:
                    moves.setdefault(b, set()).add(t)
        row = {}
        for b, targets in sorted(moves.items()):
            nxt = closure(targets)
            if nxt not in index:
                index[nxt] = len(order)
                order.append(nxt)
            row[b] = index[nxt]
        trans.append(row)
    accept = [nfa_end in s for s in order]

    # Trim: keep only states that can still reach an accepting state.
    live = {i for i, a in enumerate(accept) if a}
    changed = True
    while changed:
        changed = False
        for i, row in enumerate(trans):
            if i not in live and any(t in live for t in row.values()):
                live.add(i)
                changed = True
    assert 0 in live, "the pattern matches nothing"
    renumber = {old: new_id for new_id, old in enumerate(sorted(live))}
    new_trans = [{b: renumber[t] for b, t in trans[old].items() if t in live}
                 for old in sorted(live)]
    new_accept = [accept[old] for old in sorted(live)]
    return DFA(new_trans, new_accept, renumber[0])


EOS_ID = 0
PAD_ID = 1
_MULTI = [
    b'{"', b'":', b'": "', b'", "', b'"}', b'name', b'age', b', "', b'", ',
    b'yes', b'no', b'maybe', b'ye', b'es', b'20', b'2024', b'-0', b'0.', b'00',
    b'12', b'123', b'1}', b'9"', b'e-', b'-1', b'ab', b'abc', b'bc', b'a-',
    b'caf', b'[[', b']]', b'],', b'[1', b'1,', b'1]', b'(', b')',
]


def toy_vocab() -> list[bytes]:
    """Token id -> bytes.

    Id 0 is the end token and id 1 is a padding token. Both have empty bytes.
    Ids 2 to 257 are the 256 single-byte tokens, which is the byte fallback:
    any string can be spelled, even bytes that are half of a UTF-8 character.
    The multi-character tokens follow.
    """
    return [b"", b""] + [bytes([i]) for i in range(256)] + list(_MULTI)


# ---------------------------------------------------------------------------
# Your work starts here.
# ---------------------------------------------------------------------------


def walk(dfa: DFA, state: int, token: bytes) -> int:
    """Feed the bytes of `token` to the DFA from `state`.

    Return the state after the last byte, or DEAD as soon as a byte has no
    edge. An empty token leaves the state unchanged.
    """
    for byte in token:
        state = dfa.step(state, byte)
        if state == DEAD:
            return DEAD
    return state


def build_token_mask(dfa: DFA, state: int, vocab_bytes: list[bytes],
                     eos_id: int) -> torch.Tensor:
    """A bool tensor of shape (vocab,): True where the token is legal at `state`.

    A token is legal when walking all of its bytes from `state` never hits
    DEAD. Two cases are special. The end token is legal only when `state` is
    accepting. Any other token with empty bytes is never legal, because it
    would let the sampler stall without making progress.
    """
    allowed = []
    for token_id, token in enumerate(vocab_bytes):
        if token_id == eos_id:
            allowed.append(dfa.accepting(state))
        elif len(token) == 0:
            allowed.append(False)
        else:
            allowed.append(walk(dfa, state, token) != DEAD)
    return torch.tensor(allowed, dtype=torch.bool)


class EmptyMaskError(RuntimeError):
    """No token is legal at this state."""


def sample_constrained(logits: torch.Tensor, mask: torch.Tensor,
                       temperature: float = 1.0, top_k: int = 0,
                       generator: torch.Generator | None = None) -> int:
    """Draw one token id from a (vocab,) logits row under `mask`.

    Order: mask first (set illegal logits to -inf), then temperature, then
    top-k, then softmax and the draw. A temperature of 0 or less is greedy
    over the legal tokens. Raise EmptyMaskError when no token is legal,
    because the softmax of an all -inf row is NaN.
    """
    if not bool(mask.any()):
        raise EmptyMaskError("no token is legal at this state")
    logits = logits.float().masked_fill(~mask, float("-inf"))
    if temperature <= 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        kth = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


class TokenFSM:
    """A DFA lifted to tokens, with a per-state mask cache."""

    def __init__(self, dfa: DFA, vocab_bytes: list[bytes], eos_id: int):
        self.dfa = dfa
        self.vocab_bytes = vocab_bytes
        self.eos_id = eos_id
        self.builds = 0              # how many masks were built, not served from cache
        self._cache: dict[int, torch.Tensor] = {}

    @property
    def start(self) -> int:
        return self.dfa.start

    def is_accepting(self, state: int) -> bool:
        return self.dfa.accepting(state)

    def mask_uncached(self, state: int) -> torch.Tensor:
        """Build the mask from scratch and count it in `builds`."""
        self.builds += 1
        return build_token_mask(self.dfa, state, self.vocab_bytes, self.eos_id)

    def mask(self, state: int) -> torch.Tensor:
        """The mask for `state`. Build it once per state, then reuse it."""
        if state not in self._cache:
            self._cache[state] = self.mask_uncached(state)
        return self._cache[state]

    def advance(self, state: int, token: int) -> int:
        """The state after `token`. Raise ValueError if the token is illegal.

        The end token is legal only at an accepting state, and it leaves the
        state unchanged.
        """
        if token == self.eos_id:
            if not self.is_accepting(state):
                raise ValueError(f"end token at non-accepting state {state}")
            return state
        data = self.vocab_bytes[token]
        nxt = walk(self.dfa, state, data) if data else DEAD
        if nxt == DEAD:
            raise ValueError(f"token {token} ({data!r}) is illegal at state {state}")
        return nxt


def generate(fsm: TokenFSM, model, seed: int, max_tokens: int = 64,
             temperature: float = 1.0, top_k: int = 0) -> list[int]:
    """Sample token ids under the grammar until the end token or `max_tokens`.

    `model(prefix_ids) -> logits` of shape (vocab,). Return the ids,
    including the end token if it was drawn. Seed one torch.Generator.
    """
    gen = torch.Generator().manual_seed(seed)
    state, out = fsm.start, []
    for _ in range(max_tokens):
        token = sample_constrained(model(out), fsm.mask(state), temperature, top_k, gen)
        state = fsm.advance(state, token)
        out.append(token)
        if token == fsm.eos_id:
            break
    return out


def greedy_encode(data: bytes, vocab_bytes: list[bytes], eos_id: int) -> list[int]:
    """Encode `data` by longest match from the left.

    Ignore the end token and empty tokens. When two ids have the same bytes,
    use the lower id. Raise ValueError if some byte starts no token.
    """
    table: dict[bytes, int] = {}
    longest = 0
    for token_id, token in enumerate(vocab_bytes):
        if token_id == eos_id or not token:
            continue
        table.setdefault(token, token_id)
        longest = max(longest, len(token))
    ids, pos = [], 0
    while pos < len(data):
        for size in range(min(longest, len(data) - pos), 0, -1):
            hit = table.get(data[pos:pos + size])
            if hit is not None:
                ids.append(hit)
                pos += size
                break
        else:
            raise ValueError(f"no token starts at byte {pos}: {data[pos:pos + 1]!r}")
    return ids


def jump_forward(fsm: TokenFSM, state: int) -> tuple[list[int], int]:
    """Skip the model for text the grammar forces.

    From `state`, follow DFA states that are not accepting and have exactly
    one outgoing byte. Those bytes are forced. Encode them with
    `greedy_encode` and return (token ids, state after them). Return
    ([], state) when nothing is forced.
    """
    dfa = fsm.dfa
    forced = bytearray()
    cur = state
    while not dfa.accepting(cur) and len(dfa.trans[cur]) == 1:
        (byte, nxt), = dfa.trans[cur].items()
        forced.append(byte)
        cur = nxt
    if not forced:
        return [], state
    return greedy_encode(bytes(forced), fsm.vocab_bytes, fsm.eos_id), cur


def nested_array_regex(depth: int) -> str:
    """A regex for a JSON-like value nested at most `depth` arrays deep.

    A value is one digit, or an array `[v,v,...]` of values, or `[]`.
    Depth 0 is a single digit. Each extra level wraps the previous value
    pattern. The DFA for depth d has no counter for d + 1.
    """
    value = "[0-9]"
    for _ in range(depth):
        value = rf"(\[({value}(,{value})*)?\]|[0-9])"
    return value


class ConstraintProcessor:
    """A logits processor for a batch, with one grammar state per row.

    It has the `(input_ids, logits) -> logits` shape of chapter 11's hook.
    Row i is masked with the mask for `states[i]`. Call `advance(tokens)`
    after sampling to move every row to its next state.
    """

    def __init__(self, fsm: TokenFSM, batch: int):
        self.fsm = fsm
        self.states = [fsm.start] * batch

    def __call__(self, input_ids, logits: torch.Tensor) -> torch.Tensor:
        masks = torch.stack([self.fsm.mask(s) for s in self.states])
        if bool((~masks.any(dim=-1)).any()):
            raise EmptyMaskError("a row has no legal token")
        return logits.masked_fill(~masks, float("-inf"))

    def advance(self, tokens: list[int]) -> None:
        self.states = [self.fsm.advance(s, t) for s, t in zip(self.states, tokens)]
