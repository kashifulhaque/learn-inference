"""Lab 03 — chat templates and incremental detokenization.

Part 1 builds the prompt string a chat model expects.

Part 2 is the streaming problem: one token is not one printable unit, so
decoding tokens independently produces replacement characters wherever a
multi-byte character straddles a token boundary.
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(messages: list[dict]) -> str:
    """Render chat messages into the string the model completes.

    Each message is {"role": ..., "content": ...} with a role of "system",
    "user", or "assistant". The result opens an assistant turn and leaves it
    open, because that is where generation continues.

    For example, one user message renders as:

        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant

    with no trailing newline after the final "assistant".
    """
    # TODO
    raise NotImplementedError


class IncrementalDetokenizer:
    """Turns a stream of token ids into a stream of printable text.

    The harness gives you a `decode` callable that maps a list of token ids to
    bytes. Your job is to emit each character exactly once, and never emit a
    partial UTF-8 sequence.
    """

    def __init__(self, decode) -> None:
        self.decode = decode
        # TODO: keep whatever state you need.

    def push(self, token_id: int) -> str:
        """Accept one token and return the text that is newly printable.

        Returns an empty string when the token completes nothing on its own.
        """
        # TODO
        raise NotImplementedError

    def flush(self) -> str:
        """Return any remaining decodable text at the end of a generation."""
        # TODO
        raise NotImplementedError
