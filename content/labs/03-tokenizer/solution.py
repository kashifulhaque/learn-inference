IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        parts.append(f"{IM_START}{message['role']}\n{message['content']}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


class IncrementalDetokenizer:
    def __init__(self, decode) -> None:
        self.decode = decode
        self.ids: list[int] = []
        self.emitted = 0          # characters already returned to the caller

    def _decodable(self) -> str:
        """Longest valid prefix of the accumulated bytes.

        Trailing bytes that begin an incomplete UTF-8 sequence stay buffered
        until the token that completes them arrives.
        """
        raw = self.decode(self.ids)
        return raw.decode("utf-8", errors="ignore")

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self._decodable()
        new = text[self.emitted:]
        self.emitted = len(text)
        return new

    def flush(self) -> str:
        text = self.decode(self.ids).decode("utf-8", errors="replace")
        new = text[self.emitted:]
        self.emitted = len(text)
        return new
