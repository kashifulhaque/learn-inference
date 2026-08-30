"""Checks lab 03 with a byte-level tokenizer built for the test.

Each token is a short run of bytes, so multi-byte characters split across
tokens exactly as they do with a real byte-level BPE vocabulary.
"""

from lab_common import Checks

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def make_byte_tokenizer(text: str, chunk: int = 3):
    """Split text into fixed-size byte runs and return (ids, decode)."""
    raw = text.encode("utf-8")
    pieces = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]
    vocab = {i: piece for i, piece in enumerate(pieces)}

    def decode(ids):
        return b"".join(vocab[i] for i in ids)

    return list(vocab), decode


def run(submission):
    c = Checks()
    if not c.require(submission, "build_prompt", "IncrementalDetokenizer"):
        return c.finish()

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is a KV cache?"},
    ]
    prompt = submission.build_prompt(messages)
    expected = (
        f"{IM_START}system\nYou are helpful.{IM_END}\n"
        f"{IM_START}user\nWhat is a KV cache?{IM_END}\n"
        f"{IM_START}assistant\n"
    )
    c.check("build_prompt renders the expected string", lambda: prompt == expected,
            f"got {prompt!r}")
    c.check(
        "the assistant turn is left open",
        lambda: prompt.endswith(f"{IM_START}assistant\n") and IM_END not in prompt.split(
            f"{IM_START}assistant")[-1],
    )

    # Multi-byte content: Devanagari at 3 bytes per character, emoji at 4.
    for label, text, chunk in [
        ("ASCII", "hello world, this is a test", 3),
        ("Devanagari", "नमस्ते दुनिया कैसे हो", 3),
        ("emoji", "ok 👍🏽 done 🚀 fine", 5),
        ("mixed", "café 東京 🎉 end", 2),
    ]:
        ids, decode = make_byte_tokenizer(text, chunk)
        detok = submission.IncrementalDetokenizer(decode)
        chunks = [detok.push(i) for i in ids]
        chunks.append(detok.flush())
        streamed = "".join(chunks)

        c.check(
            f"{label}: the stream reassembles the original text",
            lambda s=streamed, t=text: s == t,
            f"got {streamed!r}",
        )
        c.check(
            f"{label}: no replacement characters are emitted",
            lambda s=streamed: "�" not in s,
        )

    ids, decode = make_byte_tokenizer("नमस्ते 👍🏽", 2)
    detok = submission.IncrementalDetokenizer(decode)
    emitted = [p for p in (detok.push(i) for i in ids) if p]
    c.check(
        "some tokens emit nothing while a character is still incomplete",
        lambda: len(emitted) < len(ids),
        f"{len(emitted)} non-empty pushes for {len(ids)} tokens",
    )

    c.metric("emitted_chunks", len(emitted))
    return c.finish()
