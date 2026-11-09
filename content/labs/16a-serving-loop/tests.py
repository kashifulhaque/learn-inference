"""Checks for the serving loop, the stop hold, and the streaming clock.

The first check calls only `safe_prefix`. Later checks build an EngineCore
around a fake model. Random requests use seed 16.

A plausible wrong implementation emits a stop token as soon as it is sampled,
before a stop string that spans the next token is resolved. For pieces "EN"
then "D" and stop "END", that sends "EN" on the first of the two tokens.
`safe_prefix("hiEN", ["END"])` must still be holding "EN", and the engine
must not put "EN" in that step's stream event.

Another plausible wrong implementation frees every block id on a cancelled
request, including a block another request still holds. The free count then
rises by both of the cancelled request's blocks.
"""

import json
import random

from lab_common import Checks


def _ran(fn, expected):
    """Run fn and always report expected and actual, including on an exception."""
    try:
        actual = fn()
    except Exception as exc:  # noqa: BLE001
        return False, f"expected {expected}, got {type(exc).__name__}: {exc}"
    if actual == expected:
        return True, f"expected {expected}, got {actual}"
    return False, f"expected {expected}, got {actual}"


def offline_text(plan, pieces, stops, max_new):
    """Concatenate the planned pieces and cut at the first stop string."""
    text = ""
    for token in plan[:max_new]:
        text += pieces[token]
        cut = None
        matched = None
        for stop in stops:
            if not stop:
                continue
            idx = text.find(stop)
            if idx == -1:
                continue
            if cut is None or idx < cut or (idx == cut and len(stop) > len(matched)):
                cut = idx
                matched = stop
        if cut is not None:
            return text[:cut]
    return text


class PlanModel:
    """Returns a one-hot row for the next planned token of each request id."""

    def __init__(self, plans, vocab):
        self.plans = plans
        self.vocab = vocab
        self.cursor = {}

    def step(self, ids):
        rows = []
        for req_id in ids:
            index = self.cursor.get(req_id, 0)
            plan = self.plans[req_id]
            token = plan[index] if index < len(plan) else 0
            self.cursor[req_id] = index + 1
            row = [0.0] * self.vocab
            row[token] = 1.0
            rows.append(row)
        return rows


class ConstantModel:
    def __init__(self, vocab, token=0):
        self.vocab = vocab
        self.token = token

    def step(self, ids):
        row = [0.0] * self.vocab
        row[self.token] = 1.0
        return [row[:] for _ in ids]


def _stream(submission, model, pieces, request):
    pool = submission.BlockPool(64, 64)
    engine = submission.EngineCore(model, pool, pieces)
    engine.add(request)
    parts = []
    per_step = []
    for _ in range(request.max_new_tokens + 2):
        events = engine.step()
        texts = [ev.get("text") or "" for ev in events if ev.get("id") == request.id]
        per_step.append("".join(texts))
        parts.extend(texts)
        if any(ev.get("id") == request.id and ev.get("finish_reason") for ev in events):
            break
    return "".join(parts), per_step, pool


def _predicted_idle(n, schedule, forward, post, overlap):
    busy = n * forward
    if overlap:
        idle = schedule + (n - 1) * max(0, schedule - forward) + n * post
        wall = idle + busy
    else:
        wall = n * (schedule + forward + post)
    return (wall - busy) / wall


def _parse_sse(body):
    if not isinstance(body, str) or not body.endswith("\n\n"):
        raise ValueError(f"expected the stream to end with a blank line, got {body!r}")
    records = []
    for block in body.split("\n\n"):
        if not block:
            continue
        if not block.startswith("data: "):
            raise ValueError(f"expected a data record, got {block!r}")
        data = block[len("data: "):]
        records.append(None if data == "[DONE]" else json.loads(data))
    return records


def run(submission):
    c = Checks()

    def hold():
        return submission.safe_prefix("hiEN", ["END"])

    c.check(
        "safe_prefix withholds a stop prefix that is not yet resolved",
        lambda: _ran(hold, ("hi", "EN", None)),
    )

    def completed():
        cases = {
            "hiEND": ("hi", "", "END"),
            "hello": ("hello", "", None),
            "EN": ("", "EN", None),
        }
        for text, expected in cases.items():
            actual = submission.safe_prefix(text, ["END"])
            if actual != expected:
                return actual, expected, text
        return ("ok", "", None), ("ok", "", None), ""

    def check_completed():
        try:
            actual, expected, text = completed()
        except Exception as exc:  # noqa: BLE001
            return False, f"expected ('hi', '', 'END') for 'hiEND', got {type(exc).__name__}: {exc}"
        if actual != expected:
            return False, f"expected {expected} for {text!r}, got {actual}"
        return True, (
            "expected ('hi', '', 'END'), ('hello', '', None), and ('', 'EN', None), got the same"
        )

    c.check("safe_prefix strips a completed stop and passes ordinary text", check_completed)

    pieces = ["a", "b", "c", "d", "e"]
    random.seed(16)
    plans = {}
    specs = []
    for i in range(50):
        length = random.randint(1, 10)
        plan = [random.randrange(len(pieces)) for _ in range(length)]
        if random.random() < 0.5:
            stops = [pieces[random.randrange(5)] + pieces[random.randrange(5)]]
        else:
            stops = []
        max_new = random.randint(1, length)
        req_id = f"r{i}"
        plans[req_id] = plan
        specs.append((req_id, plan, stops, max_new))

    def check_fifty():
        model = PlanModel(plans, len(pieces))
        mismatches = []
        for req_id, plan, stops, max_new in specs:
            request = submission.Request(req_id, max_new, stop=stops, n_blocks=1)
            try:
                got, _, _ = _stream(submission, model, pieces, request)
            except Exception as exc:  # noqa: BLE001
                return False, f"expected offline text for {req_id}, got {type(exc).__name__}: {exc}"
            expected = offline_text(plan, pieces, stops, max_new)
            if got != expected:
                mismatches.append((req_id, expected, got))
        if mismatches:
            req_id, expected, got = mismatches[0]
            return False, (
                f"expected {expected!r}, got {got!r} "
                f"({len(mismatches)} of 50 requests mismatched, first {req_id})"
            )
        return True, "expected 50 matches, got 50"

    c.check("streamed text equals offline generation for 50 seeded requests", check_fifty)

    # Emitting "EN" before "D" arrives is the wrong implementation named above.
    span_pieces = ["Say ", "EN", "D", "!"]
    span_plan = {"span": [0, 1, 2]}

    def check_span():
        model = PlanModel(span_plan, len(span_pieces))
        request = submission.Request("span", 3, stop=["END"], n_blocks=1)
        try:
            final, per_step, _ = _stream(submission, model, span_pieces, request)
        except Exception as exc:  # noqa: BLE001
            return False, f"expected the second step to emit '', got {type(exc).__name__}: {exc}"
        second = per_step[1] if len(per_step) > 1 else "<missing step>"
        if second != "":
            return False, f"expected the second step to emit '', got {second!r}"
        if final != "Say ":
            return False, f"expected final 'Say ', got {final!r}"
        return True, "expected second step '' and final 'Say ', got the same"

    c.check(
        "a stop string spanning two tokens is withheld until it resolves",
        check_span,
    )

    def check_cancel():
        try:
            pool = submission.BlockPool(8, 4)
            engine = submission.EngineCore(ConstantModel(1), pool, ["a"])
            engine.add(submission.Request("solo", 4, n_blocks=3))
            engine.cancel("solo")
        except Exception as exc:  # noqa: BLE001
            return False, f"expected free 8 and free_slots 4, got {type(exc).__name__}: {exc}"
        if pool.free != 8 or pool.free_slots != 4:
            return False, f"expected free 8 and free_slots 4, got free {pool.free} and free_slots {pool.free_slots}"
        return True, "expected free 8 and free_slots 4, got the same"

    c.check("cancel returns blocks and the recurrent slot to the pool", check_cancel)

    def check_shared():
        try:
            pool = submission.BlockPool(8, 4)
            engine = submission.EngineCore(ConstantModel(1), pool, ["a"])
            engine.add(submission.Request("a", 4, n_blocks=2))
            engine.add(submission.Request("b", 4, n_blocks=1))
            shared = pool.held("a")[0]
            pool.attach("b", shared)
            engine.cancel("a")
        except Exception as exc:  # noqa: BLE001
            return False, f"expected free 6 and free_slots 3, got {type(exc).__name__}: {exc}"
        if pool.free != 6 or pool.free_slots != 3:
            return False, f"expected free 6 and free_slots 3, got free {pool.free} and free_slots {pool.free_slots}"
        if shared not in pool.held("b"):
            return False, f"expected block {shared} still held by b, got {pool.held('b')}"
        return True, "expected free 6 and free_slots 3, got the same"

    c.check("cancel does not free a block another request still holds", check_shared)

    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    events = [
        {"content": "Say "},
        {"content": "hi"},
        {"finish_reason": "stop", "usage": usage},
    ]

    def parsed_chunks():
        return _parse_sse(submission.sse_chunks(events))

    def check_schema():
        try:
            records = parsed_chunks()
        except Exception as exc:  # noqa: BLE001
            return False, f"expected a chat.completion.chunk stream, got {type(exc).__name__}: {exc}"
        dicts = [rec for rec in records if rec is not None]
        if not records or records[-1] is not None:
            return False, f"expected the last record to be [DONE], got {records[-1]!r}"
        objects = [rec.get("object") for rec in dicts]
        if any(obj != "chat.completion.chunk" for obj in objects):
            return False, f"expected object 'chat.completion.chunk', got {objects}"
        reasons = []
        for rec in dicts:
            for choice in rec.get("choices") or []:
                if choice.get("finish_reason"):
                    reasons.append(choice["finish_reason"])
        if reasons != ["stop"]:
            return False, f"expected finish_reason ['stop'], got {reasons}"
        usage_rows = [rec.get("usage") for rec in dicts if rec.get("usage")]
        if usage_rows != [usage]:
            return False, f"expected usage [{usage}], got {usage_rows}"
        empty = [rec for rec in dicts if rec.get("choices") == [] and rec.get("usage")]
        if len(empty) != 1:
            return False, f"expected one usage chunk with empty choices, got {len(empty)}"
        return True, "expected object, finish_reason stop, and one usage chunk, got the same"

    c.check("sse_chunks sets object, finish_reason, and usage", check_schema)

    def check_round_trip():
        try:
            records = parsed_chunks()
        except Exception as exc:  # noqa: BLE001
            return False, f"expected content 'Say hi', got {type(exc).__name__}: {exc}"
        content = []
        for rec in records:
            if not rec:
                continue
            for choice in rec.get("choices") or []:
                content.append((choice.get("delta") or {}).get("content") or "")
        got = "".join(content)
        if got != "Say hi":
            return False, f"expected content 'Say hi', got {got!r}"
        return True, "expected content 'Say hi', got 'Say hi'"

    c.check("a parser round-trips the streamed content", check_round_trip)

    def check_idle():
        schedule, forward, post, n = 6, 10, 2, 8
        expected_serial = _predicted_idle(n, schedule, forward, post, False)
        expected_overlap = _predicted_idle(n, schedule, forward, post, True)
        try:
            fractions = []
            for overlap in (False, True):
                pool = submission.BlockPool(16, 16)
                engine = submission.EngineCore(
                    ConstantModel(1), pool, ["a"],
                    schedule_ticks=schedule, forward_ticks=forward,
                    post_ticks=post, overlap=overlap,
                )
                engine.add(submission.Request("load", n, n_blocks=1))
                for _ in range(n + 2):
                    events = engine.step()
                    if any(ev.get("finish_reason") for ev in events):
                        break
                fractions.append(engine.gpu_idle_fraction)
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"expected overlap {expected_overlap:.6f} below serial {expected_serial:.6f}, "
                f"got {type(exc).__name__}: {exc}"
            )
        serial, overlap = fractions
        if abs(serial - expected_serial) > 1e-9 or abs(overlap - expected_overlap) > 1e-9:
            return False, (
                f"expected serial {expected_serial:.6f} and overlap {expected_overlap:.6f}, "
                f"got serial {serial:.6f} and overlap {overlap:.6f}"
            )
        if not overlap < serial:
            return False, f"expected overlap {overlap:.6f} < serial {serial:.6f}, got overlap not smaller"
        c.metric("gpu_idle_fraction", round(overlap, 6))
        c.metric("requests", 50)
        return True, (
            f"expected serial {expected_serial:.6f} and overlap {expected_overlap:.6f}, got the same"
        )

    c.check("overlapping the schedule lowers gpu_idle_fraction", check_idle)
    return c.finish()
