"""Lab 16b — scheduling policy, fair sharing, and routing across replicas.

Everything runs on a simulated clock. Times are milliseconds. Nothing here is
an A100 measurement.

Five pieces, smallest first:

* `token_budget_for_slo` turns an inter-token latency target into a chunk budget.
* `effective_priority` and `pick_next` order a queue by priority with aging.
* `VTCScheduler` shares service fairly between clients.
* `RoundRobin`, `LeastLoaded`, and `PrefixAware` choose a replica for a request.
* `cold_start_s` and `replicas_needed` are the autoscaling arithmetic.

A router's `route(request, replicas, now)` returns the `id` of the chosen
replica. The test harness builds the replicas. Each one has `id`,
`in_flight(now)` (requests assigned and not finished), `queued_ms(now)` (work
ahead of a new request), and `match(request.blocks)` (how many leading blocks
of the prompt the replica has cached). A request has `blocks`, a list of block
ids, and 16 tokens per block.
"""

import math  # noqa: F401


def token_budget_for_slo(slo_ms: float, base_ms: float = 2.0, per_token_ms: float = 0.125) -> int:
    """Return the most tokens one step can hold and still meet `slo_ms`.

    A step costs `base_ms + per_token_ms * tokens` (chapter 16's simulated
    clock). Return the largest whole number of tokens whose step time is at
    most `slo_ms`. Return 0 when even an empty step exceeds the target.
    """
    # TODO
    raise NotImplementedError


def effective_priority(priority: float, waited_ms: float, aging_ms: float | None) -> float:
    """Return `priority` plus one level for every `aging_ms` the request waited.

    With `aging_ms` of None or 0, aging is off and the result is `priority`.
    """
    # TODO
    raise NotImplementedError


def pick_next(waiting: list[dict], now: float, aging_ms: float | None) -> int | None:
    """Return the index in `waiting` of the request to run next, or None.

    Each entry has `priority` and `arrival`. The winner has the highest
    effective priority. Break ties by the earliest `arrival`, then by the
    lower index.
    """
    # TODO
    raise NotImplementedError


class VTCScheduler:
    """Virtual Token Counter fair scheduling (Sheng et al., 2401.00588).

    Each client has a counter of weighted tokens served. The next request
    always comes from the backlogged client with the smallest counter.
    Admitting a request charges its client `w_in * in_tokens + w_out * out_tokens`.
    The lab charges the whole request at admission. A real engine charges the
    output tokens as it decodes them.

    A client whose queue is empty when a request arrives is lifted first. Its
    counter becomes the larger of its own counter and the smallest counter
    among the clients that are backlogged. When no client is backlogged, the
    reference is the counter of the client whose queue emptied most recently.
    """

    def __init__(self, w_in: float = 1.0, w_out: float = 2.0) -> None:
        self.w_in = w_in
        self.w_out = w_out
        self.counters: dict[str, float] = {}
        self.queues: dict[str, list] = {}
        self.last_left: str | None = None
        # TODO: any extra state you need.

    def counter(self, client: str) -> float:
        """Return the client's counter, 0.0 if it has never been seen."""
        # TODO
        raise NotImplementedError

    def arrive(self, client: str, request: tuple) -> None:
        """Queue `request`, a tuple (id, in_tokens, out_tokens), for `client`."""
        # TODO
        raise NotImplementedError

    def next_request(self) -> tuple[str, tuple] | None:
        """Pop and charge the next request. Return (client, request), or None."""
        # TODO
        raise NotImplementedError


class RoundRobin:
    """Send request k to replica k mod n."""

    def __init__(self) -> None:
        self.next = 0

    def route(self, request, replicas, now: float) -> int:
        choice = self.next % len(replicas)
        self.next += 1
        return replicas[choice].id


class LeastLoaded:
    """Send the request to the replica with the least queued work."""

    def route(self, request, replicas, now: float) -> int:
        # TODO
        raise NotImplementedError


class PrefixAware:
    """Send the request to the replica that already caches most of its prompt.

    Rank replicas by `match(request.blocks)` (more first), then by
    `queued_ms(now)` (less first), then by `id`. With `epsilon` of None, take
    the first replica in that ranking (pure affinity).

    With a number for `epsilon`, apply a bounded load. Let `total` be the sum
    of `in_flight` over all replicas and n the replica count. The cap is
    `ceil((1 + epsilon) * (total + 1) / n)`. Take the first replica in the
    ranking whose `in_flight` is below the cap. Some replica always is.
    """

    def __init__(self, epsilon: float | None = None) -> None:
        self.epsilon = epsilon

    def route(self, request, replicas, now: float) -> int:
        # TODO
        raise NotImplementedError


def cold_start_s(weight_bytes: float, read_gb_per_s: float, fixed_s: float) -> float:
    """Return seconds from "start a replica" to "serving": load plus fixed setup.

    `read_gb_per_s` is in decimal GB/s, so 1 GB is 1e9 bytes.
    """
    # TODO
    raise NotImplementedError


def replicas_needed(rate_per_s: float, service_s: float, target_util: float) -> int:
    """Return the fewest replicas that keep utilisation at or below `target_util`.

    Utilisation is `rate_per_s * service_s / replicas`.
    """
    # TODO
    raise NotImplementedError
