"""The cost model (NFR-6). Pure functions, no I/O.

For each worker i: a (acquisition), l (weight load), b (serving: at least
one request in flight) and d (held but idle) seconds. All are charged,
because the department bills occupancy, not utilisation.

    G = sum_i (a_i + l_i + b_i + d_i) / 3600
    C = 1000 * G / Q    (GPU-hours per 1,000 completed queries)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

Interval = tuple[float, float]


@dataclass(frozen=True)
class WorkerTimeline:
    """Lifecycle timestamps of one worker."""

    worker_id: str
    requested_at: float
    acquired_at: float
    ready_at: float | None
    released_at: float


@dataclass(frozen=True)
class WorkerCost:
    """Charged seconds of one worker, split by phase."""

    worker_id: str
    acquire_s: float
    load_s: float
    serve_s: float
    idle_s: float

    @property
    def total_s(self) -> float:
        """All charged seconds."""
        return self.acquire_s + self.load_s + self.serve_s + self.idle_s


def _union_length(intervals: Iterable[Interval], lo: float, hi: float) -> float:
    clipped = sorted((max(s, lo), min(e, hi)) for s, e in intervals)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in clipped:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None and cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None and cur_s is not None:
        total += cur_e - cur_s
    return total


def worker_cost(
    timeline: WorkerTimeline, busy: Sequence[Interval]
) -> WorkerCost:
    """Splits one worker's held time into a, l, b and d.

    Args:
        timeline: The worker's timestamps.
        busy: Intervals in which it served at least one request.

    Returns:
        The charged seconds by phase.

    Raises:
        ValueError: If the timestamps are out of order.
    """
    t = timeline
    ready = t.released_at if t.ready_at is None else t.ready_at
    if not t.requested_at <= t.acquired_at <= ready <= t.released_at:
        raise ValueError(f"timestamps out of order for {t.worker_id}")
    serve = _union_length(busy, ready, t.released_at)
    return WorkerCost(
        worker_id=t.worker_id,
        acquire_s=t.acquired_at - t.requested_at,
        load_s=ready - t.acquired_at,
        serve_s=serve,
        idle_s=(t.released_at - ready) - serve,
    )


def gpu_hours(costs: Iterable[WorkerCost]) -> float:
    """G: total charged GPU-hours."""
    return sum(c.total_s for c in costs) / 3600.0


def per_thousand(hours: float, count: int) -> float:
    """C: GPU-hours per 1,000 queries (inf when nothing completed).

    Raises:
        ValueError: If inputs are negative.
    """
    if hours < 0 or count < 0:
        raise ValueError("hours and counts must not be negative")
    return float("inf") if count == 0 else 1000.0 * hours / count


@dataclass(frozen=True)
class CostReport:
    """The NFR-6 numbers for one run."""

    gpu_hours: float
    completed: int
    correct: int
    per_1k_queries: float
    per_1k_correct: float
    tenant_token_share: dict[str, float]


def cost_report(
    timelines: Sequence[WorkerTimeline],
    busy: Mapping[str, Sequence[Interval]],
    completed: int,
    correct: int,
    tenant_tokens: Mapping[str, int],
) -> CostReport:
    """Builds the run's cost report.

    Args:
        timelines: Every worker held during the run.
        busy: Busy intervals by worker id.
        completed: Completed queries.
        correct: Correctly answered queries.
        tenant_tokens: Output tokens served per tenant.

    Returns:
        The report.
    """
    costs = [worker_cost(t, busy.get(t.worker_id, ())) for t in timelines]
    hours = gpu_hours(costs)
    total_tokens = sum(tenant_tokens.values())
    shares = {
        k: (v / total_tokens if total_tokens else 0.0)
        for k, v in tenant_tokens.items()
    }
    return CostReport(
        hours,
        completed,
        correct,
        per_thousand(hours, completed),
        per_thousand(hours, correct),
        shares,
    )
