"""The scaling policy as a pure function (FR-9, FR-10).

Rules:
- Never more than `max_workers` (default 2) workers.
- Zero workers and demand that arrived after the last action: start one
  at once (the first request is held, not failed), even inside the
  hysteresis window.
- Otherwise at most one action per `hysteresis_s`.
- Scale out when request rate per worker or queue depth per worker is
  above its threshold.
- Scale in only with nothing in flight or queued: to zero after
  `idle_timeout_s` without traffic, or by the surplus after demand has been
  low for `idle_timeout_s`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from askiisc.common.targets import AUTOSCALER_HYSTERESIS_S, MAX_GPU_WORKERS


@dataclass(frozen=True)
class ScalingConfig:
    """Policy thresholds."""

    max_workers: int = MAX_GPU_WORKERS
    rate_per_worker: float = 4.0
    queue_per_worker: int = 16
    idle_timeout_s: float = 300.0
    hysteresis_s: float = AUTOSCALER_HYSTERESIS_S

    def __post_init__(self) -> None:
        if not 0 < self.max_workers <= MAX_GPU_WORKERS:
            raise ValueError(f"max_workers must be 1..{MAX_GPU_WORKERS}")


@dataclass(frozen=True)
class Observation:
    """What the controller sees at one instant.

    Attributes:
        now: Current time.
        request_rate: Arrivals per second over the recent window.
        queue_depth: Requests waiting for an LLM slot or a worker.
        in_flight: Requests being generated.
        workers: Active workers (starting, loading or ready).
        last_traffic: Time of the last arrival, or None.
        last_action: Time of the last scaling action, or None.
        low_since: Since when demand has needed fewer workers than active,
            or None.
    """

    now: float
    request_rate: float
    queue_depth: int
    in_flight: int
    workers: int
    last_traffic: float | None = None
    last_action: float | None = None
    low_since: float | None = None


@dataclass(frozen=True)
class ScaleOut:
    """Start `count` workers."""

    count: int


@dataclass(frozen=True)
class ScaleIn:
    """Release `count` workers."""

    count: int


@dataclass(frozen=True)
class Hold:
    """Do nothing."""


Decision = ScaleOut | ScaleIn | Hold


def needed_workers(obs: Observation, config: ScalingConfig) -> int:
    """Workers the current demand needs, between 0 and max_workers."""
    if obs.request_rate <= 0 and obs.queue_depth == 0 and obs.in_flight == 0:
        return 0
    by_rate = obs.request_rate / config.rate_per_worker
    by_queue = obs.queue_depth / config.queue_per_worker
    return min(config.max_workers, max(1, math.ceil(max(by_rate, by_queue))))


def _new_demand(obs: Observation) -> bool:
    """Demand that arrived after the last action (or is waiting now).

    The arrival-rate window can still show traffic from before a scale-in;
    only a newer arrival may trigger a cold start, or the pool would flap.
    """
    if obs.queue_depth > 0 or obs.in_flight > 0:
        return True
    if obs.last_traffic is None:
        return False
    return obs.last_action is None or obs.last_traffic > obs.last_action


def decide(obs: Observation, config: ScalingConfig) -> Decision:
    """Chooses the next scaling action.

    Args:
        obs: The current observation.
        config: Thresholds.

    Returns:
        ScaleOut, ScaleIn or Hold.
    """
    need = needed_workers(obs, config)
    if obs.workers == 0:
        return ScaleOut(1) if _new_demand(obs) else Hold()
    if (
        obs.last_action is not None
        and obs.now - obs.last_action < config.hysteresis_s
    ):
        return Hold()
    if need > obs.workers:
        return ScaleOut(min(need, config.max_workers) - obs.workers)
    busy = obs.in_flight > 0 or obs.queue_depth > 0
    if busy:
        return Hold()
    idle = (
        obs.last_traffic is None
        or obs.now - obs.last_traffic >= config.idle_timeout_s
    )
    if idle:
        return ScaleIn(obs.workers)
    low_long = (
        obs.low_since is not None
        and obs.now - obs.low_since >= config.idle_timeout_s
    )
    if need < obs.workers and low_long:
        return ScaleIn(obs.workers - max(need, 1))
    return Hold()
