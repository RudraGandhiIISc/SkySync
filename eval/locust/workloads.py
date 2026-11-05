"""Load-test workloads: which users, how many, for how long (pure logic).

Each workload targets one or more PRD requirements. A plan is a list of
stages; the Locust shape (locustfile.py) walks through them. Durations are
multiplied by `scale` (use 0.1 for a quick rehearsal), user counts come
from `peak` when it is given.

| workload   | requirements             | what it does                         |
|-----------|--------------------------|---------------------------------------|
| steady     | NFR-1, NFR-4, FR-12      | ramp to N chat users, hold 10 min    |
| ramp       | NFR-3, NFR-4             | no-think users 1, 2, 4 ... 128       |
| cold-start | FR-10, NFR-2, NFR-6      | idle past the timeout, then a burst  |
| spike      | FR-9, NFR-4              | small load, big spike, then idle     |
| overload   | FR-6                     | 1x, 2x, 3x of capacity, no think time|
| fairness   | FR-11                    | one department at 10x another        |
| cache      | FR-7                     | the same few questions again, again |
| access     | FR-2, FR-3, NFR-9/10     | permission probes plus a revocation  |
| revisions  | FR-5, NFR-8              | 50 document revisions/h, freshness   |
| redteam    | NFR-9                    | the 90 red-team cases, once each     |
| failover   | FR-4                     | access service down for 10 s         |
| soak       | NFR-5                    | half load for 4 h, a worker killed   |
| diurnal    | NFR-6                    | a day of traffic, 24x faster         |
| quality    | FR-1, NFR-7              | every held-out question once         |
| auth       | security                 | bad logins, lockout, bad sessions    |
| mixed      | demo                     | a realistic mix of everything        |
| freshness  | FR-5, E5                 | 60 revisions over 20 docs, 1 per 20 s|
| revocations| FR-3, E1                 | 100 scripted revocations, 5 at a time|
| w1         | FR-7, NFR-1, E2          | PRD W1: 60 users, Exp(10 s) think    |
| w2         | FR-6, E3, E4             | PRD W2: W1 at 1x, 2x and 3x          |
| w3         | NFR-6, E3                | PRD W3: 24-hour trace, 24x faster    |

W1 to W3 follow PRD section 7: a 60 s ramp, 10 minutes steady and a 60 s
drain (users stop starting questions; open answers finish). The existing
workloads are kept unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

IDLE_TIMEOUT_S = 300.0
"""The autoscaler's idle timeout (ASKIISC_IDLE_TIMEOUT_S default)."""


@dataclass(frozen=True)
class Stage:
    """Run `users` users of `classes` for `seconds`."""

    seconds: float
    users: int
    classes: tuple[str, ...]
    spawn_rate: float = 10.0
    # Drain: users stay, but start no new question, so open answers finish.
    drain: bool = False


@dataclass(frozen=True)
class Workload:
    """A named plan."""

    name: str
    requirements: str
    stages: tuple[Stage, ...]

    @property
    def seconds(self) -> float:
        """Total duration."""
        return sum(s.seconds for s in self.stages)


def _steady(peak: int, scale: float) -> tuple[Stage, ...]:
    return (
        Stage(60 * scale, peak, ("ChatUser",), spawn_rate=max(1, peak / 60)),
        Stage(600 * scale, peak, ("ChatUser",)),
    )


def _ramp(peak: int, scale: float) -> tuple[Stage, ...]:
    steps = [2**i for i in range(int(math.log2(max(1, peak))) + 1)]
    return tuple(
        Stage(120 * scale, n, ("BurstUser",), spawn_rate=max(1, n))
        for n in steps
    )


def _cold(peak: int, scale: float) -> tuple[Stage, ...]:
    idle = IDLE_TIMEOUT_S + 120
    cycle = (
        Stage(idle * max(scale, 1.0), 0, ("TemporaryUser",)),
        Stage(180 * scale, peak, ("TemporaryUser",), spawn_rate=peak),
    )
    return cycle * 3


def _spike(peak: int, scale: float) -> tuple[Stage, ...]:
    return (
        Stage(120 * scale, max(1, peak // 20), ("ChatUser",)),
        Stage(300 * scale, peak, ("BurstUser",), spawn_rate=peak),
        Stage((IDLE_TIMEOUT_S + 120) * max(scale, 1.0), 0, ("ChatUser",)),
    )


def _overload(peak: int, scale: float) -> tuple[Stage, ...]:
    return tuple(
        Stage(300 * scale, peak * k, ("BurstUser",), spawn_rate=peak * k)
        for k in (1, 2, 3)
    )


def _soak(peak: int, scale: float) -> tuple[Stage, ...]:
    half = max(1, peak // 2)
    return (Stage(4 * 3600 * scale, half, ("ChatUser", "ChaosUser")),)


def _diurnal(peak: int, scale: float) -> tuple[Stage, ...]:
    """24 one-hour slots compressed 24x: quiet nights, busy afternoons."""
    shape = [0, 0, 0, 0, 0, 0, 1, 2, 4, 7, 9, 10, 8, 9, 10, 10, 9, 7, 5, 4, 3,
             2, 1, 0]  # fmt: skip
    return tuple(
        Stage(150 * scale, round(peak * level / 10), ("TemporaryUser",))
        for level in shape
    )


W1_RAMP_S = 60.0
W1_STEADY_S = 600.0
W1_DRAIN_S = 60.0
W2_MULTIPLES = (1, 2, 3)
W3_HOUR_S = 3600 / 24
"""One trace hour of W3 replayed 24x faster: 150 s."""
W3_SHAPE = (0, 0, 0, 0, 0, 0, 1, 1, 2, 10, 20, 12, 2, 0, 0, 6, 8, 4, 2, 1,
            1, 1, 1, 1)  # fmt: skip
"""Users per trace hour, in twentieths of the peak. The median hour is 1,
so the peak is 20x the median; 00:00 to 06:00 and 13:00 to 15:00 have no
traffic (PRD W3: two multi-hour gaps)."""


def _w1(peak: int, scale: float) -> tuple[Stage, ...]:
    classes = ("W1User",)
    return (
        Stage(W1_RAMP_S * scale, peak, classes, max(1, peak / W1_RAMP_S)),
        Stage(W1_STEADY_S * scale, peak, classes),
        Stage(W1_DRAIN_S * scale, peak, classes, drain=True),
    )


def _w2(peak: int, scale: float) -> tuple[Stage, ...]:
    classes = ("W1User",)
    stages: list[Stage] = []
    for k in W2_MULTIPLES:
        rate = max(1, peak / W1_RAMP_S)
        stages.append(Stage(W1_RAMP_S * scale, peak * k, classes, rate))
        stages.append(Stage(W1_STEADY_S * scale, peak * k, classes))
    last = peak * W2_MULTIPLES[-1]
    stages.append(Stage(W1_DRAIN_S * scale, last, classes, drain=True))
    return tuple(stages)


def _w3(peak: int, scale: float) -> tuple[Stage, ...]:
    return tuple(
        Stage(
            W3_HOUR_S * scale,
            round(peak * level / max(W3_SHAPE)),
            ("W1User",),
            max(1, peak / W1_RAMP_S),
        )
        for level in W3_SHAPE
    )


def plan(name: str, peak: int = 0, scale: float = 1.0) -> Workload:
    """The stages of a workload.

    Args:
        name: One of WORKLOADS.
        peak: Peak users (0 means the workload's default).
        scale: Multiplies every duration.

    Returns:
        The plan.

    Raises:
        KeyError: For an unknown workload.
    """
    requirements, default, build = WORKLOADS[name]
    stages = build(peak or default, scale)
    return Workload(name, requirements, stages)


Builder = Callable[[int, float], tuple[Stage, ...]]


def _fixed(classes: tuple[str, ...], seconds: float) -> Builder:
    def build(peak: int, scale: float) -> tuple[Stage, ...]:
        return (Stage(seconds * scale, peak, classes, spawn_rate=peak),)

    return build


WORKLOADS: dict[str, tuple[str, int, Builder]] = {
    "steady": ("NFR-1, NFR-4, FR-12", 60, _steady),
    "ramp": ("NFR-3, NFR-4", 128, _ramp),
    "cold-start": ("FR-10, NFR-2, NFR-6", 20, _cold),
    "spike": ("FR-9, NFR-4", 100, _spike),
    "overload": ("FR-6", 32, _overload),
    "fairness": (
        "FR-11",
        44,
        _fixed(("HeavyTenantUser", "LightTenantUser"), 600),
    ),
    "cache": ("FR-7", 32, _fixed(("RepeatUser",), 600)),
    "access": (
        "FR-2, FR-3, NFR-9, NFR-10",
        30,
        _fixed(("ProbeUser", "RevocationUser"), 900),
    ),
    "revisions": (
        "FR-5, NFR-8",
        10,
        _fixed(("ChatUser", "RevisionUser"), 3600),
    ),
    "redteam": ("NFR-9", 1, _fixed(("RedTeamUser",), 1800)),
    "failover": ("FR-4", 20, _fixed(("TemporaryUser", "ChaosUser"), 600)),
    "soak": ("NFR-5", 60, _soak),
    "diurnal": ("NFR-6", 40, _diurnal),
    "quality": ("FR-1, NFR-7", 4, _fixed(("QualityUser",), 3600)),
    "auth": ("security", 5, _fixed(("AuthAbuseUser",), 120)),
    "mixed": (
        "demo",
        30,
        _fixed(("ChatUser", "TemporaryUser", "RepeatUser", "ProbeUser"), 1800),
    ),
    "freshness": (
        "FR-5, E5",
        8,
        _fixed(("FreshnessReplayUser", "FreshnessReader"), 1500),
    ),
    "revocations": ("FR-3, E1", 5, _fixed(("ScriptedRevocationUser",), 3600)),
    "w1": ("FR-7, NFR-1, E2", 60, _w1),
    "w2": ("FR-6, E3, E4", 60, _w2),
    "w3": ("NFR-6, E3", 60, _w3),
}


def stage_at(workload: Workload, elapsed: float) -> Stage | None:
    """The stage running `elapsed` seconds into the test, or None at the end."""
    for stage in workload.stages:
        if elapsed < stage.seconds:
            return stage
        elapsed -= stage.seconds
    return None
