# ruff: noqa: E501 - SVG markup reads best unwrapped
r"""Evidence pages for the live v2 results: one folder per requirement.

For each requirement: the workload and why it tests the requirement, the
command, the measured numbers and their source files, a chart drawn from
the stored Prometheus data (or the run's own JSON) with the target marked,
and screenshots of the matching Grafana panels for exactly that run's time
window. Nothing is re-run: Prometheus keeps 30 days of data on /data4.

Run inside Slurm (it starts Prometheus and Grafana on the stored data):
    sbatch --partition=low_2h_1gpu --time=00:40:00 --mem=8G \\
        slurm/run.sh uv run python -m tools.evidence
Pages go to docs/results/live-v2/<REQ>/.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess  # nosec B404 - node with fixed arguments
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from deploy.serve import (
    PROMETHEUS_USER,
    Children,
    Paths,
    choose_ports,
    secret_file,
    start_monitoring,
)
from deploy.stack import wait_http

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASKIISC_DATA", "/data4/muthangiv/askiisc"))
LOCUST = DATA / "results" / "locust"
MEASURED = DATA / "results" / "measurements"
OUT = REPO / "docs" / "results" / "live-v2"
TOOLS = Path("/data4/muthangiv/tools")
DASHBOARD = "askiisc-main"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#7d5cc6")  # validated order
TARGET = "#d03b3b"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e5e7eb"


@dataclass
class Line:
    """A time series: a PromQL query and how to show it."""

    query: str
    name: str = ""  # one series; else the label named by `by`
    by: str = ""
    scale: float = 1.0
    top: int = 0  # keep only the largest N series (by their sum)


@dataclass
class Chart:
    """A line chart over the run window, or a bar chart of numbers."""

    title: str
    unit: str
    lines: list[Line] = field(default_factory=list)
    bars: list[tuple[str, float]] = field(default_factory=list)
    targets: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Evidence:
    """One requirement's page."""

    rid: str
    name: str
    target: str
    verdict: str
    workload: str
    why: str
    command: str
    run: str  # locust folder name (window and summary), or ""
    result: str
    panels: list[tuple[int, str]] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering lo..hi."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 0.001:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(round(t, 10))
    return ticks


def _fmt(v: float) -> str:
    if v == int(v) or abs(v) >= 100:  # noqa: PLR2004
        return f"{v:,.0f}"
    return f"{v:.3g}"


def svg_lines(
    chart: Chart, series: list[tuple[str, list[tuple[float, float]]]]
) -> str:
    """A line chart: minutes from the run start against the values."""
    w, h, left, right, top, bottom = 760, 310, 64, 150, 52, 40
    pts = [p for _, s in series for p in s]
    xs = [p[0] for p in pts] or [0.0, 1.0]
    ys = [p[1] for p in pts] + [v for _, v in chart.targets] + [0.0]
    xt = nice_ticks(0.0, max(xs))
    yt = nice_ticks(0.0, max(ys) * 1.05 if max(ys) > 0 else 1.0)

    def sx(x: float) -> float:
        return left + (x - xt[0]) / (xt[-1] - xt[0]) * (w - left - right)

    def sy(y: float) -> float:
        return h - bottom - (y - yt[0]) / (yt[-1] - yt[0]) * (h - top - bottom)  # noqa: E731

    out = [_head(w, h, chart.title)]
    for y in yt:
        out.append(
            f'<line x1="{left}" x2="{w - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="{GRID}"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{sy(y) + 4:.1f}" text-anchor="end" fill="{MUTED}">{_fmt(y)}</text>'
        )
    for x in xt:
        out.append(
            f'<text x="{sx(x):.1f}" y="{h - bottom + 18}" text-anchor="middle" fill="{MUTED}">{_fmt(x)}</text>'
        )
    out.append(
        f'<text x="{(left + w - right) / 2}" y="{h - 6}" text-anchor="middle" fill="{MUTED}">minutes from the start of the run</text>'
    )
    out.append(
        f'<text x="14" y="{top - 14}" fill="{MUTED}">{chart.unit}</text>'
    )
    for label, v in chart.targets:
        out.append(
            f'<line x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{TARGET}" stroke-width="1.5" stroke-dasharray="6 4"/>'
        )
        out.append(
            f'<text x="{w - right + 6}" y="{sy(v) + 4:.1f}" fill="{TARGET}">{label}</text>'
        )
    for i, (name, s) in enumerate(series):
        if not s:
            continue
        color = SERIES[i % len(SERIES)]
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s)
        out.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        ly = top + 14 + i * 18
        out.append(
            f'<line x1="{w - right + 6}" x2="{w - right + 24}" y1="{ly - 4}" y2="{ly - 4}" stroke="{color}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{w - right + 30}" y="{ly}" fill="{INK}">{name}</text>'
        )
    return "\n".join(out) + "\n</svg>\n"


def svg_bars(chart: Chart) -> str:
    """Horizontal bars of measured numbers, targets as dashed marks."""
    w, row, left, top = 760, 34, 260, 40
    h = top + row * len(chart.bars) + 34
    hi = max(
        [v for _, v in chart.bars] + [v for _, v in chart.targets] + [1e-9]
    )
    xt = nice_ticks(0.0, hi * 1.05)

    def sx(x: float) -> float:
        return left + x / xt[-1] * (w - left - 40)

    out = [_head(w, h, chart.title)]
    for x in xt:
        out.append(
            f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{top - 6}" y2="{h - 28}" stroke="{GRID}"/>'
        )
        out.append(
            f'<text x="{sx(x):.1f}" y="{h - 12}" text-anchor="middle" fill="{MUTED}">{_fmt(x)}</text>'
        )
    for i, (label, v) in enumerate(chart.bars):
        y = top + i * row
        out.append(
            f'<text x="{left - 10}" y="{y + 17}" text-anchor="end" fill="{INK}">{label}</text>'
        )
        out.append(
            f'<rect x="{left}" y="{y + 4}" width="{max(1.0, sx(v) - left):.1f}" height="18" rx="4" fill="{SERIES[0]}"/>'
        )
        out.append(
            f'<text x="{sx(v) + 6:.1f}" y="{y + 17}" fill="{INK}">{_fmt(v)}</text>'
        )
    for label, v in chart.targets:
        out.append(
            f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 6}" y2="{h - 28}" stroke="{TARGET}" stroke-width="1.5" stroke-dasharray="6 4"/>'
        )
        out.append(
            f'<text x="{sx(v) + 4:.1f}" y="{top - 10}" fill="{TARGET}">{label}</text>'
        )
    out.append(f'<text x="14" y="{h - 12}" fill="{MUTED}">{chart.unit}</text>')
    return "\n".join(out) + "\n</svg>\n"


def _head(w: int, h: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="system-ui, sans-serif" font-size="12">\n'
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'<text x="14" y="20" font-size="14" font-weight="600" fill="{INK}">{title}</text>'
    )


def summary(run: str) -> dict[str, Any]:
    """A Locust run's summary.json."""
    return json.loads((LOCUST / run / "summary.json").read_text())


def measurement(workload: str, run: str = "") -> dict[str, Any]:
    """The newest measurement of a workload (optionally of one run)."""
    for folder in sorted(MEASURED.glob(f"*-{workload}"), reverse=True):
        data = json.loads((folder / "summary.json").read_text())
        if not run or data.get("run") == run:
            return data
    return {}


def series(
    http: httpx.Client, url: str, line: Line, start: float, end: float
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Prometheus range query over the run window (minutes from start)."""
    r = http.get(
        url + "/api/v1/query_range",
        params={"query": line.query, "start": start, "end": end, "step": 15},
    )
    r.raise_for_status()
    out = []
    for res in r.json()["data"]["result"]:
        name = line.name or res["metric"].get(line.by, "")
        pts = [
            ((float(t) - start) / 60.0, float(v) * line.scale)
            for t, v in res["values"]
            if v not in ("NaN", "+Inf", "-Inf")
        ]
        out.append((name, pts))
    if line.top:
        out.sort(key=lambda s: -sum(v for _, v in s[1]))
        out = out[: line.top]
    return out


def page(ev: Evidence, images: list[tuple[str, str]]) -> str:
    """The README.md of one requirement."""
    lines = [
        f"# {ev.rid}: {ev.name}",
        "",
        f"**Target:** {ev.target}  ",
        f"**Result:** {ev.result}  ",
        f"**Verdict:** {ev.verdict}",
        "",
        "## Workload",
        "",
        ev.workload,
        "",
        f"**Why it tests {ev.rid}:** {ev.why}",
        "",
        "**Command:**",
        "```bash",
        ev.command,
        "```",
        "",
        "## Evidence",
        "",
    ]
    for title, path in images:
        lines += [f"**{title}**", "", f"![{title}]({path})", ""]
    if ev.sources:
        lines += ["## Source files", ""]
        lines += [f"- `{s}`" for s in ev.sources]
        lines.append("")
    return "\n".join(lines)


async def main() -> None:  # noqa: C901, PLR0912, PLR0915 - one script
    """Starts Prometheus and Grafana, then writes every page."""
    from tools.evidence_specs import specs  # noqa: PLC0415

    run_dir = Path(tempfile.mkdtemp(prefix="askiisc-evidence-"))
    paths = Paths(run_dir, DATA / "corpus", DATA / "state", DATA)
    ports = choose_ports()
    children = Children()
    password = secret_file(paths.app / "prometheus_password")
    grafana_password = secret_file(paths.app / "grafana_admin_password")
    try:
        await start_monitoring(children, paths, ports, 0)
        prom = f"http://127.0.0.1:{ports['prometheus']}"
        graf = f"http://127.0.0.1:{ports['grafana']}"
        async with httpx.AsyncClient() as client:
            await wait_http(client, graf + "/api/health", 120)
        shots: list[dict[str, str]] = []
        with httpx.Client(auth=(PROMETHEUS_USER, password), timeout=60) as http:
            for ev in specs():
                folder = OUT / ev.rid
                folder.mkdir(parents=True, exist_ok=True)
                images: list[tuple[str, str]] = []
                start = end = 0.0
                if ev.run:
                    s = summary(ev.run)
                    start, end = float(s["started"]), float(s["finished"])
                for i, chart in enumerate(ev.charts):
                    name = f"chart-{i + 1}.svg"
                    if chart.lines:
                        data = [
                            x
                            for line in chart.lines
                            for x in series(http, prom, line, start, end)
                        ]
                        svg = svg_lines(chart, data)
                    else:
                        svg = svg_bars(chart)
                    (folder / name).write_text(svg)
                    images.append((chart.title, name))
                for panel, title in ev.panels:
                    name = f"grafana-{panel}.png"
                    lo, hi = int((start - 60) * 1000), int((end + 60) * 1000)
                    shots.append({
                        "path": str(folder / name),
                        "url": f"/d/{DASHBOARD}/?orgId=1&viewPanel={panel}&from={lo}&to={hi}",
                    })  # fmt: skip
                    images.append((f"Grafana: {title}", name))
                (folder / "README.md").write_text(page(ev, images))
        listing = run_dir / "shots.json"
        listing.write_text(json.dumps(shots))
        env = {
            **os.environ,
            "PATH": f"{TOOLS}/node-v20.18.0-linux-x64/bin:{os.environ['PATH']}",
            "PLAYWRIGHT_BROWSERS_PATH": str(TOOLS / "ms-playwright"),
            "PLAYWRIGHT_MODULE": str(TOOLS / "pw" / "node_modules" / "playwright-core"),
        }  # fmt: skip
        subprocess.run(  # noqa: S603  # nosec B603 B607 - fixed program and arguments
            [str(TOOLS / "node-v20.18.0-linux-x64" / "bin" / "node"),
             str(REPO / "tools" / "grafana_shots.js"), graf, "admin",
             grafana_password, str(listing)],
            check=True, env=env,
        )  # fmt: skip
    finally:
        await children.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
