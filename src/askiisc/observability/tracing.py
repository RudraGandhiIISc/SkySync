"""OpenTelemetry spans around ask -> access -> search -> llm -> filter.

Span attributes never carry query or answer text: `span` refuses any
attribute whose name suggests text.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Tracer

FORBIDDEN_ATTRIBUTES = (
    "query",
    "question",
    "answer",
    "text",
    "prompt",
    "token",
    "secret",
)

AttrValue = str | int | float | bool


def make_tracer(exporter: SpanExporter | None = None) -> Tracer:
    """Builds a tracer with its own provider (no global state).

    Args:
        exporter: Where spans go; none means spans are dropped.

    Returns:
        A tracer.
    """
    provider = TracerProvider()
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("askiisc")


def check_attributes(attributes: dict[str, AttrValue]) -> None:
    """Rejects attribute names that could carry user text.

    Raises:
        ValueError: On a forbidden name.
    """
    for name in attributes:
        lowered = name.lower()
        if any(word in lowered for word in FORBIDDEN_ATTRIBUTES):
            raise ValueError(f"span attribute {name!r} could carry user text")


@contextmanager
def span(tracer: Tracer, name: str, **attributes: AttrValue) -> Iterator[Span]:
    """Opens a span with safe attributes.

    Raises:
        ValueError: On a forbidden attribute name.
    """
    check_attributes(attributes)
    with tracer.start_as_current_span(name, attributes=attributes) as current:
        yield current
