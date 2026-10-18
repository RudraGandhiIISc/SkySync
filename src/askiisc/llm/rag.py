"""Grounded answering with citations and honest refusals (FR-1).

prepare -> (stream from the LLM) -> finish. A citation survives only if it
names a chunk that was retrieved for this request; a cited chunk from a
superseded version is flagged stale. With no retrieved chunks the answer is
an abstention and the LLM is not called.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from askiisc.common.clock import Clock
from askiisc.common.targets import REFUSAL_TEXT
from askiisc.common.types import Answer, Chunk, ChunkId, Citation
from askiisc.llm.client import LLMClient, Message, StreamState, Usage
from askiisc.llm.prompts import build_messages
from askiisc.retrieval.search import Retriever

CITATION = re.compile(r"\[chunk:([^\]\s]+)\]")


@dataclass
class Prepared:
    """A request ready for generation."""

    chunks: tuple[Chunk, ...]
    messages: list[Message]
    policy_version: int
    at: float
    state: StreamState = field(default_factory=StreamState)

    @property
    def needs_llm(self) -> bool:
        """False when nothing was retrieved (abstain without the LLM)."""
        return bool(self.chunks)


def parse_answer(text: str, chunks: tuple[Chunk, ...], now: float) -> Answer:
    """Turns model output into an Answer.

    Args:
        text: The generated text.
        chunks: The chunks retrieved for this request.
        now: Current time, for stale flags.

    Returns:
        The answer with only valid citations, in order of first mention.
    """
    if not chunks or text.strip() == REFUSAL_TEXT:
        return Answer(text=REFUSAL_TEXT, abstained=True)
    by_id = {c.id: c for c in chunks}
    citations: list[Citation] = []
    seen: set[ChunkId] = set()
    for match in CITATION.finditer(text):
        chunk = by_id.get(ChunkId(match.group(1)))
        if chunk is None or chunk.id in seen:
            continue
        seen.add(chunk.id)
        stale = chunk.valid_to is not None and chunk.valid_to <= now
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                version=chunk.version,
                chunk_id=chunk.id,
                page=chunk.page,
                is_stale=stale,
            )
        )
    return Answer(text=text.strip(), citations=tuple(citations))


class RagAnswerer:
    """Runs retrieval, prompting and generation for one question."""

    def __init__(
        self, retriever: Retriever, llm: LLMClient, clock: Clock
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._clock = clock

    @property
    def model(self) -> str:
        """The served model name."""
        return self._llm.model

    async def prepare(
        self, token: str, question: str, at: float | None = None
    ) -> Prepared:
        """Retrieves chunks and builds the prompt.

        Raises:
            AccessDenied: If the access service cannot answer.
        """
        result = await self._retriever.search(token, question, at)
        chunks = tuple(hit.chunk for hit in result.hits)
        return Prepared(
            chunks=chunks,
            messages=build_messages(chunks, question),
            policy_version=result.policy_version,
            at=result.at,
        )

    def generate(
        self, prepared: Prepared, cache_salt: str | None = None
    ) -> AsyncIterator[str]:
        """Streams the model's answer text."""
        return self._llm.stream(prepared.messages, prepared.state, cache_salt)

    def finish(self, prepared: Prepared) -> Answer:
        """Parses the streamed text into an Answer."""
        text = "".join(prepared.state.text)
        return parse_answer(text, prepared.chunks, self._clock.now())

    async def answer(
        self,
        token: str,
        question: str,
        at: float | None = None,
        cache_salt: str | None = None,
    ) -> tuple[Answer, Prepared]:
        """Answers a question without streaming.

        Returns:
            The answer and the prepared request (chunks, usage, versions).
        """
        prepared = await self.prepare(token, question, at)
        if prepared.needs_llm:
            async for _ in self.generate(prepared, cache_salt):
                pass
        return self.finish(prepared), prepared

    @staticmethod
    def usage(prepared: Prepared) -> Usage:
        """Token usage of a finished request."""
        return prepared.state.usage
