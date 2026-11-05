"""Grounded answering with citations and honest refusals (FR-1).

prepare -> (stream from the LLM) -> finish. The question's time decides
which edition of yearly documents is searched (retrieval/temporal.py): now
by default, the named academic year otherwise, both years for "what
changed" questions. A citation survives only if it
names a chunk that was retrieved for this request; a cited chunk from a
superseded version is flagged stale. With no retrieved chunks the answer is
an abstention and the LLM is not called.

Output guards (NFR-9), applied to the final answer:
- an answer that cites no retrieved chunk is not grounded: it becomes the
  refusal (this also stops "reply only with X" injections in the question);
- an answer that repeats a phrase of the system prompt becomes the refusal;
- links that are not in a cited chunk, and HTML tags, are removed.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, field

from askiisc.common.clock import Clock
from askiisc.common.targets import MIN_GROUNDED_SHARE, REFUSAL_TEXT
from askiisc.common.types import (
    Answer,
    Chunk,
    ChunkId,
    Citation,
    DocId,
    DocRef,
    Version,
)
from askiisc.llm.client import LLMClient, Message, StreamState, Usage
from askiisc.llm.prompts import (
    EDITION_RULE,
    SYSTEM_PROMPT,
    DocLabels,
    build_messages,
    fit_context,
    retrieval_query,
)
from askiisc.observability.stages import StageTimer
from askiisc.retrieval.search import Retriever, SearchResult
from askiisc.retrieval.store import ScoredChunk
from askiisc.retrieval.temporal import TimeScope, time_scope, without_time

# One bracket may hold several ids: [chunk:a] or [chunk:a, chunk:b] or
# [chunk:a, b] (models write all three).
CITATION = re.compile(r"\[(chunk:[^\]]+)\]")
_CITED_ID = re.compile(r"\s*(?:chunk:)?([^\s,;\]]+)")


_URL = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_WORDS = re.compile(r"[a-z0-9]+")
_CLAIM_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
# Words that carry no claim of their own: an answer may add them freely.
_FILLER_WORDS = """a an the of in on at to for from by with and or is are was
were be been this that these those it its as which who whom what when where
how why do does did has have had not no yes can may must will would should
could there their they them he she his her you your we our us also than
then into about over under per each any all some more most other such only
same both between during before after very just one two based provided
documents document according answer question see"""
_FILLER = frozenset(_FILLER_WORDS.split())
STEM_CHARS = 5
ECHO_WORDS = 6


def _shingles(text: str, n: int = ECHO_WORDS) -> set[tuple[str, ...]]:
    words = _WORDS.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


# Six-word runs of the rules (the refusal sentence is allowed on its own).
_SYSTEM_SHINGLES = _shingles(
    SYSTEM_PROMPT.replace(REFUSAL_TEXT, " ").replace(EDITION_RULE, " ")
)


def echoes_system_prompt(text: str) -> bool:
    """Whether the text repeats six words in a row of the system prompt."""
    return not _SYSTEM_SHINGLES.isdisjoint(_shingles(text))


def scrub(text: str, sources: Iterable[str]) -> str:
    """Removes HTML tags and links that do not appear in the sources."""
    allowed = "\n".join(sources)
    text = _TAG.sub("", text)
    return _URL.sub(
        lambda m: m.group(0) if m.group(0) in allowed else "[link removed]",
        text,
    )


def cited_ids(text: str) -> list[ChunkId]:
    """Every chunk id cited in the text, in order of mention."""
    return [
        ChunkId(i)
        for m in CITATION.finditer(text)
        for i in _CITED_ID.findall(m.group(1))
    ]


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


def resolve(cited: str, chunks: Sequence[Chunk]) -> Chunk | None:
    """The chunk a citation names: its prompt position ("3") or its id."""
    if cited.isdigit():
        index = int(cited) - 1
        return chunks[index] if 0 <= index < len(chunks) else None
    return next((c for c in chunks if c.id == cited), None)


def canonical_marks(text: str, chunks: Sequence[Chunk]) -> str:
    """Rewrites [chunk:3] marks to real chunk ids; drops unknown ones."""

    def mark(match: re.Match[str]) -> str:
        found = [resolve(i, chunks) for i in _CITED_ID.findall(match.group(1))]
        ids = [f"chunk:{c.id}" for c in found if c is not None]
        return f"[{', '.join(ids)}]" if ids else ""

    return CITATION.sub(mark, text)


Successor = Callable[[DocId, Version], DocRef | None]
"""Names the version that replaced a document version (ingest/lineage)."""


def parse_answer(
    text: str,
    chunks: tuple[Chunk, ...],
    now: float,
    successor: Successor | None = None,
) -> Answer:
    """Turns model output into an Answer.

    Citations may name a chunk by its position in the prompt or by its id;
    the returned text always carries the chunk ids.

    Args:
        text: The generated text.
        chunks: The chunks retrieved for this request, in prompt order.
        now: Current time, for stale flags.
        successor: Names what replaced a stale citation's version.

    Returns:
        The answer with only valid citations, in order of first mention,
        or the refusal if it cites nothing retrieved or repeats the rules.
    """
    refusal = Answer(text=REFUSAL_TEXT, abstained=True)
    if not chunks or text.strip() == REFUSAL_TEXT:
        return refusal
    if echoes_system_prompt(text):
        return refusal
    text = canonical_marks(text, chunks)
    by_id = {c.id: c for c in chunks}
    citations: list[Citation] = []
    seen: set[ChunkId] = set()
    for cited in cited_ids(text):
        chunk = by_id.get(cited)
        if chunk is None or chunk.id in seen:
            continue
        seen.add(chunk.id)
        stale = chunk.valid_to is not None and chunk.valid_to <= now
        replaced = (
            successor(chunk.doc_id, chunk.version)
            if stale and successor is not None
            else None
        )
        citations.append(
            Citation(
                doc_id=chunk.doc_id,
                version=chunk.version,
                chunk_id=chunk.id,
                page=chunk.page,
                is_stale=stale,
                superseded_by=replaced,
            )
        )
    if not citations:
        return refusal
    sources = [by_id[c.chunk_id].text for c in citations]
    if grounded_share(CITATION.sub(" ", text), sources) < MIN_GROUNDED_SHARE:
        return refusal
    return Answer(text=scrub(text, sources).strip(), citations=tuple(citations))


def _claim_words(text: str) -> list[str]:
    return [
        w
        for w in _CLAIM_WORD.findall(text.lower())
        if w not in _FILLER and (len(w) > 2 or w.isdigit())  # noqa: PLR2004
    ]


def grounded_share(text: str, sources: Sequence[str]) -> float:
    """Share of an answer's content words found in the chunks it cites.

    A word counts as found if a source has it, or a source word of the
    same first STEM_CHARS letters (plurals, tenses). An answer with no
    content words is fully grounded.
    """
    words = _claim_words(text)
    if not words:
        return 1.0
    seen = set(_claim_words(" ".join(sources)))
    stems = {w[:STEM_CHARS] for w in seen if len(w) > STEM_CHARS}
    found = sum(
        w in seen or (len(w) > STEM_CHARS and w[:STEM_CHARS] in stems)
        for w in words
    )
    return found / len(words)


def merge_hits(results: Sequence[SearchResult]) -> list[ScoredChunk]:
    """Interleaves the hits of several as-of searches, best first.

    Each year's best chunk comes before any year's second best, so every
    edition asked about reaches the prompt.
    """
    merged: list[ScoredChunk] = []
    seen: set[ChunkId] = set()
    depth = max((len(r.hits) for r in results), default=0)
    for i in range(depth):
        for result in results:
            if i < len(result.hits) and result.hits[i].chunk.id not in seen:
                seen.add(result.hits[i].chunk.id)
                merged.append(result.hits[i])
    return merged


def _stage(
    timer: StageTimer | None, stage: str
) -> contextlib.AbstractContextManager[None]:
    """Times a block on the request's timer, or does nothing without one."""
    return contextlib.nullcontext() if timer is None else timer.measure(stage)


class RagAnswerer:
    """Runs retrieval, prompting and generation for one question."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        clock: Clock,
        labels: DocLabels | None = None,
        successor: Successor | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._clock = clock
        self._labels = labels
        self._successor = successor

    @property
    def model(self) -> str:
        """The served model name."""
        return self._llm.model

    async def prepare(
        self,
        token: str,
        question: str,
        at: float | None = None,
        history: Sequence[Message] = (),
        timer: StageTimer | None = None,
    ) -> Prepared:
        """Retrieves chunks and builds the prompt.

        Args:
            token: The user's token.
            question: The question.
            at: As-of time; None means now.
            history: Earlier turns of a chat, oldest first.
            timer: The request's stage timer (access, embed, search and
                rerank are added to it), if it keeps one.

        Raises:
            AccessDenied: If the access service cannot answer.
        """
        with _stage(timer, "prompt"):
            query = retrieval_query(question, history)
            now = self._clock.now()
            scope = (
                TimeScope((at,)) if at is not None else time_scope(query, now)
            )
            text = without_time(query) if scope.label else query
        results = [
            await self._search(token, text, instant, timer)
            for instant in scope.instants
        ]
        with _stage(timer, "prompt"):
            chunks = fit_context([hit.chunk for hit in merge_hits(results)])
            messages = build_messages(
                chunks, question, history, self._labels, now, scope.label
            )
        return Prepared(
            chunks=chunks,
            messages=messages,
            policy_version=results[-1].policy_version,
            at=results[-1].at,
        )

    async def _search(
        self,
        token: str,
        text: str,
        at: float | None,
        timer: StageTimer | None,
    ) -> SearchResult:
        # Without a timer the call keeps its three-argument form, which
        # any retriever (and the simple test recorders) accepts.
        if timer is None:
            return await self._retriever.search(token, text, at)
        return await self._retriever.search(token, text, at, timer)

    def generate(
        self, prepared: Prepared, cache_salt: str | None = None
    ) -> AsyncIterator[str]:
        """Streams the model's answer text."""
        return self._llm.stream(prepared.messages, prepared.state, cache_salt)

    def finish(self, prepared: Prepared) -> Answer:
        """Parses the streamed text into an Answer."""
        text = "".join(prepared.state.text)
        return parse_answer(
            text, prepared.chunks, self._clock.now(), self._successor
        )

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
