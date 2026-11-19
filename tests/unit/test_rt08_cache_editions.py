"""Issue #8: a cached answer is not served once an edition it cites ends.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

from typing import Any

import pytest

from askiisc.access.policy import DocPolicy
from askiisc.common.types import Chunk, ChunkId, DocId, Version
from tests import corpus
from tests.gateway_fixtures import Stack, build, final

QUESTION = "What is the mess rebate edition rule?"


def editions(stack: Stack, end: float) -> None:
    """Two editions: the old one ends at `end`, the new one starts then."""
    texts = {
        "ed-old": "Mess rebate edition rule: the mess rebate is 100 a day.",
        "ed-new": "Mess rebate edition rule: the mess rebate is 150 a day.",
    }
    windows = {"ed-old": (0.0, end), "ed-new": (end, None)}
    index = stack.deps.rag._retriever._index  # type: ignore[attr-defined]  # noqa: SLF001
    index.upsert(
        [
            Chunk(
                id=ChunkId(f"{doc}@v1#0"),
                doc_id=DocId(doc),
                version=Version(1),
                text=text,
                page=1,
                valid_from=windows[doc][0],
                valid_to=windows[doc][1],
                embedding=corpus.EMBEDDER.embed_one(text),
            )
            for doc, text in texts.items()
        ]
    )
    for doc in texts:
        stack.w.policies.put(DocPolicy(doc_id=DocId(doc), visibility="public"))


async def ask(stack: Stack) -> dict[str, Any]:
    return final(await stack.ask("u_cds_stu", QUESTION))  # type: ignore[return-value]


def cited(answer: dict[str, Any]) -> list[str]:
    return [c["doc_id"] for c in answer["citations"]]


async def test_issue8_a_cached_answer_ends_with_the_edition_it_cites() -> None:
    stack = build()
    editions(stack, stack.w.clock.now() + 60.0)
    first, again = await ask(stack), await ask(stack)
    assert cited(first) == ["ed-old"]
    assert again["cache_hit"] is True  # control: the cache works
    stack.w.clock.advance(61.0)  # the old edition has ended
    after = await ask(stack)
    assert cited(after) == ["ed-new"]
    assert after["cache_hit"] is False
    assert after["staleness_flag"] is False


async def test_issue8_an_edition_ending_while_the_answer_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = build()
    editions(stack, stack.w.clock.now() + 1.0)
    access = stack.deps.access
    real = access.authorize

    async def slow(token: str, doc_id: DocId, at: float | None = None) -> bool:
        stack.w.clock.advance(2.0)  # the output filter's check takes 2 s
        return await real(token, doc_id, at)

    monkeypatch.setattr(access, "authorize", slow)
    first = await ask(stack)  # flagged before the end; stored after it
    assert cited(first) == ["ed-old"]
    assert first["staleness_flag"] is False
    after = await ask(stack)
    assert after["cache_hit"] is False
    assert cited(after) == ["ed-new"]
