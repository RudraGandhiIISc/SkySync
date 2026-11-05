"""Chat routes: chats, messages (streamed), temporary chats, ratings.

A message goes through the same gateway path as POST /v1/ask (auth, load
limits, access filter, cache, retrieval, fair queue, LLM, output filter,
audit) with the chat's earlier turns as history. Temporary chats are
never stored; their history comes from the browser.

When a chat is opened, any earlier answer that cites a document the user
can no longer read is hidden, and such answers are never sent back to
the model as history (FR-3).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from askiisc.common.errors import AccessDenied, NotFound
from askiisc.common.targets import CHAT_HISTORY_MESSAGES
from askiisc.gateway.app import (
    REQUEST_ID_HEADER,
    TURN_MAX_CHARS,
    AskRequest,
    Origin,
    Turn,
    ask,
    sse,
)
from askiisc.web.app import Me, WebDeps, error, is_real, run
from askiisc.web.store import StoredMessage

TITLE_CHARS = 60
REVOKED_TEXT = (
    "This answer cited a document you can no longer read, so it is hidden."
)
UNAVAILABLE = "the access service is unavailable; try again shortly"


class NewChat(BaseModel):
    """POST /api/chats body."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="New chat", min_length=1, max_length=200)


class Rename(BaseModel):
    """PATCH /api/chats/{id} body."""

    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)


class Question(BaseModel):
    """POST /api/chats/{id}/messages body."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=8192)


class TemporaryQuestion(Question):
    """POST /api/temporary body: the browser keeps the history."""

    history: list[Turn] = Field(default_factory=list, max_length=20)


class Rating(BaseModel):
    """POST /api/messages/{id}/rating body."""

    model_config = ConfigDict(extra="forbid")
    rating: int = Field(ge=-1, le=1)


def visible(
    messages: list[StoredMessage], readable: frozenset[str]
) -> list[dict[str, Any]]:
    """Messages as the user may see them now.

    Args:
        messages: Stored messages, oldest first.
        readable: Documents the user can read now.

    Returns:
        JSON-ready messages; answers citing an unreadable document are
        replaced by a notice.
    """
    out = []
    for m in messages:
        cited = {c["doc_id"] for c in m.meta.get("citations", [])}
        hidden = m.role == "assistant" and not cited <= readable
        out.append(
            {
                "message_id": m.message_id,
                "role": m.role,
                "content": REVOKED_TEXT if hidden else m.content,
                "meta": {"hidden": True} if hidden else m.meta,
                "created": m.created,
                "rating": m.rating,
            }
        )
    return out


def history_of(shown: list[dict[str, Any]]) -> list[Turn]:
    """Earlier turns to send with a follow-up (hidden answers left out).

    Only the turns the prompt uses are sent (the last
    CHAT_HISTORY_MESSAGES), each cut to the request's limit, so a long chat
    or a long earlier question never fails the request's validation.
    """
    turns = [m for m in shown if not m["meta"].get("hidden") and m["content"]]
    return [
        Turn(role=m["role"], content=m["content"][:TURN_MAX_CHARS])
        for m in turns[-CHAT_HISTORY_MESSAGES:]
    ]


FRAGMENT_WORDS = 8
_CHUNK_MARK = re.compile(r"\s*\[(chunk:[^\]]+)\]")


def export_markdown(title: str, messages: list[dict[str, Any]]) -> str:
    """A chat as Markdown: answers with numbered sources, as in the app."""
    lines = [f"# {title}", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"**You:** {m['content']}", ""]
            continue
        cited = m["meta"].get("citations", [])
        number = {str(c["chunk_id"]): i for i, c in enumerate(cited, 1)}

        def marks(found: re.Match[str], number: dict[str, int] = number) -> str:
            ids = [
                x.strip().removeprefix("chunk:") for x in found[1].split(",")
            ]
            nums = dict.fromkeys(number[i] for i in ids if i in number)
            return "".join(f" [{n}]" for n in nums)

        lines += [f"**Ask IISc:** {_CHUNK_MARK.sub(marks, m['content'])}", ""]
        for n, c in enumerate(cited, 1):
            name = c.get("title") or c["doc_id"]
            where = c.get("link") or c.get("url") or ""
            edition = f", {c['edition']}" if c.get("edition") else ""
            lines.append(
                f"{n}. {name}{edition}" + (f" <{where}>" if where else "")
            )
        if cited:
            lines.append("")
    return "\n".join(lines)


def source_link(url: str, kind: str, page: int, text: str | None) -> str:
    """A link to the exact place a citation came from.

    PDFs open at the cited page (#page=N). Web pages get a text fragment
    (#:~:text=...) with the opening words of the cited chunk, so the
    browser scrolls to the passage and highlights it.

    Args:
        url: The document's address (empty for uploaded documents).
        kind: "pdf", "html" or another file kind.
        page: The cited page, from 1.
        text: The cited chunk's text, if known.

    Returns:
        The link, or "" when the document has no address.
    """
    if not url:
        return ""
    if kind == "pdf":
        return f"{url}#page={max(1, page)}"
    if kind != "html" or not text:
        return url
    for line in text.splitlines():
        words = [w for w in line.split() if any(c.isalpha() for c in w)]
        if len(words) >= 4:  # noqa: PLR2004 - short lines are menus
            phrase = " ".join(line.split()[:FRAGMENT_WORDS]).rstrip(".,;:")
            return f"{url}#:~:text={quote(phrase, safe='').replace('-', '%2D')}"
    return url


def enrich(web: WebDeps, payload: dict[str, Any]) -> dict[str, Any]:
    """Adds document titles and exact source links to the citations."""
    citations = []
    for c in payload["citations"]:
        doc = web.docs(str(c["doc_id"]))
        link = source_link(
            doc.get("url", ""),
            doc.get("kind", ""),
            int(c.get("page") or 1),
            web.chunk_text(str(c.get("chunk_id", ""))),
        )
        citations.append({**c, **doc, "link": link})
    return {**payload, "citations": citations}


OnAnswer = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


def relay(web: WebDeps, response: Response, on_answer: OnAnswer) -> Response:
    """Passes a gateway stream through, enriching and saving the answer.

    A rejection (401, 403, 413, 429) is returned unchanged.
    """
    if not isinstance(response, StreamingResponse):
        return response
    upstream = response.body_iterator

    async def events() -> AsyncIterator[bytes]:
        buffer = b""
        async for piece in upstream:
            buffer += piece if isinstance(piece, bytes) else str(piece).encode()
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                if not block.startswith(b"event: answer"):
                    yield block + b"\n\n"
                    continue
                payload = enrich(web, json.loads(block.split(b"data: ", 1)[1]))
                payload.pop("retrieved", None)
                yield sse("answer", payload)
                saved = await on_answer(payload)
                if saved is not None:
                    yield sse("saved", saved)

    # The audit record's id (FR-12) travels with the relayed stream.
    headers = (
        {REQUEST_ID_HEADER: response.headers[REQUEST_ID_HEADER]}
        if REQUEST_ID_HEADER in response.headers
        else None
    )
    return StreamingResponse(
        events(), media_type="text/event-stream", headers=headers
    )


def chat_router(  # noqa: PLR0915 - one route table
    web: WebDeps, current: Callable[[Request], Awaitable[Me]]
) -> APIRouter:
    """The /api/chats, /api/temporary and /api/messages routes."""
    router = APIRouter(prefix="/api")
    user_dep = Depends(current)

    async def readable(me: Me) -> frozenset[str]:
        allowed = await web.gateway.access.filter(me.token)
        return frozenset(str(d) for d in allowed.doc_ids)

    async def shown(me: Me, chat_id: str) -> list[dict[str, Any]]:
        messages = await run(web.store.messages, chat_id)
        return visible(messages, await readable(me))

    @router.get("/chats")
    async def list_chats(q: str = "", me: Me = user_dep) -> dict[str, Any]:
        chats = await run(web.store.chats, me.user_id)
        needle = q.strip().lower()
        return {
            "chats": [
                {"chat_id": c.chat_id, "title": c.title, "updated": c.updated}
                for c in chats
                if needle in c.title.lower()
            ]
        }

    @router.post("/chats", status_code=201)
    async def new_chat(body: NewChat, me: Me = user_dep) -> dict[str, Any]:
        chat = await run(
            web.store.create_chat, me.user_id, body.title, web.clock.now()
        )
        web.m.chat_events.labels("created").inc()
        return {"chat_id": chat.chat_id, "title": chat.title}

    @router.get("/chats/{chat_id}")
    async def open_chat(chat_id: str, me: Me = user_dep) -> Response:
        try:
            chat = await run(web.store.chat, me.user_id, chat_id)
            messages = await shown(me, chat_id)
        except NotFound:
            return error(404, "no such chat")
        except AccessDenied:
            return error(503, UNAVAILABLE)
        return JSONResponse(
            {
                "chat_id": chat.chat_id,
                "title": chat.title,
                "messages": messages,
            }
        )

    @router.patch("/chats/{chat_id}")
    async def rename(chat_id: str, body: Rename, me: Me = user_dep) -> Response:
        try:
            await run(web.store.rename_chat, me.user_id, chat_id, body.title)
        except NotFound:
            return error(404, "no such chat")
        web.m.chat_events.labels("renamed").inc()
        return JSONResponse({"ok": True})

    @router.delete("/chats/{chat_id}")
    async def delete(chat_id: str, me: Me = user_dep) -> Response:
        try:
            await run(web.store.delete_chat, me.user_id, chat_id)
        except NotFound:
            return error(404, "no such chat")
        web.m.chat_events.labels("deleted").inc()
        return JSONResponse({"ok": True})

    @router.delete("/chats")
    async def delete_all(me: Me = user_dep) -> dict[str, int]:
        count = await run(web.store.delete_all_chats, me.user_id)
        web.m.chat_events.labels("deleted").inc(count)
        return {"deleted": count}

    @router.get("/chats/{chat_id}/export")
    async def export(chat_id: str, me: Me = user_dep) -> Response:
        try:
            chat = await run(web.store.chat, me.user_id, chat_id)
            messages = await shown(me, chat_id)
        except NotFound:
            return error(404, "no such chat")
        except AccessDenied:
            return error(503, UNAVAILABLE)
        return Response(
            export_markdown(chat.title, messages),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{chat_id}.md"'
            },
        )

    @router.post("/chats/{chat_id}/messages")
    async def send(chat_id: str, body: Question, me: Me = user_dep) -> Response:
        return await _send(web, me, chat_id, body.question, shown)

    @router.post("/chats/{chat_id}/regenerate")
    async def regenerate(chat_id: str, me: Me = user_dep) -> Response:
        try:
            await run(web.store.chat, me.user_id, chat_id)
        except NotFound:
            return error(404, "no such chat")
        messages = await run(web.store.messages, chat_id)
        if len(messages) < 2 or [m.role for m in messages[-2:]] != [  # noqa: PLR2004
            "user",
            "assistant",
        ]:
            return error(400, "nothing to regenerate")
        question, answer = messages[-2], messages[-1]
        return await _send(
            web,
            me,
            chat_id,
            question.content,
            shown,
            replaces=(question.message_id, answer.message_id),
        )

    @router.post("/temporary")
    async def temporary(body: TemporaryQuestion, me: Me = user_dep) -> Response:
        request = AskRequest(question=body.question, history=body.history)
        origin = Origin("temporary", not is_real(me.user_id))
        response = await ask(web.gateway, request, f"Bearer {me.token}", origin)

        async def nothing(payload: dict[str, Any]) -> None:
            return None

        web.m.chat_events.labels("temporary_message").inc()
        return relay(web, response, nothing)

    @router.post("/messages/{message_id}/rating")
    async def rate(
        message_id: str, body: Rating, me: Me = user_dep
    ) -> Response:
        try:
            await run(web.store.rate, me.user_id, message_id, body.rating)
        except NotFound:
            return error(404, "no such answer")
        web.m.ratings.labels(str(body.rating)).inc()
        return JSONResponse({"ok": True})

    return router


async def _send(
    web: WebDeps,
    me: Me,
    chat_id: str,
    question: str,
    shown: Callable[[Me, str], Awaitable[list[dict[str, Any]]]],
    replaces: tuple[str, str] | None = None,
) -> Response:
    """Asks the gateway and saves the turn once the answer is streaming.

    Args:
        web: The web app's dependencies.
        me: The user.
        chat_id: The chat.
        question: The question to ask.
        shown: The chat's messages as the user may see them.
        replaces: The question and answer a regeneration replaces. They
            stay out of the history and are deleted only once the gateway
            has accepted the request, so a refusal (429, access service
            down) loses nothing.
    """
    try:
        chat = await run(web.store.chat, me.user_id, chat_id)
        earlier = await shown(me, chat_id)
        if replaces is not None:
            earlier = [m for m in earlier if m["message_id"] not in replaces]
        history = history_of(earlier)
    except NotFound:
        return error(404, "no such chat")
    except AccessDenied:
        return error(503, UNAVAILABLE)
    request = AskRequest(question=question, history=history)
    origin = Origin("chat", not is_real(me.user_id))
    response = await ask(web.gateway, request, f"Bearer {me.token}", origin)
    if not isinstance(response, StreamingResponse):
        return response
    store, now = web.store, web.clock.now()
    for message_id in replaces or ():
        await run(store.delete_message, chat_id, message_id)
    question_id = await run(
        store.add_message, chat_id, "user", question, {}, now
    )
    if not history and chat.title == "New chat":
        await run(
            store.rename_chat, me.user_id, chat_id, question[:TITLE_CHARS]
        )
    web.m.chat_events.labels("message").inc()

    async def save(payload: dict[str, Any]) -> dict[str, Any]:
        meta = {k: v for k, v in payload.items() if k != "text"}
        answer_id = await run(
            store.add_message,
            chat_id,
            "assistant",
            payload["text"],
            meta,
            web.clock.now(),
        )
        title = (await run(store.chat, me.user_id, chat_id)).title
        return {
            "question_id": question_id,
            "answer_id": answer_id,
            "title": title,
        }

    return relay(web, response, save)
