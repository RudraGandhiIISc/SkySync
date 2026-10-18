"""Re-checks every cited document before an answer leaves the gateway.

If any cited document is no longer readable by the user (for example a
revocation during generation), or the check itself fails, the answer is
replaced by a safe refusal. Fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from askiisc.access.client import AccessAPI
from askiisc.common.errors import AccessDenied
from askiisc.common.targets import REFUSAL_TEXT
from askiisc.common.types import Answer

SAFE_REFUSAL = Answer(text=REFUSAL_TEXT, abstained=True)


@dataclass(frozen=True)
class FilterResult:
    """The answer to send and whether anything was withheld."""

    answer: Answer
    withheld: bool


async def filter_answer(
    access: AccessAPI, token: str, answer: Answer
) -> FilterResult:
    """Checks every cited document with the access service.

    Args:
        access: The access API.
        token: The user's verified token.
        answer: The parsed answer.

    Returns:
        The original answer, or the safe refusal if anything failed.
    """
    try:
        for doc_id in {c.doc_id for c in answer.citations}:
            if not await access.authorize(token, doc_id):
                return FilterResult(SAFE_REFUSAL, withheld=True)
    except AccessDenied:
        return FilterResult(SAFE_REFUSAL, withheld=True)
    return FilterResult(answer, withheld=False)
