"""Fixed phrases Locust users send, and the ids that stand for questions.

Kept apart from the locustfile so that tools which must know every string
a load test can send (eval/audit_check.py) can read them without importing
Locust.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

FOLLOW_UPS = (
    "Can you explain that in more detail?",
    "Which document says this?",
    "Does this apply to PhD students too?",
    "What are the deadlines for that?",
)


def question_ids(questions: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Maps each question's text to the id logged in its place.

    Args:
        questions: Dataset rows with "id" and "question".

    Returns:
        Text to id, including "follow-up-N" for the fixed follow-ups.
    """
    ids = {str(q["question"]): str(q["id"]) for q in questions}
    for n, text in enumerate(FOLLOW_UPS, 1):
        ids[text] = f"follow-up-{n}"
    return ids
