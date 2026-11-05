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


class ZipfQuestions:
    """Draws questions with Zipf popularity and a set repeat rate (W1).

    PRD section 7, W1: questions from the held-out set with Zipf(1.1)
    popularity and about 60% repeats. Questions get a popularity rank in a
    seeded order. A draw is a repeat with probability `repeat`: an earlier
    question, chosen with Zipf weights by rank. Otherwise it is the most
    popular question not asked yet. Once every question has been asked,
    every draw is a repeat, so a long run's measured rate (reported with
    each run) can be above `repeat`.
    """

    def __init__(
        self,
        questions: list[Mapping[str, Any]],
        alpha: float,
        repeat: float,
        rng: Any,
    ) -> None:
        """Creates the generator.

        Args:
            questions: The question set (at least one row).
            alpha: The Zipf exponent.
            repeat: The chance that a draw repeats an earlier question.
            rng: A `random.Random`.

        Raises:
            ValueError: For an empty set or a rate outside [0, 1].
        """
        if not questions or not 0 <= repeat <= 1:
            raise ValueError("need questions and a repeat rate in [0, 1]")
        self._ranked = list(questions)
        rng.shuffle(self._ranked)
        self._alpha = alpha
        self._repeat = repeat
        self._rng = rng
        self._asked = 0
        self.draws = 0
        self.repeats = 0

    def draw(self) -> Mapping[str, Any]:
        """The next question."""
        self.draws += 1
        fresh = self._asked < len(self._ranked)
        if self._asked and (not fresh or self._rng.random() < self._repeat):
            self.repeats += 1
            weights = [1 / (r + 1) ** self._alpha for r in range(self._asked)]
            (rank,) = self._rng.choices(range(self._asked), weights)
            return self._ranked[rank]
        self._asked += 1
        return self._ranked[self._asked - 1]

    def repeat_rate(self) -> float:
        """Share of draws so far that repeated an earlier question."""
        return self.repeats / self.draws if self.draws else 0.0
