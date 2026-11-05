"""W1's question draws: Zipf popularity with a set repeat rate."""

import random
from collections import Counter

import pytest
from eval.locust.questions import ZipfQuestions

SET = [{"id": f"q{i}", "question": f"question {i}"} for i in range(200)]


def test_repeat_rate_is_the_dial_while_fresh_questions_last() -> None:
    gen = ZipfQuestions(SET, 1.1, 0.6, random.Random(1))
    for _ in range(400):
        gen.draw()
    assert gen.repeat_rate() == pytest.approx(0.6, abs=0.06)


def test_popular_questions_repeat_most() -> None:
    gen = ZipfQuestions(SET, 1.1, 0.6, random.Random(2))
    counts = Counter(gen.draw()["id"] for _ in range(400))
    ((top, n),) = counts.most_common(1)
    assert n > 20  # noqa: PLR2004 - the head is asked again and again
    assert sum(1 for c in counts.values() if c == 1) > 50  # a long tail


def test_once_all_are_asked_every_draw_repeats() -> None:
    gen = ZipfQuestions(SET[:3], 1.1, 0.0, random.Random(3))
    first = {gen.draw()["id"] for _ in range(3)}
    assert first == {"q0", "q1", "q2"}
    for _ in range(10):
        gen.draw()
    assert gen.repeats == 10  # noqa: PLR2004
    assert gen.repeat_rate() == pytest.approx(10 / 13)


def test_bad_arguments_are_refused() -> None:
    assert ZipfQuestions(SET, 1.1, 0.5, random.Random()).repeat_rate() == 0
    with pytest.raises(ValueError):
        ZipfQuestions([], 1.1, 0.5, random.Random())
    with pytest.raises(ValueError):
        ZipfQuestions(SET, 1.1, 1.5, random.Random())
