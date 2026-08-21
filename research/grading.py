"""One grader, shared. Answer-matching bugs have cost us real accuracy points
in both directions, so the rule lives in exactly one place.

Two normalisations, because no single one is right:

    "T-A-C."      needs punctuation DELETED  -> "tac"
    "twenty-five" needs punctuation as SPACE -> "twenty five"

Deleting breaks the second ("twentyfive"), spacing breaks the first ("t a c").
A hit under either spelling is a hit -- the model answered, and which
punctuation it chose is not what we are measuring.
"""

from __future__ import annotations

import re

# The model is not wrong for saying "Fall". Grading it wrong measures the
# grader, not the model.
SYNONYMS = {
    "autumn": ("autumn", "fall"),
    "zero degrees celsius": ("zero degrees celsius", "zero degrees", "0 degrees",
                             "32 degrees fahrenheit"),
}


def _forms(t: str) -> tuple[str, str]:
    t = re.sub(r"<[^>]*>", " ", t.lower())          # strip <turn|> and friends
    spaced = " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())
    dropped = re.sub(r"[^a-z0-9]+", "", t)
    return spaced, dropped


def hit(expected: str, reply: str) -> bool:
    r_spaced, r_dropped = _forms(reply)
    for alt in SYNONYMS.get(expected.lower(), (expected,)):
        a_spaced, a_dropped = _forms(alt)
        if a_spaced and a_spaced in r_spaced:
            return True
        if a_dropped and a_dropped in r_dropped:
            return True
    return False


def graded(reply: str, answers: list) -> bool:
    """Multi-answer form: 2 of 3+ required, else 1. Matches prior behaviour."""
    hits = sum(hit(a, reply) for a in answers)
    return hits >= (2 if len(answers) >= 3 else 1)


def demo() -> None:
    assert hit("twenty five", "Five times five is twenty-five.")   # was scored MISS
    assert hit("twenty four", "There are twenty-four hours in a day.")
    assert hit("autumn", "Fall comes after summer.")
    assert hit("tac", "It is spelled T-A-C.")                      # the old bug
    assert hit("paris", "The capital of France is Paris.")
    assert not hit("paris", "The capital of France is Lyon.")
    assert not hit("eight", "Spiders have six legs.")
    assert graded("Blue, like the sky.", ["blue"])
    assert not graded("I'm here to help!", ["paris"])
    print("grading demo ok")


if __name__ == "__main__":
    demo()
