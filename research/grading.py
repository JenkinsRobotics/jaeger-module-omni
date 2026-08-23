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

# "There are 24 hours in a day" is not a wrong answer for "twenty four".
# Numbers get spelled either way and the grader must not care which.
NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}


def _digits(t: str) -> str:
    """twenty four -> 24, twenty-five -> 25. Compound tens+units are joined so
    the spelled and numeric forms land on the same string."""
    words, out = t.split(), []
    for w in words:
        out.append(NUMBERS.get(w, w))
    joined, i = [], 0
    while i < len(out):
        cur = out[i]
        nxt = out[i + 1] if i + 1 < len(out) else None
        if (cur in {"20","30","40","50","60","70","80","90"} and nxt
                and nxt.isdigit() and len(nxt) == 1 and nxt != "0"):
            joined.append(str(int(cur) + int(nxt))); i += 2
        else:
            joined.append(cur); i += 1
    return " ".join(joined)


# The model is not wrong for saying "Fall". Grading it wrong measures the
# grader, not the model.
SYNONYMS = {
    "autumn": ("autumn", "fall"),
    "zero degrees celsius": ("zero degrees celsius", "zero degrees", "0 degrees",
                             "32 degrees fahrenheit"),
}


def _forms(t: str) -> tuple[str, str, str]:
    t = re.sub(r"<[^>]*>", " ", t.lower())          # strip <turn|> and friends
    spaced = " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())
    dropped = re.sub(r"[^a-z0-9]+", "", t)
    return spaced, dropped, _digits(spaced)


def hit(expected: str, reply: str) -> bool:
    r_spaced, r_dropped, r_num = _forms(reply)
    for alt in SYNONYMS.get(expected.lower(), (expected,)):
        a_spaced, a_dropped, a_num = _forms(alt)
        if a_spaced and a_spaced in r_spaced:
            return True
        if a_dropped and a_dropped in r_dropped:
            return True
        # Numeric forms must align to complete normalized tokens. A plain
        # substring check falsely graded expected "seven" as present in
        # "-0.137" because both normalize to strings containing "7".
        if a_num and f" {a_num} " in f" {r_num} ":  # 24 == twenty four
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
    assert hit("twenty four", "There are 24 hours in a day.")      # digit form
    assert hit("seven", "There are 7 days in a week.")
    assert not hit("seven", "The result is -0.137.")
    assert not hit("twenty four", "There are 23 hours in a day.")
    assert hit("paris", "The capital of France is Paris.")
    assert not hit("paris", "The capital of France is Lyon.")
    assert not hit("eight", "Spiders have six legs.")
    assert graded("Blue, like the sky.", ["blue"])
    assert not graded("I'm here to help!", ["paris"])
    print("grading demo ok")


if __name__ == "__main__":
    demo()
