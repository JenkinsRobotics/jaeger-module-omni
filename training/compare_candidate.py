#!/usr/bin/env python3
"""Refuse promotion when a candidate regresses any independent quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WER_GATES = (
    "devclean64", "mindsvox591", "clean360val128", "novelvox128", "ami64",
)


def load(root: Path, label: str, gate: str) -> dict:
    path = root / f"{label}-{gate}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--root", type=Path,
                        default=Path("research/candidate_evals"))
    parser.add_argument("--include-lockbox", action="store_true")
    parser.add_argument("--skip-cases12", action="store_true",
                        help="select only from corpus gates")
    args = parser.parse_args()

    gates = list(WER_GATES)
    if args.include_lockbox:
        gates.append("testclean2603")
    regressions = []
    improvements = []
    print("gate                  baseline       candidate      delta")
    for gate in gates:
        before = load(args.root, args.baseline, gate)
        after = load(args.root, args.candidate, gate)
        if int(before["reference_words"]) != int(after["reference_words"]):
            raise ValueError(f"{gate}: reference-word population changed")
        old, new = int(before["edits"]), int(after["edits"])
        delta = new - old
        print(f"{gate:<21} {old:5d} edits   {new:5d} edits   {delta:+5d}")
        if delta > 0:
            regressions.append(f"{gate} {delta:+d} edits")
        elif delta < 0:
            improvements.append(f"{gate} {delta:+d} edits")

    if not args.skip_cases12:
        before = load(args.root, args.baseline, "cases12")
        after = load(args.root, args.candidate, "cases12")
        if int(before["n"]) != int(after["n"]):
            raise ValueError("cases12 population changed")
        old, new = int(before["correct"]), int(after["correct"])
        print(f"{'cases12':<21} {old:5d} correct {new:5d} correct "
              f"{new-old:+5d}")
        if new < old:
            regressions.append(f"cases12 {new-old:+d} answers")
        elif new > old:
            improvements.append(f"cases12 {new-old:+d} answers")

    if regressions:
        print("REJECT: " + "; ".join(regressions))
        return 1
    if not improvements:
        print("REJECT: no independent quality gate improved")
        return 1
    print("QUALITY GATES PASS: no regression; improved "
          + "; ".join(improvements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
