#!/usr/bin/env python3
"""Rescore free-running reports with Whisper's standard English normalizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.evaluate import wer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True,
                        help="repeatable NAME=RESULT.json")
    parser.add_argument("--whisper", default="openai/whisper-medium")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import WhisperTokenizer
    tokenizer = WhisperTokenizer.from_pretrained(args.whisper)
    datasets = {}
    total_edits = total_words = 0
    for spec in args.input:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        report = json.loads(path.read_text())
        edits = words = 0
        for row in report["rows"]:
            reference = tokenizer.normalize(row["reference"]).split()
            hypothesis = tokenizer.normalize(row["hypothesis"]).split()
            edits += int(round(wer(reference, hypothesis) * len(reference)))
            words += len(reference)
        score = edits / max(1, words)
        datasets[name] = {
            "report": str(path), "normalized_wer": score,
            "normalized_edits": edits, "normalized_reference_words": words,
            "passes_under_5pct": score < 0.05,
        }
        total_edits += edits
        total_words += words
    aggregate = total_edits / max(1, total_words)
    result = {
        "normalizer": "transformers.WhisperTokenizer.normalize",
        "whisper": args.whisper, "datasets": datasets,
        "aggregate": {
            "normalized_wer": aggregate, "normalized_edits": total_edits,
            "normalized_reference_words": total_words,
        },
        "gate_pass": aggregate < 0.05 and all(
            row["passes_under_5pct"] for row in datasets.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
