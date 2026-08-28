#!/usr/bin/env python3
"""Bucket a human-labelled manifest by matched Whisper decoder quality.

The decoder report must have been produced from the same manifest by
``research/direct_whisper_wer.py``.  Human text remains the training target;
Whisper output is used only as a data-quality/curriculum gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.evaluate import wer


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--id-prefix", default="",
                        help="optional dataset-local ID prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    tokenizer = None
    decoded = report["datasets"][args.dataset]
    manifest = Path(decoded["manifest"])
    source_rows = {
        str(row["id"]): row for row in
        (json.loads(line) for line in manifest.read_text().splitlines() if line)
    }
    source_root = manifest.parent
    buckets = {"gold_under1": [], "trusted_under5": [], "hard_over5": []}
    missing = []
    for result in decoded["rows"]:
        if args.id_prefix and not str(result["id"]).startswith(args.id_prefix):
            continue
        row = source_rows.get(str(result["id"]))
        if row is None:
            missing.append(result["id"])
            continue
        if "normalized_wer" in result:
            score = float(result["normalized_wer"])
        else:
            from transformers import WhisperTokenizer
            tokenizer = tokenizer or WhisperTokenizer.from_pretrained(
                report["model"])
            reference = tokenizer.normalize(result["reference"]).split()
            hypothesis = tokenizer.normalize(result["hypothesis"]).split()
            score = wer(reference, hypothesis)
        enriched = dict(row)
        for field in ("audio", "feature"):
            if enriched.get(field):
                path = Path(str(enriched[field])).expanduser()
                if not path.is_absolute():
                    path = (source_root / path).resolve()
                enriched[field] = str(path)
        enriched["whisper_quality"] = {
            "model": report["model"], "normalized_wer": score,
            "decoder_hypothesis": result["hypothesis"],
        }
        if score < 0.01:
            buckets["gold_under1"].append(enriched)
        elif score < 0.05:
            buckets["trusted_under5"].append(enriched)
        else:
            buckets["hard_over5"].append(enriched)
    if missing:
        raise RuntimeError(f"decoder report has {len(missing)} unknown IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    summary = {
        "source_report": str(args.report.resolve()),
        "source_manifest": str(manifest.resolve()),
        "dataset": args.dataset, "model": report["model"],
        "id_prefix": args.id_prefix,
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "policy": {
            "gold_under1": "normalized per-clip WER < 1%",
            "trusted_under5": "1% <= normalized per-clip WER < 5%",
            "hard_over5": "audit or robustness curriculum; never pseudo-label",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
