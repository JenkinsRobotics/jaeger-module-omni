#!/usr/bin/env python3
"""Compare checkpoint-pinned latency breakdowns without mixing harnesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = (
    "encoder_ms", "adapter_and_audio_prefill_ms", "prompt_close_ms",
    "first_token_ms", "decode_after_first_ms", "turn_end_to_reply_ms")


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("kind") != "adapter_latency_breakdown":
        raise ValueError(f"{path}: not an adapter latency breakdown")
    missing = [stage for stage in STAGES
               if stage not in payload.get("median", {})]
    if missing:
        raise ValueError(f"{path}: missing median stages {missing}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.reports) < 2:
        parser.error("provide at least two reports")

    reports = [(path, load(path)) for path in args.reports]
    baseline_path, baseline = reports[0]
    baseline_rows = {
        (row["case"], row["repeat"]): row for row in baseline["rows"]}
    comparisons = []
    for path, candidate in reports[1:]:
        candidate_rows = {
            (row["case"], row["repeat"]): row
            for row in candidate["rows"]}
        if candidate_rows.keys() != baseline_rows.keys():
            raise ValueError(f"{path}: cases/repeats differ from {baseline_path}")
        same_checkpoint = candidate["checkpoint"] == baseline["checkpoint"]
        replies_identical = all(
            candidate_rows[key]["reply"] == baseline_rows[key]["reply"]
            for key in baseline_rows)
        stages = {}
        for stage in STAGES:
            base = float(baseline["median"][stage])
            value = float(candidate["median"][stage])
            stages[stage] = {
                "baseline_ms": base,
                "candidate_ms": value,
                "delta_ms": value - base,
                "change_percent": (
                    100.0 * (value - base) / base if base else None),
            }
        comparisons.append({
            "baseline": str(baseline_path),
            "candidate": str(path),
            "baseline_model": baseline["model_id"],
            "candidate_model": candidate["model_id"],
            "same_checkpoint": same_checkpoint,
            "replies_identical": replies_identical,
            "semantic_equivalence_required": same_checkpoint,
            "semantic_equivalence_pass": (
                replies_identical if same_checkpoint else None),
            "stages": stages,
        })

    result = {
        "schema_version": 1,
        "kind": "adapter_latency_comparison",
        "comparisons": comparisons,
    }
    for comparison in comparisons:
        total = comparison["stages"]["turn_end_to_reply_ms"]
        print(f"{comparison['candidate_model']}: "
              f"{total['candidate_ms']:.1f} ms, "
              f"{total['delta_ms']:+.1f} ms "
              f"({total['change_percent']:+.1f}%)")
        if comparison["same_checkpoint"]:
            print("  identical replies: "
                  f"{'yes' if comparison['replies_identical'] else 'NO'}")
        for stage in STAGES[:-1]:
            row = comparison["stages"][stage]
            print(f"  {stage:<32} {row['candidate_ms']:7.1f} ms  "
                  f"{row['delta_ms']:+7.1f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
