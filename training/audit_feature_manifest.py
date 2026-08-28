#!/usr/bin/env python3
"""Audit cached speech features before an adapter-training comparison.

The audit is read-only. It distinguishes manifest rows from underlying clip
IDs, verifies every referenced feature, samples tensor contents for finite
values, and checks ID/speaker isolation against one or more evaluation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def evaluation_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    conditions = payload.get("conditions", {})
    for preferred in ("with_instruction", "empty_instruction"):
        rows = conditions.get(preferred, {}).get("per_clip")
        if rows:
            return rows
    for key in ("rows", "clips", "per_clip"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError(f"cannot find evaluation rows in {path}")


def speaker_of(row: dict) -> str:
    speaker = str(row.get("speaker") or "").strip()
    if speaker:
        return speaker
    clip_id = str(row.get("id") or "")
    return clip_id.split("-", 1)[0] if "-" in clip_id else ""


def component_of(row: dict) -> str:
    feature = Path(str(row.get("feature") or ""))
    return feature.parent.parent.name if len(feature.parents) >= 2 else "unknown"


def normalized_text(value: str) -> str:
    """Ignore presentation-only case/punctuation drift in duplicate labels."""
    return re.sub(r"[^a-z0-9']+", " ", value.lower()).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", default=[])
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--expected-width", type=int, default=1024)
    parser.add_argument("--finite-samples", type=int, default=512,
                        help="deterministic evenly spaced tensors to read")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    all_rows = load_jsonl(manifest)
    rows = [row for row in all_rows
            if row.get("text")
            and row.get("bucket", "train") == "train"
            and 0 < float(row.get("seconds", 0)) <= args.max_seconds]

    ids = [str(row.get("id") or "") for row in rows]
    speakers = {speaker_of(row) for row in rows if speaker_of(row)}
    paths = [Path(str(row.get("feature") or "")).expanduser() for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    duplicate_ids = {key: count for key, count in Counter(ids).items()
                     if key and count > 1}
    duplicate_features = sum(count - 1 for count in Counter(paths).values()
                             if count > 1)
    text_by_id: dict[str, set[str]] = defaultdict(set)
    raw_text_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        clip_id = str(row.get("id") or "")
        raw = str(row.get("text") or "")
        text_by_id[clip_id].add(normalized_text(raw))
        raw_text_by_id[clip_id].add(raw)
    conflicting_text_ids = sorted(
        key for key, values in text_by_id.items() if key and len(values) > 1)
    presentation_variant_ids = sum(
        len(raw_text_by_id[key]) > 1 and len(text_by_id[key]) == 1
        for key in raw_text_by_id)

    shape_errors = []
    valid_paths = [path for path in paths if path.is_file()]
    for path in valid_paths:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[1] != args.expected_width:
                shape_errors.append({"path": str(path),
                                     "shape": list(array.shape)})
        except Exception as exc:  # include corrupt headers in the report
            shape_errors.append({"path": str(path), "error": str(exc)})

    finite_errors = []
    if valid_paths and args.finite_samples:
        count = min(args.finite_samples, len(valid_paths))
        indices = np.linspace(0, len(valid_paths) - 1, count,
                              dtype=np.int64)
        for index in indices:
            path = valid_paths[int(index)]
            try:
                array = np.load(path, allow_pickle=False)
                if not np.isfinite(array).all():
                    finite_errors.append(str(path))
            except Exception as exc:
                finite_errors.append(f"{path}: {exc}")

    train_ids = set(ids)
    evaluations = []
    for value in args.eval:
        path = value.expanduser().resolve()
        eval_rows = evaluation_rows(path)
        eval_ids = {str(row.get("id") or "") for row in eval_rows}
        eval_speakers = {speaker_of(row) for row in eval_rows if speaker_of(row)}
        evaluations.append({
            "path": str(path),
            "rows": len(eval_rows),
            "id_overlap": sorted(train_ids & eval_ids),
            "speaker_overlap": sorted(speakers & eval_speakers),
            "speakers": sorted(eval_speakers),
        })

    report = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "manifest_rows": len(all_rows),
        "effective_training_rows": len(rows),
        "effective_hours": sum(float(row["seconds"]) for row in rows) / 3600,
        "unique_clip_ids": len(set(ids)),
        "duplicate_id_extra_rows": len(rows) - len(set(ids)),
        "duplicate_id_groups": len(duplicate_ids),
        "conflicting_text_ids": conflicting_text_ids,
        "presentation_only_text_variant_ids": presentation_variant_ids,
        "unique_speakers": len(speakers),
        "components": dict(Counter(component_of(row) for row in rows)),
        "sources": dict(Counter(str(row.get("source") or "unknown")
                                for row in rows)),
        "missing_features": missing,
        "duplicate_feature_paths": duplicate_features,
        "shape_errors": shape_errors,
        "finite_tensors_sampled": min(args.finite_samples, len(valid_paths)),
        "finite_errors": finite_errors,
        "evaluations": evaluations,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    hard_errors = (missing or shape_errors or finite_errors
                   or conflicting_text_ids
                   or any(item["id_overlap"] or item["speaker_overlap"]
                          for item in evaluations))
    return 1 if hard_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
