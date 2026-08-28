#!/usr/bin/env python3
"""Select non-overlapping capture rows without copying their audio files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def fields(row: dict) -> tuple[str, float]:
    text = str(row.get("transcript") or row.get("text") or "").strip()
    seconds = float(row.get("duration_s", row.get("seconds", 0.0)))
    return text, seconds


def identity(row: dict, precision_ms: float) -> tuple[str, int]:
    text, seconds = fields(row)
    quantum = max(1, round(precision_ms))
    return normalized(text), round(seconds * 1000 / quantum) * quantum


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=f".{path.name}.",
                suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append",
                        default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-precision-ms", type=float, default=2.0)
    args = parser.parse_args()
    corpus = args.corpus.expanduser().resolve()
    manifest = corpus / "turns.jsonl"
    source_rows = lines(manifest)
    excluded = set()
    for path in args.exclude_manifest:
        path = path.expanduser().resolve()
        for row in lines(path):
            key = identity(row, args.duration_precision_ms)
            if key[0]:
                excluded.add(key)

    selected = []
    duplicate_keys = set()
    seconds = 0.0
    for row in source_rows:
        key = identity(row, args.duration_precision_ms)
        if key in excluded:
            duplicate_keys.add(key)
            continue
        excluded.add(key)
        audio = Path(str(row["audio_path"])).expanduser()
        row = dict(row)
        row["audio_path"] = str(
            audio.resolve() if audio.is_absolute() else
            (corpus / audio).resolve())
        selected.append(row)
        seconds += fields(row)[1]

    output = args.output.expanduser().resolve()
    atomic_jsonl(output / "turns.jsonl", selected)
    digest = hashlib.sha256((output / "turns.jsonl").read_bytes()).hexdigest()
    print(json.dumps({
        "source_rows": len(source_rows), "excluded_rows": len(source_rows) - len(selected),
        "distinct_excluded_keys": len(duplicate_keys), "selected_rows": len(selected),
        "selected_hours": seconds / 3600, "manifest_sha256": digest,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
