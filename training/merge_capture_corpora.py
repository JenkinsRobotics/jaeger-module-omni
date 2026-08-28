#!/usr/bin/env python3
"""Merge capture corpora by content identity without copying audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.select_novel_capture import identity  # noqa: E402


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
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-precision-ms", type=float, default=2.0)
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="stop after crossing this duration; zero keeps all")
    args = parser.parse_args()
    if args.max_hours < 0:
        raise ValueError("max hours cannot be negative")
    seen_content, seen_ids = set(), set()
    rows = []
    seconds = duplicates = 0.0
    source_counts = {}
    for raw_root in args.corpus:
        if args.max_hours and seconds >= args.max_hours * 3600:
            break
        root = raw_root.expanduser().resolve()
        manifest = root / "turns.jsonl"
        for line in manifest.read_text().splitlines():
            if args.max_hours and seconds >= args.max_hours * 3600:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            key = identity(row, args.duration_precision_ms)
            if key in seen_content:
                duplicates += 1
                continue
            seen_content.add(key)
            item_id = str(row["turn_id"])
            if item_id in seen_ids:
                raise ValueError(f"turn ID collision across corpora: {item_id}")
            seen_ids.add(item_id)
            audio = Path(str(row["audio_path"])).expanduser()
            row["audio_path"] = str(
                audio.resolve() if audio.is_absolute() else
                (root / audio).resolve())
            seconds += float(row.get("duration_s", 0.0))
            source_counts[str(root)] = source_counts.get(str(root), 0) + 1
            rows.append(row)
    output = args.output.expanduser().resolve()
    atomic_jsonl(output / "turns.jsonl", rows)
    digest = hashlib.sha256((output / "turns.jsonl").read_bytes()).hexdigest()
    print(json.dumps({
        "rows": len(rows), "hours": seconds / 3600,
        "content_duplicates_removed": int(duplicates),
        "input_rows_selected": source_counts, "manifest_sha256": digest,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
