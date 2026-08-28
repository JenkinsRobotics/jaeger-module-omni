#!/usr/bin/env python3
"""Audit capture-format training audio and normalize declared provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import soundfile as sf


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--set-license",
                        help="fill missing license; reject conflicting values")
    parser.add_argument("--expected-source")
    parser.add_argument("--duration-tolerance-ms", type=float, default=2.0)
    args = parser.parse_args()
    corpus = args.corpus.expanduser().resolve()
    manifest = corpus / "turns.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    rows = []
    identifiers = set()
    sources, licenses, speakers = Counter(), Counter(), set()
    seconds = 0.0
    changed = 0
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row.get("turn_id") or "")
        if not item_id or item_id in identifiers:
            raise ValueError(f"line {number}: missing/duplicate turn_id {item_id!r}")
        identifiers.add(item_id)
        if not str(row.get("transcript") or "").strip():
            raise ValueError(f"{item_id}: empty human transcript")
        if float(row.get("confidence", 0.0)) < 1.0:
            raise ValueError(f"{item_id}: label is not marked human confidence=1")
        audio = corpus / str(row.get("audio_path") or "")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        info = sf.info(audio)
        actual = info.frames / info.samplerate
        declared = float(row.get("duration_s", 0.0))
        if abs(actual - declared) * 1000 > args.duration_tolerance_ms:
            raise ValueError(
                f"{item_id}: duration {declared:.6f} != audio {actual:.6f}")
        meta = row.setdefault("meta", {})
        source = str(meta.get("source") or row.get("session_id") or "")
        if args.expected_source and source != args.expected_source:
            raise ValueError(
                f"{item_id}: source {source!r} != {args.expected_source!r}")
        current_license = str(meta.get("license") or "")
        if args.set_license:
            if current_license and current_license != args.set_license:
                raise ValueError(
                    f"{item_id}: license {current_license!r} conflicts with "
                    f"{args.set_license!r}")
            if not current_license:
                meta["license"] = args.set_license
                current_license = args.set_license
                changed += 1
        sources[source] += 1
        licenses[current_license or "MISSING"] += 1
        if meta.get("speaker") not in (None, ""):
            speakers.add(str(meta["speaker"]))
        seconds += actual
        rows.append(row)

    if changed:
        atomic_jsonl(manifest, rows)
    report = {
        "corpus": str(corpus), "rows": len(rows), "hours": seconds / 3600,
        "speakers": len(speakers), "sources": dict(sources),
        "licenses": dict(licenses), "metadata_rows_updated": changed,
        "manifest_sha256": sha256(manifest),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
