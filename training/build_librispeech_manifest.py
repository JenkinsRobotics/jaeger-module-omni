#!/usr/bin/env python3
"""Build deterministic context-training manifests from local LibriSpeech.

No audio is copied into the repository. The resulting external cache points
at the original corpus files, retains human transcripts and speaker ids, and
uses a speaker-level hash split so one reader cannot leak into both training
and validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import soundfile as sf


def bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


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


def existing_ids(manifests: list[Path]) -> set[str]:
    result = set()
    for manifest in manifests:
        manifest = manifest.expanduser().resolve()
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        for line in manifest.read_text().splitlines():
            if line.strip():
                result.add(str(json.loads(line)["id"]))
    return result


def source_rows(root: Path, max_seconds: float) -> list[dict]:
    rows = []
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            item_id, separator, text = line.partition(" ")
            if not separator or not text.strip():
                continue
            audio = transcript.parent / f"{item_id}.flac"
            if not audio.is_file():
                raise FileNotFoundError(audio)
            info = sf.info(audio)
            seconds = info.frames / info.samplerate
            if seconds <= 0 or (max_seconds and seconds > max_seconds):
                continue
            speaker = item_id.split("-", 1)[0]
            rows.append({
                "id": item_id,
                "audio": str(audio.resolve()),
                "seconds": seconds,
                "text": text.strip(),
                "speaker": speaker,
                "source": "local:train-other-500",
                "license": "CC-BY-4.0",
                "human_label": True,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True,
                        help="LibriSpeech train-other-500 directory")
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=12000)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--exclude-manifest", type=Path, action="append",
                        default=[])
    args = parser.parse_args()
    if not 0 < args.validation_percent < 100:
        raise ValueError("validation percent must be between 1 and 99")
    if args.max_examples <= 0:
        raise ValueError("max examples must be positive")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    excluded = existing_ids(args.exclude_manifest)
    rows = [row for row in source_rows(root, args.max_seconds)
            if row["id"] not in excluded]
    rows.sort(key=lambda row: hashlib.sha256(
        f"{args.seed}:{row['id']}".encode()).digest())
    selected = rows[:args.max_examples]
    partitions = {"train_other": [], "train_other_validation": []}
    for row in selected:
        split = ("train_other_validation"
                 if bucket(row["speaker"], args.seed)
                 < args.validation_percent else "train_other")
        partitions[split].append(row)

    output = args.output_cache.expanduser().resolve()
    for split, split_rows in partitions.items():
        atomic_jsonl(output / split / "manifest.jsonl", split_rows)
    train_speakers = {row["speaker"] for row in partitions["train_other"]}
    validation_speakers = {
        row["speaker"] for row in partitions["train_other_validation"]}
    if train_speakers & validation_speakers:
        raise AssertionError("speaker leakage across partitions")
    counts = Counter(row["speaker"] for row in selected)
    print(f"source usable: {len(rows):,}; selected: {len(selected):,}")
    print(f"train: {len(partitions['train_other']):,} clips / "
          f"{len(train_speakers)} speakers")
    print(f"validation: {len(partitions['train_other_validation']):,} "
          f"clips / {len(validation_speakers)} speakers")
    print(f"largest speaker contribution: {max(counts.values(), default=0)}")
    print(f"wrote manifests under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
