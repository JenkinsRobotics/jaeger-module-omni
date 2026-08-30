#!/usr/bin/env python3
"""Cache a stock Whisper encoder over an existing audio manifest.

The source cache is treated only as data: this command imports no Python code
from it. Output manifests retain audio provenance and human labels while
replacing the encoder feature path with the requested stock Whisper model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.core.runtime.stock_whisper import (
    ENCODER_DIMS,
    INPUT_BOUNDARY,
    StockWhisperFrontEnd,
)
from training.context_distill import read_audio, refuse_gpu_contention


def safe_name(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:20] + ".npy"


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_source(cache: Path, split: str, max_seconds: float,
                selected_ids: list[str] | None = None) -> list[dict]:
    root = cache.expanduser().resolve() / split
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("audio"):
            continue
        if max_seconds and float(row.get("seconds", 0.0)) > max_seconds:
            continue
        row["audio"] = str(resolve(root, row["audio"]).resolve())
        rows.append(row)
    if selected_ids is None:
        return rows
    by_id = {str(row["id"]): row for row in rows}
    missing = [item_id for item_id in selected_ids if item_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} selected IDs absent from {manifest}: {preview}")
    return [by_id[item_id] for item_id in selected_ids]


def read_id_file(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    ids = [line.strip() for line in path.expanduser().read_text().splitlines()
           if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in selection file: {path}")
    if not ids:
        raise ValueError(f"empty selection file: {path}")
    return ids


def load_corpus(corpus: Path, max_seconds: float) -> list[dict]:
    """Map this repo's capture format onto the context-cache schema."""
    corpus = corpus.expanduser().resolve()
    manifest = corpus / "turns.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        seconds = float(item.get("duration_s", 0.0))
        text = str(item.get("transcript") or "").strip()
        if not text or (max_seconds and seconds > max_seconds):
            continue
        meta = item.get("meta") or {}
        audio = resolve(corpus, item["audio_path"]).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        rows.append({
            "id": str(item["turn_id"]),
            "audio": str(audio),
            "seconds": seconds,
            "text": text,
            "speaker": str(meta.get("speaker") or ""),
            "source": str(meta.get("source") or item.get("session_id")
                          or "capture"),
            "license": str(meta.get("license") or "CC-BY-4.0"),
            "human_label": float(item.get("confidence", 0.0)) >= 1.0,
        })
    return rows


def corpus_partitions(rows: list[dict], name: str,
                      validation_percent: int) -> list[tuple[str, list[dict]]]:
    """Make speaker-disjoint train/validation partitions."""
    if not 0 <= validation_percent < 100:
        raise ValueError("validation percent must be between 0 and 99")
    if validation_percent == 0:
        return [(name, rows)]
    result = {name: [], f"{name}_validation": []}
    for row in rows:
        speaker = row.get("speaker") or row["id"]
        value = int(hashlib.sha256(str(speaker).encode()).hexdigest()[:8], 16)
        split = (f"{name}_validation" if value % 100 < validation_percent
                 else name)
        result[split].append(row)
    return list(result.items())


def completed(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    return {
        str(json.loads(line)["id"])
        for line in manifest.read_text().splitlines() if line.strip()
    }


def append_manifest(path: Path, payload: dict) -> None:
    """Append and fsync one completed feature record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_feature(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, values.astype(np.float16))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-cache", type=Path)
    source.add_argument("--source-corpus", type=Path,
                        help="capture-format directory with turns.jsonl")
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument(
        "--id-file", type=Path,
        help="newline-delimited source IDs to encode, preserving file order")
    parser.add_argument("--corpus-split", default="capture_train")
    parser.add_argument("--validation-percent", type=int, default=10,
                        help="speaker-level holdout; zero writes training only")
    parser.add_argument("--whisper", choices=tuple(ENCODER_DIMS), required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.id_file and not args.source_cache:
        parser.error("--id-file is only valid with --source-cache")
    selected_ids = read_id_file(args.id_file)

    refuse_gpu_contention()
    print(f"loading frozen encoder {args.whisper}", flush=True)
    front_end = StockWhisperFrontEnd(args.whisper, args.device)
    total_written = 0
    started = time.time()
    if args.source_cache:
        if not args.split:
            parser.error("--source-cache requires at least one --split")
        partitions = [
            (split, load_source(args.source_cache, split, args.max_seconds,
                                selected_ids))
            for split in args.split
        ]
    else:
        if args.split:
            parser.error("use --corpus-split, not --split, with source corpus")
        partitions = corpus_partitions(
            load_corpus(args.source_corpus, args.max_seconds),
            args.corpus_split, args.validation_percent)
    for split, rows in partitions:
        output_root = args.output_cache.expanduser().resolve() / split
        manifest = output_root / "manifest.jsonl"
        done = completed(manifest)
        written = 0
        print(f"  {split}: {len(rows)} source, {len(done)} complete", flush=True)
        for row in rows:
            item_id = str(row["id"])
            if item_id in done:
                continue
            waveform = read_audio(Path(row["audio"]))
            features = front_end(waveform)
            expected = ENCODER_DIMS[args.whisper]
            if features.ndim != 2 or features.shape[1] != expected:
                raise RuntimeError(
                    f"{item_id}: encoder shape {features.shape}, "
                    f"expected (*, {expected})")
            name = safe_name(item_id)
            save_feature(output_root / "features" / name, features)
            payload = dict(row)
            payload.update({
                "feature": f"features/{name}",
                "frames": int(features.shape[0]),
                "encoder": args.whisper,
                "input_dim": expected,
                "boundary": INPUT_BOUNDARY,
            })
            append_manifest(manifest, payload)
            written += 1
            total_written += 1
            if total_written == 1 or total_written % args.log_every == 0:
                elapsed = time.time() - started
                print(f"cached {total_written}: {split}/{item_id} "
                      f"{tuple(features.shape)} "
                      f"{elapsed / total_written:.2f}s/example", flush=True)
            if args.max_examples and total_written >= args.max_examples:
                break
        print(f"  {split}: wrote {written}", flush=True)
        if args.max_examples and total_written >= args.max_examples:
            break
    print(f"cached {total_written} new examples at {args.output_cache}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
