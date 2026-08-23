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

from jaeger_agent_omni.runtime.stock_whisper import (
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


def load_source(cache: Path, split: str, max_seconds: float) -> list[dict]:
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
    return rows


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
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--whisper", choices=tuple(ENCODER_DIMS), required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    refuse_gpu_contention()
    print(f"loading frozen encoder {args.whisper}", flush=True)
    front_end = StockWhisperFrontEnd(args.whisper, args.device)
    total_written = 0
    started = time.time()
    for split in args.split:
        rows = load_source(args.source_cache, split, args.max_seconds)
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
