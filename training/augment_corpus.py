#!/usr/bin/env python3
"""Build deterministic room/noise variants from human-labelled audio.

The output uses Jaeger's capture corpus format and lives outside the repo.  It
contains waveforms plus provenance only; stock-Whisper feature caching remains
a separate, explicitly GPU-guarded step.  Augmentation never changes the human
transcript and never touches validation data unless it is deliberately passed
as a source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

SAMPLE_RATE = 16_000


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _read_audio(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim > 1:
        values = values.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        values = resample_poly(
            values, SAMPLE_RATE // divisor, int(rate) // divisor)
    return np.asarray(values, dtype=np.float32)


def _audio_path(root: Path, row: dict) -> Path:
    for field in ("audio", "source_pcm", "audio_path"):
        if row.get(field):
            candidate = _resolve(root, str(row[field])).resolve()
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"{row.get('id', '<unknown>')}: no readable audio/source_pcm path")


def _transcript(row: dict) -> str:
    return str(row.get("transcript") or row.get("text") or "").strip()


def _seed(seed: int, item_id: str, copy: int) -> int:
    digest = hashlib.sha256(f"{seed}:{item_id}:{copy}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64) + 1e-12))


def _fit_noise(noise: np.ndarray, length: int, rng) -> np.ndarray:
    if not len(noise):
        return np.zeros(length, dtype=np.float32)
    if len(noise) < length:
        repeats = math.ceil(length / len(noise))
        noise = np.tile(noise, repeats)
    start = int(rng.integers(max(1, len(noise) - length + 1)))
    return noise[start:start + length]


def augment(speech: np.ndarray, rir_path: Path | None,
            noise_path: Path | None, snr_db: float, rng=None) -> np.ndarray:
    """Apply energy-normalized RIR and additive noise without time warping."""
    result = np.asarray(speech, dtype=np.float32)
    if rir_path is not None:
        rir = _read_audio(rir_path)
        if len(rir):
            peak = int(np.argmax(np.abs(rir)))
            rir = rir[peak:]
            rir /= np.sqrt(np.sum(np.square(rir), dtype=np.float64) + 1e-12)
            result = fftconvolve(result, rir, mode="full")[:len(result)]
            result = np.asarray(result, dtype=np.float32)
    if noise_path is not None:
        noise = _read_audio(noise_path)
        rng = rng or np.random.default_rng(0)
        segment = _fit_noise(noise, len(result), rng)
        scale = _rms(result) / (10 ** (snr_db / 20.0) * _rms(segment))
        result = result + segment * scale
    peak = float(np.max(np.abs(result), initial=0.0))
    if peak > 0.99:
        result = result * (0.99 / peak)
    return np.asarray(result, dtype=np.float32)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
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


def _rows(cache: Path, split: str) -> list[tuple[Path, dict]]:
    root = cache.expanduser().resolve() / split
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    result = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _transcript(row):
            result.append((root, row))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", nargs=2, action="append", required=True,
                        metavar=("CACHE", "SPLIT"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--musan", type=Path, required=True)
    parser.add_argument("--rirs", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--rir-probability", type=float, default=0.8)
    parser.add_argument("--noise-probability", type=float, default=0.8)
    parser.add_argument("--snr-min", type=float, default=8.0)
    parser.add_argument("--snr-max", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.copies < 1:
        raise ValueError("--copies must be positive")
    for value, name in ((args.rir_probability, "rir"),
                        (args.noise_probability, "noise")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name}-probability must be in [0, 1]")
    if args.snr_min > args.snr_max:
        raise ValueError("--snr-min cannot exceed --snr-max")

    rirs = sorted(args.rirs.expanduser().glob("simulated_rirs/**/*.wav"))
    noises = sorted(path for path in args.musan.expanduser().glob("**/*.wav")
                    if "/noise/" in str(path) or "/music/" in str(path)
                    or "/speech/" in str(path))
    if not rirs or not noises:
        raise RuntimeError(
            f"augmentation assets missing: {len(rirs)} RIRs, {len(noises)} noises")

    sources = []
    seen = set()
    for cache, split in args.source:
        for root, row in _rows(Path(cache), split):
            item_id = str(row["id"])
            if item_id in seen:
                raise ValueError(f"duplicate source ID across inputs: {item_id}")
            seen.add(item_id)
            sources.append((root, row))
    sources.sort(key=lambda item: hashlib.sha256(
        f"{args.seed}:{item[1]['id']}".encode()).digest())
    if args.max_examples:
        sources = sources[:args.max_examples]
    print(f"source rows={len(sources):,}; RIRs={len(rirs):,}; "
          f"noise/music/speech files={len(noises):,}")
    if args.dry_run:
        print(f"dry-run: would write {len(sources) * args.copies:,} variants "
              f"under {args.output.expanduser()}")
        return 0

    output = args.output.expanduser().resolve()
    audio_root = output / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (root, row) in enumerate(sources, 1):
        speech_path = _audio_path(root, row)
        speech = _read_audio(speech_path)
        for copy in range(args.copies):
            rng = np.random.default_rng(_seed(args.seed, str(row["id"]), copy))
            rir = rirs[int(rng.integers(len(rirs)))] \
                if rng.random() < args.rir_probability else None
            noise = noises[int(rng.integers(len(noises)))] \
                if rng.random() < args.noise_probability else None
            snr = float(rng.uniform(args.snr_min, args.snr_max))
            values = augment(speech, rir, noise, snr, rng)
            aug_id = f"aug:v1:{row['id']}:{copy}"
            name = hashlib.sha256(aug_id.encode()).hexdigest()[:24] + ".wav"
            destination = audio_root / name
            sf.write(destination, values, SAMPLE_RATE, subtype="FLOAT")
            license_name = str(row.get("license") or "CC-BY-4.0")
            records.append({
                "turn_id": aug_id,
                "session_id": "jaeger-room-augmentation-v1",
                "audio_path": str(destination.relative_to(output)),
                "duration_s": len(values) / SAMPLE_RATE,
                "transcript": _transcript(row),
                "response": "",
                "confidence": 1.0,
                "meta": {
                    "source": "augmented:" + str(row.get("source") or "unknown"),
                    "clean_id": str(row["id"]),
                    "speaker": str(row.get("speaker") or ""),
                    "license": license_name,
                    "rir": str(rir or ""),
                    "noise": str(noise or ""),
                    "snr_db": round(snr, 3),
                    "augmentation": "rir+musan-v1",
                },
            })
        if index == 1 or index % 250 == 0:
            print(f"augmented {index:,}/{len(sources):,} clean rows", flush=True)
    _atomic_jsonl(output / "turns.jsonl", records)
    summary = {
        "source_rows": len(sources), "copies": args.copies,
        "output_rows": len(records), "seed": args.seed,
        "snr_db": [args.snr_min, args.snr_max],
        "rir_probability": args.rir_probability,
        "noise_probability": args.noise_probability,
        "hours": sum(row["duration_s"] for row in records) / 3600,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
