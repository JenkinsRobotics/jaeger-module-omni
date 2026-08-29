#!/usr/bin/env python3
"""Build a source-file-disjoint speech/music/noise/silence corpus from MUSAN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


RATE = 16_000
CLASSES = ("silence", "speech", "music", "noise")


def stable(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16)


def audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.suffix.lower() in {".wav", ".flac"})


def read_chunk(path: Path, seconds: float) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        count = max(1, round(len(audio) * RATE / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, count),
            np.arange(len(audio)), audio).astype(np.float32)
    needed = round(seconds * RATE)
    if len(audio) < needed:
        audio = np.pad(audio, (0, needed - len(audio)))
    elif len(audio) > needed:
        offset = stable(str(path)) % (len(audio) - needed + 1)
        audio = audio[offset:offset + needed]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return np.asarray(audio, dtype=np.float32)


def partition(path: Path) -> str:
    fold = stable(str(path.resolve())) % 10
    return "test" if fold == 0 else "validation" if fold == 1 else "train"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--musan", type=Path,
                        default=Path.home() / "datasets/augment/musan")
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "models/jaeger-training-data/"
                        "audio-perception-scenes-v1")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--validation-per-class", type=int, default=32)
    parser.add_argument("--test-per-class", type=int, default=32)
    args = parser.parse_args()
    source = {
        "speech": audio_files(args.musan / "speech"),
        "music": audio_files(args.musan / "music"),
        "noise": audio_files(args.musan / "noise"),
    }
    limits = {"train": args.train_per_class,
              "validation": args.validation_per_class,
              "test": args.test_per_class}
    selected: dict[str, dict[str, list[Path | None]]] = {
        split: {name: [] for name in CLASSES} for split in limits}
    for name, paths in source.items():
        for path in sorted(paths, key=lambda item: stable(str(item))):
            split = partition(path)
            if len(selected[split][name]) < limits[split]:
                selected[split][name].append(path)
    for split, limit in limits.items():
        selected[split]["silence"] = [None] * limit
        short = {name: len(items) for name, items in selected[split].items()
                 if len(items) < limit}
        if short:
            raise RuntimeError(f"{split}: insufficient source files {short}")

    for split in limits:
        root = args.out / f"scenes_{split}"
        audio_root = root / "audio"
        audio_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for class_index, name in enumerate(CLASSES):
            for index, source_path in enumerate(selected[split][name]):
                item_id = f"{name}_{index:04d}"
                destination = audio_root / f"{item_id}.wav"
                audio = (np.zeros(round(args.seconds * RATE), np.float32)
                         if source_path is None else
                         read_chunk(source_path, args.seconds))
                sf.write(destination, audio, RATE)
                rows.append({
                    "id": item_id, "audio": f"audio/{destination.name}",
                    "seconds": args.seconds, "scene": name,
                    "scene_index": class_index,
                    "speech_activity": int(name == "speech"),
                    "source_file": (str(source_path.resolve())
                                    if source_path is not None else None),
                    "source": "MUSAN" if source_path is not None else "synthetic",
                    "license": "MUSAN corpus terms",
                })
        rows.sort(key=lambda row: row["id"])
        (root / "manifest.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
        print(f"{split}: {len(rows)} clips -> {root}")
    print("source files are disjoint by stable hash; silence is deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
