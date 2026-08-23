#!/usr/bin/env python3
"""Redownload exact teacher-cache waveforms into one removable corpus root.

The selected stock-Whisper recipe distils pre-Gemma vectors from an existing
tuned-Whisper teacher cache.  Distillation therefore needs the *same utterance
IDs* on both sides.  This tool streams only licensed, human-labelled datasets,
matches their stable IDs against teacher manifests, and retains normalized
16 kHz PCM plus one append-safe provenance manifest::

    python training/prepare_paired_audio.py \
      --teacher-cache ~/models/gemma-audio-adapter/cache \
      --teacher-split train_minds14=train \
      --teacher-split validation_minds14=validation \
      --source minds14 \
      --out ~/models/jaeger-training-audio/stock-round-20260823

Deleting ``--out`` removes the entire downloaded round.  No generated or ASR
text is admitted: transcripts come from the teacher cache's human labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Source:
    dataset: str
    configs: tuple[str, ...]
    dataset_split: str
    licence: str
    id_fields: tuple[str, ...]


SOURCES = {
    "minds14": Source(
        "PolyAI/minds14", ("en-US", "en-GB", "en-AU"), "train",
        "CC-BY-4.0", ("path",)),
    "voxpopuli": Source(
        "facebook/voxpopuli", ("en",), "train", "CC0",
        ("id", "audio_id", "path")),
}


def _safe_name(item_id: str) -> str:
    return hashlib.sha1(item_id.encode()).hexdigest() + ".wav"


def _read_audio(audio: dict) -> np.ndarray:
    import soundfile as sf

    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    waveform, rate = sf.read(source, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform, SAMPLE_RATE // divisor, int(rate) // divisor)
    return np.asarray(waveform, dtype=np.float32)


def _split_specs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected TEACHER_SPLIT=train|validation: {value!r}")
        split, partition = value.rsplit("=", 1)
        if not split or partition not in ("train", "validation"):
            raise ValueError(f"invalid teacher split mapping: {value!r}")
        result[split] = partition
    return result


def _teacher_targets(cache: Path, split_specs: dict[str, str]) -> dict[str, dict]:
    targets = {}
    for split, partition in split_specs.items():
        manifest = cache / split / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row["id"])
            previous = targets.get(item_id)
            if previous and previous["split"] != partition:
                raise ValueError(f"teacher ID occurs in both partitions: {item_id}")
            targets[item_id] = {
                "id": item_id,
                "split": partition,
                "transcript": str(row.get("text") or "").strip(),
                "teacher_split": split,
                "source": str(row.get("source") or ""),
            }
    return targets


def _candidate_ids(source_key: str, config: str, row: dict,
                   fields: tuple[str, ...]) -> list[str]:
    raw = [str(row[field]) for field in fields if row.get(field)]
    if source_key == "minds14":
        return [f"minds14:{config}:{value}" for value in raw]
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument(
        "--teacher-split", action="append", required=True,
        metavar="NAME=train|validation")
    parser.add_argument("--source", action="append", choices=tuple(SOURCES),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    from datasets import Audio, load_dataset
    import soundfile as sf

    split_specs = _split_specs(args.teacher_split)
    targets = _teacher_targets(args.teacher_cache.expanduser(), split_specs)
    args.out = args.out.expanduser().resolve()
    audio_root = args.out / "audio"
    manifest = args.out / "manifest.jsonl"
    audio_root.mkdir(parents=True, exist_ok=True)
    completed = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[str(row["id"])] = row

    wanted = set(targets) - set(completed)
    written = seen = 0
    print(f"targets={len(targets)} already={len(targets) - len(wanted)} "
          f"remaining={len(wanted)}", flush=True)
    with manifest.open("a", buffering=1) as output:
        for source_key in dict.fromkeys(args.source):
            source = SOURCES[source_key]
            for config in source.configs:
                if not wanted:
                    break
                print(f"streaming {source.dataset}/{config}:{source.dataset_split}",
                      flush=True)
                dataset = load_dataset(
                    source.dataset, config, split=source.dataset_split,
                    streaming=True).cast_column("audio", Audio(decode=False))
                for row in dataset:
                    seen += 1
                    matches = wanted.intersection(_candidate_ids(
                        source_key, config, row, source.id_fields))
                    if not matches:
                        continue
                    # Candidate fields should resolve to one teacher ID.  If a
                    # dataset aliases them, one PCM file can safely serve both.
                    waveform = _read_audio(row["audio"])
                    for item_id in sorted(matches):
                        target = targets[item_id]
                        relative = Path("audio") / _safe_name(item_id)
                        destination = args.out / relative
                        sf.write(destination, waveform, SAMPLE_RATE,
                                 subtype="FLOAT")
                        record = {
                            **target,
                            "audio_path": str(relative),
                            "seconds": len(waveform) / SAMPLE_RATE,
                            "dataset": source.dataset,
                            "dataset_config": config,
                            "dataset_split": source.dataset_split,
                            "license": source.licence,
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        wanted.remove(item_id)
                        written += 1
                        if written == 1 or written % args.log_every == 0:
                            print(f"saved {written}: {item_id} "
                                  f"({record['seconds']:.2f}s)", flush=True)
                        if args.max_examples and written >= args.max_examples:
                            break
                    if (not wanted or
                            (args.max_examples and written >= args.max_examples)):
                        break
                if args.max_examples and written >= args.max_examples:
                    break

    missing = sorted(wanted)
    summary = {
        "targets": len(targets), "written": written,
        "completed": len(targets) - len(missing), "missing": len(missing),
        "seen": seen, "root": str(args.out),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RESULT " + json.dumps(summary, sort_keys=True), flush=True)
    if missing:
        (args.out / "missing_ids.txt").write_text("\n".join(missing) + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
