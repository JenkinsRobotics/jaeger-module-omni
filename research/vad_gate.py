#!/usr/bin/env python3
"""Corpus-level CPU gate for neural speech admission and endpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from jaeger_agent_omni.runtime.endpoint import Endpoint
from jaeger_agent_omni.runtime.neural_vad import configure_endpoint


RATE = 16_000
FRAME = 480


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        count = max(1, round(len(audio) * RATE / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, count),
            np.arange(len(audio)), audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def evaluate_file(path: Path, expected: str, silence_ms: float) -> dict:
    endpoint = Endpoint(silence_ms=silence_ms)
    configure_endpoint(endpoint, mode="neural", energy_threshold=0.012)
    audio = np.concatenate((load_audio(path), np.zeros(RATE, np.float32)))
    events = []
    for offset in range(0, len(audio) - FRAME + 1, FRAME):
        event = endpoint.push(audio[offset:offset + FRAME])
        if event:
            events.append({"event": event, "at_ms": offset / RATE * 1000})
    opened = any(item["event"] == "speech_start" for item in events)
    ended = any(item["event"] == "speech_end" for item in events)
    return {
        "audio": str(path.resolve()), "expected": expected,
        "opened": opened, "ended": ended,
        "correct": opened == (expected == "speech"),
        "probability_max": endpoint.probability_max,
        "probability_mean": (
            endpoint.probability_sum / max(1, endpoint.probability_frames)),
        "events": events,
    }


def wavs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.suffix.lower() in {".wav", ".flac"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--nonspeech-dir", type=Path)
    parser.add_argument("--silence-ms", type=float, default=510.0)
    parser.add_argument("--out", type=Path,
                        default=Path("research/results/vad-gate.json"))
    args = parser.parse_args()
    positive = wavs(args.speech_dir)
    negative = wavs(args.nonspeech_dir) if args.nonspeech_dir else []
    if not positive:
        parser.error("--speech-dir contains no WAV/FLAC files")
    results = [evaluate_file(path, "speech", args.silence_ms)
               for path in positive]
    results += [evaluate_file(path, "nonspeech", args.silence_ms)
                for path in negative]
    speech = [row for row in results if row["expected"] == "speech"]
    nonspeech = [row for row in results if row["expected"] == "nonspeech"]
    report = {
        "kind": "jaeger_neural_vad_gate",
        "evaluation": "free_running_corpus_cpu",
        "configuration": {
            "model": "silero-vad", "frame_ms": 30,
            "open_threshold": 0.30, "close_threshold": 0.125,
            "open_evidence": "4 positive frames in 180ms",
            "silence_ms": args.silence_ms,
        },
        "summary": {
            "clips": len(results),
            "speech_opened": sum(row["opened"] for row in speech),
            "speech_total": len(speech),
            "speech_ended": sum(row["ended"] for row in speech),
            "false_opens": sum(row["opened"] for row in nonspeech),
            "nonspeech_total": len(nonspeech),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    print(f"speech opened {summary['speech_opened']}/"
          f"{summary['speech_total']} · ended {summary['speech_ended']}/"
          f"{summary['speech_total']} · "
          f"false opens {summary['false_opens']}/"
          f"{summary['nonspeech_total']} · {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
