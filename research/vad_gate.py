#!/usr/bin/env python3
"""Corpus-level CPU gate for neural speech admission and endpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from jaeger_agent_omni.core.runtime.endpoint import Endpoint
from jaeger_agent_omni.core.runtime.neural_vad import configure_endpoint


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


def evaluate_audio(audio: np.ndarray, source: str, expected: str,
                   silence_ms: float) -> dict:
    endpoint = Endpoint(silence_ms=silence_ms)
    configure_endpoint(endpoint, mode="neural", energy_threshold=0.012)
    audio = np.concatenate((audio, np.zeros(RATE, np.float32)))
    events = []
    for offset in range(0, len(audio) - FRAME + 1, FRAME):
        event = endpoint.push(audio[offset:offset + FRAME])
        if event:
            events.append({"event": event, "at_ms": offset / RATE * 1000})
    opened = any(item["event"] == "speech_start" for item in events)
    ended = any(item["event"] == "speech_end" for item in events)
    return {
        "audio": source, "expected": expected,
        "opened": opened, "ended": ended,
        "correct": opened == (expected == "speech"),
        "probability_max": endpoint.probability_max,
        "probability_mean": (
            endpoint.probability_sum / max(1, endpoint.probability_frames)),
        "events": events,
    }


def evaluate_file(path: Path, expected: str, silence_ms: float) -> dict:
    return evaluate_audio(
        load_audio(path), str(path.resolve()), expected, silence_ms)


def synthetic_nonspeech() -> list[tuple[str, np.ndarray]]:
    """Small deterministic sanity set; not a substitute for a room corpus."""
    count = RATE * 2
    timeline = np.arange(count, dtype=np.float32) / RATE
    rng = np.random.default_rng(9137)
    clicks = np.zeros(count, np.float32)
    clicks[::RATE // 4] = 0.7
    return [
        ("synthetic:silence", np.zeros(count, np.float32)),
        ("synthetic:60hz-hum", (0.04 * np.sin(2*np.pi*60*timeline)).astype(np.float32)),
        ("synthetic:1khz-tone", (0.05 * np.sin(2*np.pi*1000*timeline)).astype(np.float32)),
        ("synthetic:low-white-noise", rng.normal(0, 0.008, count).astype(np.float32)),
        ("synthetic:impulse-clicks", clicks),
    ]


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
    if not negative:
        results += [
            evaluate_audio(audio, source, "nonspeech", args.silence_ms)
            for source, audio in synthetic_nonspeech()
        ]
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
            "negative_fixture": (
                "provided corpus" if negative else
                "five deterministic sanity signals; physical room still required"),
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
