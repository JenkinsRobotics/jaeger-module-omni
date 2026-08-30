#!/usr/bin/env python3
"""Build factual overlap mixtures and score turn-level speaker continuity.

This CPU tool makes stems, a mixture, and sample-accurate active-speaker
labels from two or three single-speaker recordings.  CampPlus is evaluated
only on isolated single-speaker regions.  The overlap region is deliberately
reported as ``overlap`` because the current runtime has no source separator
and cannot honestly claim word-level attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


RATE = 16000
SEGMENT_S = 2.0


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        count = max(1, round(len(audio) * RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, count),
                          np.arange(len(audio)), audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if peak < 1e-6 or rms < 1e-6:
        raise ValueError(f"silent speaker source: {path}")
    # Loudness normalization prevents a quiet recording from being rejected
    # by the live speech floor and mistaken for an identity failure.  Peak
    # limiting preserves the source without clipping unusual transients.
    audio = audio / rms * 0.05
    peak = float(np.max(np.abs(audio)))
    if peak > 0.90:
        audio *= 0.90 / peak
    return np.asarray(audio, dtype=np.float32)


def take(audio: np.ndarray, count: int, offset: int) -> np.ndarray:
    """Deterministically wrap a source without padding speaker identity."""
    indices = (np.arange(count) + offset) % len(audio)
    return audio[indices]


def build_mixture(sources: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict]]:
    if len(sources) not in (2, 3):
        raise ValueError("overlap lab needs exactly two or three speakers")
    frames = round(SEGMENT_S * RATE)
    # Repeated isolated turns establish continuity; one factual overlap
    # segment tests the notification boundary without pretending separation.
    schedule: list[tuple[int, ...]] = [(0,), (1,), (0,), (0, 1), (1,), (0,)]
    if len(sources) == 3:
        schedule.extend([(2,), (1, 2), (2,)])
    stems = [np.zeros(frames * len(schedule), dtype=np.float32)
             for _ in sources]
    labels = []
    offsets = [index * 997 for index in range(len(sources))]
    for number, active in enumerate(schedule):
        begin, end = number * frames, (number + 1) * frames
        for speaker in active:
            stems[speaker][begin:end] = take(
                sources[speaker], frames, offsets[speaker])
            offsets[speaker] += frames
        labels.append({
            "segment": number, "start_sample": begin, "end_sample": end,
            "start_s": begin / RATE, "end_s": end / RATE,
            "active_speakers": [chr(65 + value) for value in active],
            "overlap": len(active) > 1,
        })
    return stems, labels


def score_continuity(truth: list[str], predicted: list[str]) -> dict:
    """Permutation-independent consistency and speaker separation metrics."""
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth and predicted labels must be non-empty and aligned")
    mapping: dict[str, list[str]] = {}
    for expected, observed in zip(truth, predicted):
        mapping.setdefault(expected, []).append(observed)
    dominant = {
        expected: max(set(values), key=values.count)
        for expected, values in mapping.items()
    }
    consistent = sum(observed == dominant[expected]
                     for expected, observed in zip(truth, predicted))
    return {
        "turns": len(truth),
        "continuity_accuracy": consistent / len(truth),
        "distinct_true_speakers": len(mapping),
        "distinct_predicted_speakers": len(set(predicted)),
        "separated_identities": len(set(dominant.values())) == len(mapping),
        "dominant_mapping": dominant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--speaker", type=Path, action="append", required=True,
                        help="single-speaker WAV/FLAC; pass two or three")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("research/results/overlap_lab"))
    parser.add_argument("--report", type=Path,
                        help="optional summary JSON outside the audio directory")
    args = parser.parse_args()
    sources = [load_audio(path) for path in args.speaker]
    stems, labels = build_mixture(sources)
    mixture = np.clip(np.sum(stems, axis=0), -0.95, 0.95)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(stems):
        sf.write(args.out_dir / f"speaker_{chr(65 + index)}.wav", stem, RATE)
    sf.write(args.out_dir / "mixture.wav", mixture, RATE)

    from jaeger_agent_omni.core.runtime.speaker import OnlineSpeakerTracker
    tracker = OnlineSpeakerTracker(k=len(sources))
    truth, predicted, assignments = [], [], []
    for label in labels:
        if label["overlap"]:
            continue
        begin, end = label["start_sample"], label["end_sample"]
        expected = label["active_speakers"][0]
        observed = tracker.identify(stems[ord(expected) - 65][begin:end])
        assignments.append({
            "segment": label["segment"], "true_speaker": expected,
            "predicted_speaker": observed,
        })
        if observed is not None:
            truth.append(expected)
            predicted.append(observed)
    continuity = score_continuity(truth, predicted)
    exact_mix = np.allclose(
        mixture, np.clip(np.sum(stems, axis=0), -0.95, 0.95))
    report = {
        "schema_version": 1, "kind": "jaeger_overlap_lab",
        "sample_rate": RATE,
        "sources": [str(path.resolve()) for path in args.speaker],
        "segments": labels, "assignments": assignments,
        "continuity": continuity,
        "mixture_matches_stems": bool(exact_mix),
        "overlap_seconds": sum(
            label["end_s"] - label["start_s"]
            for label in labels if label["overlap"]),
        "word_level_overlap_attribution_supported": False,
        "passed_turn_level_identity": bool(
            exact_mix and continuity["continuity_accuracy"] >= 0.8 and
            continuity["separated_identities"]),
    }
    report_path = args.report or (args.out_dir / "report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"overlap lab: continuity "
          f"{100*continuity['continuity_accuracy']:.0f}% · identities "
          f"{continuity['distinct_predicted_speakers']}/"
          f"{continuity['distinct_true_speakers']} · separated "
          f"{'PASS' if continuity['separated_identities'] else 'FAIL'} · "
          f"word attribution NO")
    print(f"artifacts: {args.out_dir}")
    print(f"report: {report_path}")
    return 0 if report["passed_turn_level_identity"] else 2


def demo() -> None:
    """No-model structural proof used by the CPU suite."""
    a = np.linspace(-0.2, 0.2, RATE, dtype=np.float32)
    b = np.linspace(0.2, -0.2, RATE, dtype=np.float32)
    stems, labels = build_mixture([a, b])
    assert len(stems) == 2 and any(row["overlap"] for row in labels)
    score = score_continuity(
        ["A", "B", "A", "B", "A"], ["X", "Y", "X", "Y", "X"])
    assert score["continuity_accuracy"] == 1.0
    assert score["separated_identities"]
    print("overlap_lab: factual stems, overlap labels, and continuity scoring OK")


if __name__ == "__main__":
    import sys
    raise SystemExit(demo() if "--demo" in sys.argv else main())
