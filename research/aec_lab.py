#!/usr/bin/env python3
"""CPU-only corpus lab for echo cancellation and double-talk detection.

The lab never loads Gemma or an audio encoder.  It generates multiple
speech-shaped far/near signals, passes the exact far-end reference through
room-delay/tap variants, and measures SpeexDSP on echo-only and double-talk
regions.  Results are corpus aggregates, not a hand-picked clip.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from jaeger_agent_omni.runtime.aec import EchoAwareBargeIn, SpeexEchoCanceller


RATE = 16000
SECONDS = 8
FRAME = 1600
NEAR_START_S = 6.0
NEAR_END_S = 7.0


def speech_shaped(seed: int, seconds: int = SECONDS) -> np.ndarray:
    """Deterministic band-limited, amplitude-modulated test excitation."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, RATE * seconds).astype(np.float32)
    x = np.convolve(x, np.ones(7, dtype=np.float32) / 7, mode="same")
    t = np.arange(x.size, dtype=np.float32) / RATE
    envelope = np.square(np.sin(2 * np.pi * (1.7 + seed * 0.03) * t))
    envelope = 0.12 + 0.88 * envelope
    x = x * envelope
    return np.asarray(x / (np.max(np.abs(x)) + 1e-9) * 0.16,
                      dtype=np.float32)


def room_echo(far: np.ndarray, delay_ms: float, gain: float,
              tail: float) -> np.ndarray:
    """Apply a known sparse room path while retaining the dry reference."""
    out = np.zeros_like(far)
    taps = ((delay_ms, gain), (delay_ms + 37, gain * tail),
            (delay_ms + 113, gain * tail * 0.55))
    for milliseconds, amplitude in taps:
        delay = round(milliseconds * RATE / 1000)
        if delay < far.size:
            out[delay:] += amplitude * far[:-delay]
    return out


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def run_case(case_id: int, delay_ms: float, gain: float,
             near_gain: float) -> dict:
    far = speech_shaped(100 + case_id)
    near_source = speech_shaped(1000 + case_id)
    near = np.zeros_like(far)
    start, end = round(NEAR_START_S * RATE), round(NEAR_END_S * RATE)
    near[start:end] = near_gain * near_source[start:end]
    echo = room_echo(far, delay_ms, gain, tail=0.32)
    mic = np.clip(echo + near, -0.95, 0.95).astype(np.float32)

    aec = SpeexEchoCanceller(RATE)
    detector = EchoAwareBargeIn(RATE, speech_rms=0.012)
    aec.start_playback(far, RATE, started_at=0.0)
    cleaned = []
    detections = []
    for offset in range(0, far.size, FRAME):
        block = aec.process(
            mic[offset:offset + FRAME],
            captured_at=(offset + min(FRAME, far.size - offset)) / RATE)
        cleaned.append(block)
        if detector.update(aec, _rms(block), len(block)):
            detections.append((offset + len(block)) / RATE)
    cleaned = np.concatenate(cleaned)

    echo_slice = slice(round(3.0 * RATE), round(5.0 * RATE))
    near_slice = slice(start, end)
    erle_db = 20 * math.log10(
        (_rms(mic[echo_slice]) + 1e-9) /
        (_rms(cleaned[echo_slice]) + 1e-9))
    target = near[near_slice]
    observed = cleaned[near_slice]
    near_projection = float(np.dot(observed, target) /
                            (np.dot(target, target) + 1e-12))
    projected = near_projection * target
    near_error = observed - projected
    near_snr_db = 10 * math.log10(
        (float(np.dot(projected, projected)) + 1e-12) /
        (float(np.dot(near_error, near_error)) + 1e-12))
    false_detections = [value for value in detections if value < NEAR_START_S]
    valid_detections = [value for value in detections if value >= NEAR_START_S]
    latency_ms = ((valid_detections[0] - NEAR_START_S) * 1000
                  if valid_detections else None)
    result = {
        "id": case_id, "delay_ms": delay_ms, "echo_gain": gain,
        "near_gain": near_gain, "erle_db": erle_db,
        "near_projection": near_projection, "near_snr_db": near_snr_db,
        "false_barge_events": len(false_detections),
        "barge_latency_ms": latency_ms,
    }
    aec.close()
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    """Conservative nearest-rank percentile.

    Rounding a zero-based index can hide the slowest sample in a small
    corpus (12 samples made a 300 ms event look like a 100 ms p95).  The
    nearest-rank definition deliberately includes that tail event.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[min(rank - 1, len(ordered) - 1)]


def run_corpus() -> dict:
    cases = []
    case_id = 0
    for delay_ms in (40.0, 85.0, 145.0):
        for gain in (0.40, 0.65):
            # Near-end speech must be audible over the loudspeaker echo.  A
            # much weaker talker is an SNR limitation, not a valid missed-
            # barge regression.  These span roughly equal to and louder than
            # the direct echo path.
            for near_gain in (0.65, 1.00):
                cases.append(run_case(case_id, delay_ms, gain, near_gain))
                case_id += 1
    erle = [case["erle_db"] for case in cases]
    projection = [case["near_projection"] for case in cases]
    latency = [case["barge_latency_ms"] for case in cases
               if case["barge_latency_ms"] is not None]
    summary = {
        "cases": len(cases),
        "median_erle_db": float(np.median(erle)),
        "minimum_erle_db": min(erle),
        "median_near_projection": float(np.median(projection)),
        "minimum_near_projection": min(projection),
        "false_barge_events": sum(
            case["false_barge_events"] for case in cases),
        "barge_detection_cases": len(latency),
        "barge_latency_p95_ms": percentile(latency, 0.95),
    }
    gates = {
        "median_erle_at_least_12db": summary["median_erle_db"] >= 12.0,
        "median_near_projection_at_least_0_65":
            summary["median_near_projection"] >= 0.65,
        "no_self_echo_false_barge": summary["false_barge_events"] == 0,
        "all_double_talk_detected": len(latency) == len(cases),
        "barge_p95_at_most_150ms": (
            summary["barge_latency_p95_ms"] is not None and
            summary["barge_latency_p95_ms"] <= 150.0),
    }
    return {
        "schema_version": 1, "kind": "jaeger_aec_corpus",
        "sample_rate": RATE, "summary": summary, "gates": gates,
        "passed": all(gates.values()), "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path,
                        default=Path("research/results/aec_lab.json"))
    args = parser.parse_args()
    result = run_corpus()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["summary"]
    print(f"AEC lab: {summary['cases']} cases · median ERLE "
          f"{summary['median_erle_db']:.1f} dB · near projection "
          f"{summary['median_near_projection']:.2f} · false barges "
          f"{summary['false_barge_events']} · p95 detection "
          f"{summary['barge_latency_p95_ms']} ms")
    print(f"report: {args.out}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
