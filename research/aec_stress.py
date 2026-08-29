"""Reference-aware AEC/barge-in stress test using real project speech.

This is deliberately not a one-tone echo demo.  It creates deterministic
rooms from the spoken cases12 questions and Kokoro answers, then evaluates:

* far-end-only false barge-ins (Jaeger hearing itself);
* double-talk detection recall and delay;
* echo attenuation before overlap; and
* near-end speech survival during overlap.

No model or GPU is used.  The room paths include delay, reflections, spectral
colouring, background noise, and mild loudspeaker non-linearity.

    python research/aec_stress.py
    python research/aec_stress.py --cases 43 --json /tmp/aec-stress.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from jaeger_agent_omni.runtime.aec import (
    EchoAwareBargeIn,
    SpeexEchoCanceller,
)

ROOT = Path(__file__).resolve().parent.parent
RATE = 16_000
BLOCK = 1_600                         # live callback cadence: 100 ms
SPEECH_RMS = 0.012


def _read(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if rate != RATE:
        from math import gcd
        divisor = gcd(rate, RATE)
        audio = resample_poly(audio, RATE // divisor, rate // divisor)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio / max(peak, 1e-6)


def _speech_stream(paths: list[Path], samples: int, offset: int) -> np.ndarray:
    """Build a deterministic continuous real-speech stream."""
    pieces: list[np.ndarray] = []
    cursor = offset
    while sum(map(len, pieces)) < samples:
        wav = _read(paths[cursor % len(paths)])
        pieces.extend((wav, np.zeros(round(0.08 * RATE), np.float32)))
        cursor += 1
    return np.concatenate(pieces)[:samples]


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def _at_rms(audio: np.ndarray, target: float) -> np.ndarray:
    return (audio * (target / max(_rms(audio), 1e-8))).astype(np.float32)


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    """Return a conservative percentile that includes small-corpus tails."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return float(ordered[min(rank - 1, len(ordered) - 1)])


def _trim_leading_silence(audio: np.ndarray) -> np.ndarray:
    """Make reported barge delay start at acoustic speech, not WAV padding."""
    frame = round(0.020 * RATE)
    threshold = max(0.004, 0.15 * _rms(audio))
    for start in range(0, len(audio), frame):
        if _rms(audio[start:start + frame]) >= threshold:
            return audio[start:]
    return audio


def _room_echo(far: np.ndarray, rng: np.random.Generator,
               nonlinear: bool) -> np.ndarray:
    """Delayed, coloured multi-path loudspeaker-to-microphone signal."""
    delay = int(rng.uniform(0.025, 0.180) * RATE)
    impulse = np.zeros(delay + int(0.260 * RATE), np.float32)
    impulse[delay] = rng.uniform(0.28, 0.70)
    for _ in range(5):
        tap = delay + int(rng.uniform(0.012, 0.240) * RATE)
        impulse[tap] += rng.uniform(-0.22, 0.30) * np.exp(
            -(tap - delay) / (0.15 * RATE))
    echo = np.convolve(far, impulse, mode="full")[:len(far)]
    # Small loudspeaker/enclosure colour.  Speex must adapt to this rather than
    # receiving an unrealistically sample-identical reference.
    colour = np.asarray([0.08, 0.18, 0.30, 0.25, 0.13, 0.06], np.float32)
    echo = np.convolve(echo, colour, mode="same")
    if nonlinear:
        drive = rng.uniform(1.2, 2.0)
        echo = np.tanh(echo * drive) / drive
    return np.asarray(echo, np.float32)


def _run_aec(mic: np.ndarray, far: np.ndarray,
             with_detector: bool = False,
             detector_options: dict | None = None,
             include_trace: bool = False,
             reference_mode: str = "wallclock") -> tuple[np.ndarray, dict]:
    aec = SpeexEchoCanceller(RATE, reference_mode=reference_mode)
    detector = EchoAwareBargeIn(
        RATE, SPEECH_RMS, **(detector_options or {}))
    aec.start_playback(far, RATE, started_at=0.0)
    cleaned: list[np.ndarray] = []
    old_first = None
    new_first = None
    trace = []
    for start in range(0, len(mic), BLOCK):
        block = aec.process(
            mic[start:start + BLOCK], captured_at=(start + BLOCK) / RATE)
        cleaned.append(block)
        if with_detector:
            clean_rms = _rms(block)
            old = (clean_rms > SPEECH_RMS * 1.1 and
                   (aec.block_playback_age > 1.0 or
                    aec.block_attenuation_db < 3.0))
            now = (start + len(block)) / RATE
            if old and old_first is None:
                old_first = now
            fired = detector.update(aec, clean_rms, len(block))
            if include_trace:
                trace.append({
                    "end_s": now,
                    "clean_rms": clean_rms,
                    "echo_floor": detector.echo_floor,
                    "raw_correlation": aec.raw_reference_correlation,
                    "residual_correlation":
                        aec.residual_reference_correlation,
                    "attenuation_db": aec.block_attenuation_db,
                    "reference_rms": aec.reference_rms,
                    "fired": fired,
                })
            if fired and new_first is None:
                new_first = now
    metrics = {
        "old_first_s": old_first,
        "new_first_s": new_first,
        "aggregate_attenuation_db": aec.attenuation_db,
        "trace": trace,
    }
    aec.close()
    return np.concatenate(cleaned), metrics


def evaluate(count: int = 43, seed: int = 20260825,
             detector_options: dict | None = None,
             include_traces: bool = False,
             reference_mode: str = "wallclock") -> dict:
    far_paths = sorted((ROOT / "demo_out").glob("q*.wav"))
    near_paths = sorted((ROOT / "research/cases12").glob("q*.wav"))
    if not far_paths or not near_paths:
        raise FileNotFoundError("requires demo_out/q*.wav and research/cases12/q*.wav")

    rng = np.random.default_rng(seed)
    rows = []
    duration = 8.0
    total = round(duration * RATE)
    for case in range(count):
        far = _speech_stream(far_paths, total, case)
        far = _at_rms(far, rng.uniform(0.075, 0.145))
        echo = _room_echo(far, rng, nonlinear=(case % 4 == 0))
        noise = rng.normal(0.0, rng.uniform(0.0003, 0.0020), total).astype(
            np.float32)

        onset = rng.uniform(3.2, 5.0)
        near_duration = rng.uniform(1.2, 2.2)
        begin = round(onset * RATE)
        end = min(total, begin + round(near_duration * RATE))
        near = np.zeros(total, np.float32)
        voice = _trim_leading_silence(
            _speech_stream(near_paths, end - begin, count + case))
        if len(voice) < end - begin:
            voice = np.pad(voice, (0, end - begin - len(voice)))
        voice = voice[:end - begin]
        near[begin:end] = _at_rms(voice, rng.uniform(0.025, 0.070))

        echo_clean, _ = _run_aec(
            echo + noise, far, reference_mode=reference_mode)
        mixed_clean, decisions = _run_aec(
            echo + near + noise, far, True, detector_options, include_traces,
            reference_mode=reference_mode)

        # Measure a full second immediately before the near talker enters,
        # after at least two seconds of adaptive-filter convergence.
        pre = slice(max(0, begin - RATE), begin)
        overlap = slice(begin, end)
        attenuation = 20 * np.log10(
            (_rms((echo + noise)[pre]) + 1e-9) /
            (_rms(echo_clean[pre]) + 1e-9))
        # Difference the matched echo-only run to estimate how much independent
        # near speech survives the canceller during double-talk.
        recovered_near = mixed_clean[overlap] - echo_clean[overlap]
        survival = _rms(recovered_near) / (_rms(near[overlap]) + 1e-9)

        old_first = decisions["old_first_s"]
        new_first = decisions["new_first_s"]
        rows.append({
            "case": case,
            "near_onset_s": onset,
            "echo_attenuation_db": float(attenuation),
            "near_survival": float(survival),
            "old_false_barge": old_first is not None and old_first < onset,
            "new_false_barge": new_first is not None and new_first < onset,
            "old_delay_ms": (None if old_first is None or old_first < onset else
                             1000 * (old_first - onset)),
            "new_delay_ms": (None if new_first is None or new_first < onset else
                             1000 * (new_first - onset)),
            **({"detector_trace": decisions["trace"]}
               if include_traces else {}),
        })

    def detected(key: str) -> list[float]:
        return [row[key] for row in rows if row[key] is not None]

    old_delays = detected("old_delay_ms")
    new_delays = detected("new_delay_ms")
    result = {
        "cases": count,
        "seed": seed,
        "reference_mode": reference_mode,
        "detector": detector_options or {},
        "callback_ms": 100,
        "echo_attenuation_median_db": float(np.median(
            [r["echo_attenuation_db"] for r in rows])),
        "near_survival_median_percent": float(100 * np.median(
            [r["near_survival"] for r in rows])),
        "old": {
            "false_barges": sum(r["old_false_barge"] for r in rows),
            "detected": len(old_delays),
            "recall_percent": 100 * len(old_delays) / count,
            "median_delay_ms": (float(np.median(old_delays))
                                if old_delays else None),
        },
        "reference_aware": {
            "false_barges": sum(r["new_false_barge"] for r in rows),
            "detected": len(new_delays),
            "recall_percent": 100 * len(new_delays) / count,
            "median_delay_ms": (float(np.median(new_delays))
                                if new_delays else None),
            "p95_delay_ms": _nearest_rank(new_delays, 0.95),
        },
        "rows": rows,
    }
    result["gates"] = {
        "echo_attenuation_at_least_6db":
            result["echo_attenuation_median_db"] >= 6.0,
        "near_survival_at_least_70_percent":
            result["near_survival_median_percent"] >= 70.0,
        "no_reference_aware_false_barges":
            result["reference_aware"]["false_barges"] == 0,
        "reference_aware_recall_at_least_90_percent":
            result["reference_aware"]["recall_percent"] >= 90.0,
        "reference_aware_p95_at_most_300ms": (
            result["reference_aware"]["p95_delay_ms"] is not None and
            result["reference_aware"]["p95_delay_ms"] <= 300.0),
    }
    result["passed"] = all(result["gates"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=43)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--echo-correlation", type=float, default=0.45)
    parser.add_argument("--residual-echo-correlation", type=float,
                        default=0.35)
    parser.add_argument("--residual-margin", type=float, default=1.1)
    parser.add_argument("--confirm-ms", type=float, default=80.0)
    parser.add_argument("--strong-margin", type=float, default=1.5,
                        help="loudness-over-floor ratio required by the "
                             "low-attenuation double-talk override")
    parser.add_argument("--strong-attenuation-db", type=float,
                        default=2.5,
                        help="block attenuation below which a loud "
                             "correlated block counts as double-talk")
    parser.add_argument("--strong-confirm-ms", type=float, default=250.0,
                        help="sustained low-attenuation loudness required "
                             "by the double-talk override")
    parser.add_argument("--reference-mode",
                        choices=("wallclock", "contiguous"),
                        default="wallclock",
                        help="wallclock is the promotion gate; contiguous "
                             "measures the experimental reply-stream mode "
                             "on the same corpus")
    parser.add_argument("--trace", action="store_true",
                        help="include per-callback detector telemetry")
    parser.add_argument("--json", type=Path,
                        default=Path("research/results/aec_stress.json"))
    args = parser.parse_args()
    if args.cases < 8:
        parser.error("--cases must be at least 8")

    detector_options = {
        "echo_correlation": args.echo_correlation,
        "residual_echo_correlation": args.residual_echo_correlation,
        "residual_margin": args.residual_margin,
        "confirm_ms": args.confirm_ms,
        "strong_margin": args.strong_margin,
        "strong_attenuation_db": args.strong_attenuation_db,
        "strong_confirm_ms": args.strong_confirm_ms,
    }
    result = evaluate(
        args.cases, args.seed, detector_options, include_traces=args.trace,
        reference_mode=args.reference_mode)
    old, new = result["old"], result["reference_aware"]
    def delay_text(value) -> str:
        return "n/a" if value is None else f"{value:.0f} ms"

    print(f"AEC stress: {result['cases']} real-speech synthetic rooms "
          f"({result['reference_mode']} reference)")
    print(f"  echo attenuation       {result['echo_attenuation_median_db']:.1f} dB median")
    print(f"  double-talk survival   {result['near_survival_median_percent']:.0f}% median")
    print(f"  old energy rule        {old['false_barges']}/{args.cases} false, "
          f"{old['detected']}/{args.cases} detected, "
          f"{delay_text(old['median_delay_ms'])} median")
    print(f"  reference-aware rule   {new['false_barges']}/{args.cases} false, "
          f"{new['detected']}/{args.cases} detected, "
          f"{delay_text(new['median_delay_ms'])} median, "
          f"{delay_text(new['p95_delay_ms'])} p95")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  artifact               {args.json}")

    # This is a regression gate, not a claim about the untested physical room.
    print(f"  gate                   "
          f"{'PASS' if result['passed'] else 'FAIL'}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
