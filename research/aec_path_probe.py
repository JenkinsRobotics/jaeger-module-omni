#!/usr/bin/env python3
"""Physically measure the loudspeaker-to-microphone echo path.

Two probes, both against the SAME devices the live loop uses (default output,
default input):

``--probe chirp``
    Plays three short log chirps and cross-correlates the recording against
    the template.  Reports per-repetition delay, correlation strength, and
    the drift between repetitions.  This answers the root-cause question the
    0.17/0.51 dB runs could not: is the total output+acoustic+input delay
    inside Speex's ``filter_ms`` span at all?  Chirp correlation is robust to
    background room speech.

``--probe aec --reference-mode wallclock|contiguous``
    Plays ONE fixed Kokoro waveform (identical bytes, identical duration for
    every mode) through the exact live mechanics — reference registered
    immediately before ``sd.play``, mic blocks processed by
    ``SpeexEchoCanceller`` at 16 kHz — and reports attenuation, correlations,
    and the contiguous-mode telemetry.  Unlike the 2026-08-29 forced-playback
    runs, exposure is equal by construction.  Continuous room audio during
    the probe acts as constant double-talk: absolute attenuation is diluted
    identically in both modes, so only the A/B delta is meaningful.

    python research/aec_path_probe.py --probe chirp
    python research/aec_path_probe.py --probe aec --reference-mode wallclock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATE = 16_000
BLOCK = 1_600                     # 100 ms, the live callback cadence
FIXED_SENTENCE = (
    "This is the fixed Jaeger echo path measurement sentence, "
    "repeated so both reference modes hear identical playback. ")


def _chirp(rate: int, seconds: float = 1.0, f0: float = 300.0,
           f1: float = 4000.0, amplitude: float = 0.30) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    phase = 2 * np.pi * f0 * seconds / np.log(f1 / f0) * (
        np.exp(t / seconds * np.log(f1 / f0)) - 1)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    return (amplitude * np.sin(phase) * fade).astype(np.float32)


def probe_chirp(out_path: Path, filter_ms: int) -> dict:
    import sounddevice as sd

    rate = 48_000                              # native device rate
    template = _chirp(rate)
    gap = np.zeros(round(1.5 * rate), np.float32)
    starts_s = []
    playback = [np.zeros(round(0.5 * rate), np.float32)]
    cursor = 0.5
    for _ in range(3):
        starts_s.append(cursor)
        playback.extend((template, gap))
        cursor += len(template) / rate + len(gap) / rate
    signal = np.concatenate(playback)
    # playrec aligns input and output on one stream clock, so the measured
    # lag is output latency + acoustics + input latency — the span the
    # adaptive filter must cover (live adds wall-clock registration skew on
    # top of this).
    recorded = sd.playrec(signal[:, None], samplerate=rate, channels=1,
                          dtype="float32")
    sd.wait()
    recorded = recorded[:, 0]

    kernel = (template - template.mean())[::-1]
    detections = []
    for start in starts_s:
        begin = round(start * rate)
        window = recorded[begin:begin + round(1.5 * rate) + len(template)]
        corr = np.convolve(window - window.mean(), kernel, mode="valid")
        energy = np.sqrt(
            np.convolve(np.square(window), np.ones(len(template)),
                        mode="valid") * float(np.dot(template, template)))
        normalized = corr / np.maximum(energy, 1e-9)
        peak = int(np.argmax(np.abs(normalized)))
        detections.append({
            "chirp_at_s": start,
            "delay_ms": 1000.0 * peak / rate,
            "peak_correlation": float(np.abs(normalized[peak])),
        })
    delays = [d["delay_ms"] for d in detections]
    span_s = starts_s[-1] - starts_s[0]
    result = {
        "schema_version": 1,
        "kind": "aec_path_chirp_probe",
        "sample_rate": rate,
        "detections": detections,
        "median_delay_ms": float(np.median(delays)),
        "delay_spread_ms": float(max(delays) - min(delays)),
        "drift_ppm_between_chirps": (
            1000.0 * (delays[-1] - delays[0]) / span_s if span_s else 0.0),
        "speex_filter_ms": filter_ms,
        "delay_within_filter_span": bool(np.median(delays) < filter_ms),
        "weak_coupling": bool(
            np.median([d["peak_correlation"] for d in detections]) < 0.10),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def probe_aec(out_path: Path, reference_mode: str, filter_ms: int,
              seconds: float, output_latency_ms: float = 0.0) -> dict:
    import sounddevice as sd
    import soundfile as sf

    from jaeger_agent_omni.core.runtime.aec import SpeexEchoCanceller
    from jaeger_agent_omni.voice import speak_kokoro

    # One deterministic waveform, reused verbatim by every mode.
    fixed = Path("/tmp/_aec_probe_fixed.wav")
    if not fixed.exists():
        repeats = max(1, round(seconds / 4.5))
        speak_kokoro(FIXED_SENTENCE * repeats, fixed, "af_heart", 1.0)
    wav, sr = sf.read(fixed, dtype="float32")

    aec = SpeexEchoCanceller(RATE, filter_ms=filter_ms,
                             reference_mode=reference_mode,
                             output_latency_ms=output_latency_ms)
    raw_all, clean_all = [], []

    def on_block(indata, frames, t, status):
        # Process INSIDE the callback, exactly like live.py's on_block —
        # a consumer queue would add variable lag between capture and
        # process() and jitter the wallclock reference alignment.
        raw = indata[:, 0].copy()
        raw_all.append(raw)
        clean_all.append(aec.process(raw))

    with sd.InputStream(samplerate=RATE, blocksize=BLOCK, channels=1,
                        dtype="float32", callback=on_block):
        # Exactly the live ordering: register the reference, then play.
        if reference_mode == "contiguous":
            aec.begin_reply_playback()
            aec.append_playback(wav, sr)
        else:
            aec.start_playback(wav, sr)
        sd.play(wav, sr)
        deadline = time.monotonic() + len(wav) / sr + 1.5
        while time.monotonic() < deadline:
            time.sleep(0.1)
        sd.wait()

    raw_cat = np.concatenate(raw_all)
    clean_cat = np.concatenate(clean_all)

    # Echo-specific measurement: continuous room speech is uncorrelated with
    # the reference, so cross-correlation isolates the echo component that
    # aggregate raw/clean energy cannot.  The raw peak's lag independently
    # re-measures the output-path delay; the raw→clean peak drop is the
    # actual echo suppression.
    from scipy.signal import correlate

    ref16 = np.interp(
        np.arange(round(len(wav) * RATE / sr)) * sr / RATE,
        np.arange(len(wav), dtype=np.float64),
        wav if wav.ndim == 1 else wav.mean(1)).astype(np.float32)
    ref16 -= ref16.mean()

    def coupling(mic: np.ndarray) -> tuple[float, float]:
        mic = mic - mic.mean()
        corr = correlate(mic, ref16, mode="valid", method="fft")
        peak = int(np.argmax(np.abs(corr)))
        window = mic[peak:peak + len(ref16)]
        denominator = float(np.linalg.norm(ref16) *
                            max(np.linalg.norm(window), 1e-9))
        return float(np.abs(corr[peak]) / denominator), 1000.0 * peak / RATE

    raw_coupling, raw_lag_ms = coupling(raw_cat)
    clean_coupling, _ = coupling(clean_cat)
    import soundfile as _sf

    dump = out_path.with_suffix("")
    dump.mkdir(parents=True, exist_ok=True)
    _sf.write(str(dump / "raw.wav"), raw_cat, RATE)
    _sf.write(str(dump / "clean.wav"), clean_cat, RATE)
    _sf.write(str(dump / "ref16.wav"), ref16, RATE)

    result = {
        "schema_version": 1,
        "kind": "aec_path_fixed_waveform_probe",
        "reference_mode": reference_mode,
        "output_latency_ms": output_latency_ms,
        "playback_seconds": len(wav) / sr,
        "processed_seconds": len(raw_cat) / RATE,
        "speex_filter_ms": filter_ms,
        "attenuation_db": aec.attenuation_db,
        "raw_rms": float(np.sqrt(np.mean(np.square(raw_cat)))),
        "clean_rms": float(np.sqrt(np.mean(np.square(clean_cat)))),
        "echo_coupling_raw": raw_coupling,
        "echo_coupling_clean": clean_coupling,
        "echo_lag_from_coupling_ms": raw_lag_ms,
        "coupling_reduction_db": (
            20.0 * float(np.log10(max(raw_coupling, 1e-9) /
                                  max(clean_coupling, 1e-9)))),
        "waveform_dir": str(dump),
        "raw_reference_correlation_last": aec.raw_reference_correlation,
        "residual_reference_correlation_last":
            aec.residual_reference_correlation,
        "alignment_corrections": aec.alignment_corrections,
        "estimated_lag_ms": aec.estimated_lag_ms,
        "estimated_drift_ppm": aec.estimated_drift_ppm,
        "reference_underflow_seconds":
            aec.reference_underflow_samples / RATE,
        "note": "room background acts as constant double-talk; compare "
                "modes against each other, not against silence",
    }
    aec.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", choices=("chirp", "aec"), required=True)
    parser.add_argument("--reference-mode",
                        choices=("wallclock", "contiguous"),
                        default="wallclock")
    parser.add_argument("--filter-ms", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=18.0,
                        help="aec probe fixed-playback length")
    parser.add_argument("--output-latency-ms", type=float, default=0.0,
                        help="shift the reference by the measured constant "
                             "output-path latency (chirp probe)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.probe == "chirp":
        out = args.output or Path("research/results/aec-path-chirp.json")
        result = probe_chirp(out, args.filter_ms)
    else:
        out = args.output or Path(
            f"research/results/aec-path-{args.reference_mode}.json")
        result = probe_aec(out, args.reference_mode, args.filter_ms,
                           args.seconds, args.output_latency_ms)
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
