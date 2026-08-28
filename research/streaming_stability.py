#!/usr/bin/env python3
"""Measure whether stock-Whisper adapter vectors are safe to commit early.

This is a representation diagnostic, not a WER or promotion score. For each
growing waveform prefix it compares both Whisper states and final adapter
vectors with the corresponding positions from one full-utterance pass. A
right-context sweep measures how much audio must remain uncommitted before
earlier vectors stop changing.

The current Whisper encoder and 41M context adapter are bidirectional. This
program measures their revision; it does not feed provisional vectors into
Gemma's irreversible KV cache or claim that the runtime is causal.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16_000
WHISPER_HZ = 50.0


def vector_error(candidate: np.ndarray, final: np.ndarray) -> dict:
    """Return per-frame direction and magnitude error for aligned matrices."""
    candidate = np.asarray(candidate, dtype=np.float32)
    final = np.asarray(final, dtype=np.float32)
    if candidate.shape != final.shape or candidate.ndim != 2:
        raise ValueError(
            f"aligned 2-D shapes required: {candidate.shape}, {final.shape}")
    if not len(candidate):
        raise ValueError("cannot score zero frames")
    dot = np.sum(candidate * final, axis=-1)
    candidate_norm = np.linalg.norm(candidate, axis=-1)
    final_norm = np.linalg.norm(final, axis=-1)
    cosine = dot / np.maximum(candidate_norm * final_norm, 1e-8)
    relative_l2 = np.linalg.norm(candidate - final, axis=-1) / np.maximum(
        final_norm, 1e-8)
    return {"cosine": cosine, "relative_l2": relative_l2}


def summarize(rows: list[dict], right_context_s: float, key: str,
              default_rate_hz: float | None = None) -> dict:
    cosines: list[float] = []
    relative_l2: list[float] = []
    comparisons = 0
    retained = 0
    for row in rows:
        rate = default_rate_hz or float(row[f"{key}_rate_hz"])
        holdback = math.ceil(right_context_s * rate)
        candidate = row[f"prefix_{key}"]
        final = row[f"final_{key}"]
        usable = min(len(candidate), len(final)) - holdback
        if usable <= 0:
            continue
        errors = vector_error(candidate[:usable], final[:usable])
        cosines.extend(errors["cosine"].astype(float).tolist())
        relative_l2.extend(errors["relative_l2"].astype(float).tolist())
        comparisons += 1
        retained += usable
    cosine_p05 = float(np.percentile(cosines, 5)) if cosines else None
    relative_p95 = (
        float(np.percentile(relative_l2, 95)) if relative_l2 else None)
    return {
        "right_context_s": right_context_s,
        "prefix_comparisons": comparisons,
        "compared_positions": retained,
        "cosine_median": statistics.median(cosines) if cosines else None,
        "cosine_p05": cosine_p05,
        "relative_l2_median": (
            statistics.median(relative_l2) if relative_l2 else None),
        "relative_l2_p95": relative_p95,
        # Strict engineering signal only; downstream free-running gates remain
        # mandatory before enabling cached prefill.
        "strict_stability_pass": bool(
            cosine_p05 is not None and cosine_p05 >= 0.999 and
            relative_p95 is not None and relative_p95 <= 0.05),
    }


def selfcheck() -> None:
    base = np.eye(3, dtype=np.float32)
    same = vector_error(base, base)
    assert np.allclose(same["cosine"], 1.0)
    assert np.allclose(same["relative_l2"], 0.0)
    changed = vector_error(base * 2, base)
    assert np.allclose(changed["cosine"], 1.0)
    assert np.allclose(changed["relative_l2"], 1.0)
    rows = [{
        "prefix_probe": base,
        "final_probe": base,
        "probe_rate_hz": 1.0,
    }]
    report = summarize(rows, 1.0, "probe")
    assert report["compared_positions"] == 2
    assert report["strict_stability_pass"]
    print("streaming_stability: cosine, magnitude, right-context gates OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-end", default="jaegeromni-41m-medium")
    parser.add_argument("--cases", type=Path,
                        default=ROOT / "research/cases12")
    parser.add_argument("--cadence-s", type=float, default=1.0)
    parser.add_argument("--min-prefix-s", type=float, default=1.0)
    parser.add_argument("--right-context-s", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.output is None:
        parser.error("--output is required unless --selfcheck is used")
    if args.cadence_s <= 0 or args.min_prefix_s <= 0:
        parser.error("cadence and minimum prefix must be positive")
    if any(value < 0 for value in args.right_context_s):
        parser.error("right context cannot be negative")

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.omni import build

    cases = json.loads((args.cases / "cases.json").read_text())
    session, meta = build(front_end=args.front_end)
    if meta["front_end"] != "stock":
        raise SystemExit("streaming stability currently requires a stock node")
    mx = session.mx

    def project(features: np.ndarray) -> np.ndarray:
        output = session.adapter(mx.array(features))
        mx.eval(output)
        return np.asarray(output, dtype=np.float32)

    rows = []
    case_reports = []
    started = time.monotonic()
    for case in cases:
        audio = read_wav(args.cases / f"{case['key']}.wav")
        full_features = session.front_end(audio)
        full_adapter = project(full_features)
        duration_s = len(audio) / SAMPLE_RATE
        cut_s = args.min_prefix_s
        case_rows = 0
        while cut_s < duration_s - 1e-6:
            cut = min(len(audio), int(round(cut_s * SAMPLE_RATE)))
            prefix_features = session.front_end(audio[:cut])
            prefix_adapter = project(prefix_features)
            rows.append({
                "case": case["key"],
                "prefix_s": cut / SAMPLE_RATE,
                "prefix_whisper": prefix_features,
                "final_whisper": full_features,
                "prefix_adapter": prefix_adapter,
                "final_adapter": full_adapter,
                # Pooling is defined in encoder frames, so its nominal rate is
                # exact even when a clip tail is discarded by integer pooling.
                "adapter_rate_hz": WHISPER_HZ / session.pool_step,
            })
            case_rows += 1
            cut_s += args.cadence_s
        case_reports.append({
            "case": case["key"],
            "duration_s": duration_s,
            "prefixes": case_rows,
            "whisper_frames": len(full_features),
            "adapter_tokens": len(full_adapter),
        })
        print(f"{case['key']}: {duration_s:.2f}s, {case_rows} prefixes",
              flush=True)

    whisper_sweep = [
        summarize(rows, context, "whisper", WHISPER_HZ)
        for context in args.right_context_s
    ]
    adapter_sweep = [
        summarize(rows, context, "adapter")
        for context in args.right_context_s
    ]
    payload = {
        "schema_version": 1,
        "kind": "streaming_representation_stability",
        "diagnostic_only": True,
        "promotion_metric": False,
        "front_end": args.front_end,
        "model_id": meta["model_id"],
        "checkpoint": meta["checkpoint"],
        "cadence_s": args.cadence_s,
        "cases": case_reports,
        "prefix_comparisons": len(rows),
        "whisper_sweep": whisper_sweep,
        "adapter_sweep": adapter_sweep,
        "elapsed_s": time.monotonic() - started,
        "interpretation": (
            "Only vectors passing a selected right-context gate become "
            "candidates for later free-running cached-prefill evaluation."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "whisper_sweep": whisper_sweep,
        "adapter_sweep": adapter_sweep,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
