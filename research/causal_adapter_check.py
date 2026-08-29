#!/usr/bin/env python3
"""Verify causal prefix stability and perception-head wiring for one node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front-end", required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx

    from jaeger_agent_omni.omni import build

    features = np.load(args.feature).astype(np.float32)
    if features.ndim != 2 or len(features) < 8:
        raise ValueError("feature array must contain at least eight frames")
    session, meta = build(front_end=args.front_end)
    adapter = session.adapter
    if not hasattr(adapter, "perceive"):
        raise ValueError(f"{args.front_end} has no causal perception heads")

    split = max(4, (len(features) // 2) // 2 * 2)
    full, full_heads = adapter.perceive(mx.array(features))
    prefix, prefix_heads = adapter.perceive(mx.array(features[:split]))
    mx.eval(full, prefix, full_heads, prefix_heads)
    prefix_frames = int(prefix.shape[0])
    drift = float(mx.max(mx.abs(full[:prefix_frames] - prefix)))

    n = int(full.shape[0])
    zeros = np.zeros(n, np.float32)
    ones = np.ones(n, np.float32)
    emb_zero, heads_zero = adapter.perceive(mx.array(features), zeros)
    emb_one, heads_one = adapter.perceive(mx.array(features), ones)
    mx.eval(emb_zero, emb_one, heads_zero, heads_one)
    self_speech_delta = float(mx.max(mx.abs(emb_one - emb_zero)))

    expected = {
        "speaker_change": (n,),
        "overlap": (n,),
        "turn_end": (n,),
        "speaker_id": (n, 4),
    }
    heads = {}
    for name, shape in expected.items():
        values = np.asarray(full_heads[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise AssertionError(f"{name}: expected finite {shape}, got {values.shape}")
        heads[name] = {
            "shape": list(values.shape),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    if drift >= 1e-4:
        raise AssertionError(f"future frames changed causal prefix: {drift}")
    if self_speech_delta <= 1e-4:
        raise AssertionError("self-speech input is disconnected")

    payload = {
        "schema_version": 1,
        "kind": "causal_adapter_integrity",
        "front_end": args.front_end,
        "checkpoint": meta["checkpoint"],
        "feature": str(args.feature),
        "input_frames_50hz": len(features),
        "output_frames_25hz": n,
        "prefix_frames": prefix_frames,
        "causal_prefix_max_abs_drift": drift,
        "causality_pass": True,
        "self_speech_embedding_max_abs_delta": self_speech_delta,
        "self_speech_connected": True,
        "heads": heads,
        "calibration_validated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
