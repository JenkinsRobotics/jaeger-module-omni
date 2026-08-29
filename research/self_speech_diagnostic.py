#!/usr/bin/env python3
"""Measure the causal adapter's None-versus-zero self-speech gap.

The live runtime calls ``adapter(x)`` with no self-speech flags, which SKIPS
``self_proj`` entirely.  Training and ``perceive(x, zeros)`` apply the layer,
so a nonzero ``self_proj`` bias makes flag-free deployment differ from the
flag=0 training convention by exactly that bias term.  This script measures
the gap on one cached feature file; it changes no live behavior and promotes
nothing.

Run it ALONE — it loads the full Whisper/adapter/Gemma stack, so it must not
share the accelerator with another job.

    python research/self_speech_diagnostic.py \
        --front-end jaegeromni-41m-causal-balanced \
        --feature FEATURES.npy --output research/results/self-speech.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--front-end", required=True)
    parser.add_argument("--feature", type=Path, required=True,
                        help="cached 50 Hz encoder-state .npy, as used by "
                             "causal_adapter_check.py")
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

    x = mx.array(features)
    deployed = adapter(x)                       # the live __call__ path
    none_emb, _ = adapter.perceive(x)           # trunk without flags
    n = int(none_emb.shape[0])
    zero_emb, _ = adapter.perceive(x, np.zeros(n, np.float32))
    one_emb, _ = adapter.perceive(x, np.ones(n, np.float32))
    mx.eval(deployed, none_emb, zero_emb, one_emb)

    def max_abs(a, b) -> float:
        return float(mx.max(mx.abs(a - b)))

    bias = adapter.self_proj.bias
    weight = adapter.self_proj.weight
    mx.eval(bias, weight)
    mean_norm = float(mx.mean(mx.sqrt(mx.sum(zero_emb * zero_emb, axis=-1))))
    none_vs_zero = max_abs(none_emb, zero_emb)
    payload = {
        "schema_version": 1,
        "kind": "self_speech_deployment_diagnostic",
        "front_end": args.front_end,
        "checkpoint": meta["checkpoint"],
        "feature": str(args.feature),
        "input_frames_50hz": len(features),
        "output_frames_25hz": n,
        "output_mode": getattr(session, "output_mode", None),
        # __call__ must be exactly perceive-without-flags; anything else
        # means the diagnostic is not measuring the deployed computation.
        "call_vs_perceive_none_max_abs_delta": max_abs(deployed, none_emb),
        "none_vs_zero_max_abs_delta": none_vs_zero,
        "zero_vs_one_max_abs_delta": max_abs(zero_emb, one_emb),
        "mean_embedding_l2_norm": mean_norm,
        "self_proj_bias_l2": float(mx.sqrt(mx.sum(bias * bias))),
        "self_proj_bias_max_abs": float(mx.max(mx.abs(bias))),
        "self_proj_weight_l2": float(mx.sqrt(mx.sum(weight * weight))),
        # The mismatch exists whenever flag-free deployment differs from the
        # flag=0 training convention beyond numeric noise.
        "deployment_training_mismatch": bool(none_vs_zero > 1e-4),
        "measurement_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
