#!/usr/bin/env python3
"""Import a measured Gemma-native cleanroom checkpoint into Jaeger format.

The source contains only the owned LayerNorm/MLP tensors.  This importer adds
strict runtime metadata; it does not transform, quantize, or retrain weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    required = ("ln_gamma", "ln_beta", "w1", "b1", "w2", "b2")
    with np.load(args.source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in required}
    hidden = int(arrays["w1"].shape[0])
    expected = {
        "ln_gamma": (1024,), "ln_beta": (1024,),
        "w1": (hidden, 1024), "b1": (hidden,),
        "w2": (2560, hidden), "b2": (2560,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{args.source}: {key} is {arrays[key].shape}, "
                             f"expected {shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{args.source}: {key} contains non-finite values")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.destination, **arrays,
        adapter_arch=np.array("gemma_native_distill_v1"),
        hidden_dim=np.array(hidden, dtype=np.int32),
        input_boundary=np.array("stock_whisper_encoder_50hz"),
        encoder=np.array("openai/whisper-medium"),
        lineage=np.array("qwen_free"),
        teacher_lineage=np.array("google_gemma_native_audio_tower"),
        pool_step=np.array(2, dtype=np.int32),
        output_mode=np.array("raw"),
    )
    print(f"imported {args.source} -> {args.destination} "
          f"({hidden=}, pool_step=2, output_mode=raw)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
