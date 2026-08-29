#!/usr/bin/env python3
"""Interpolate compatible owned adapter checkpoints without training."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from jaeger_agent_omni.runtime.stock_whisper import validate_checkpoint


CONTEXT_ARCHITECTURES = frozenset({
    "cleanroom_context_v2", "streaming_causal_v1",
})


def atomic_save(path: Path, values: dict[str, np.ndarray]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True,
                        help="candidate weight; zero is base, one candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-architecture", choices=sorted(CONTEXT_ARCHITECTURES),
        help=("required when crossing bidirectional/causal masks; also "
              "selects which checkpoint contributes architecture-only heads"))
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")

    base = validate_checkpoint(args.base)
    candidate = validate_checkpoint(args.candidate)
    base_arch = str(base["adapter_arch"])
    candidate_arch = str(candidate["adapter_arch"])
    output_arch = args.output_architecture or base_arch
    if base_arch != candidate_arch:
        if ({base_arch, candidate_arch} != CONTEXT_ARCHITECTURES or
                args.output_architecture is None):
            raise ValueError(
                "adapter architectures differ; compatible context masks "
                "require explicit --output-architecture")
    elif args.output_architecture not in (None, base_arch):
        raise ValueError("cannot invent a different output architecture")
    if output_arch == base_arch:
        values = dict(base)
    elif output_arch == candidate_arch:
        # Preserve causal perception heads and self-speech projection. Only
        # tensors present in both checkpoints are blended below.
        values = dict(candidate)
    else:
        raise ValueError("selected output architecture is not an input")
    interpolated = []
    for key, left in base.items():
        if (key not in candidate or left.shape != candidate[key].shape
                or left.ndim == 0 or left.dtype.kind != "f"):
            continue
        right = candidate[key]
        values[key] = ((1.0 - args.alpha) * left.astype(np.float32)
                       + args.alpha * right.astype(np.float32)).astype(
                           left.dtype)
        interpolated.append(key)
    values.update({
        "adapter_arch": np.array(output_arch),
        "stage": np.array("checkpoint_interpolation"),
        "interpolation_alpha": np.array(args.alpha, dtype=np.float32),
        "interpolation_base": np.array(str(args.base.resolve())),
        "interpolation_candidate": np.array(str(args.candidate.resolve())),
        "trainer": np.array("training/interpolate_adapter.py"),
    })
    atomic_save(args.output, values)
    validate_checkpoint(args.output)
    print(f"interpolated {len(interpolated)} tensor arrays at "
          f"alpha={args.alpha:.3f}, arch={output_arch}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
