#!/usr/bin/env python3
"""Create the CPU-only starting checkpoint for the large-v3-turbo A/B.

This is deliberately not training. It copies the proven Gemma-width blocks
from the selected medium contextual adapter and initializes a new encoder-
facing 2560-wide input stack for Whisper large-v3-turbo's 1280-wide frames.
The output is a normal runtime checkpoint and is validated before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.core.runtime.stock_whisper import (  # noqa: E402
    LARGE_WHISPER,
    validate_checkpoint,
)
from training.context_distill import (  # noqa: E402
    atomic_save,
    retarget_encoder_arrays,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    resume = args.resume.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == resume:
        raise ValueError("large port output must not overwrite its medium source")

    medium = validate_checkpoint(resume)
    values = retarget_encoder_arrays(medium, LARGE_WHISPER, args.seed)
    values.update({
        "stage": np.array("large_v3_turbo_fresh_input_port"),
        "trainer": np.array("training/prepare_large_port.py"),
        "port_source": np.array(str(resume)),
        "port_source_sha256": np.array(sha256(resume)),
        "port_seed": np.array(args.seed, dtype=np.int32),
        "input_step": np.array(0, dtype=np.int32),
        "input_full_step": np.array(0, dtype=np.int32),
    })
    atomic_save(output, values)
    loaded = validate_checkpoint(output)
    report = {
        "output": str(output),
        "sha256": sha256(output),
        "encoder": str(loaded["encoder"]),
        "input_dim": int(loaded["input_dim"]),
        "pool_step": int(loaded["pool_step"]),
        "stack_input": int(loaded["stack_w1"].shape[1]),
        "output_dim": int(loaded["stack_w2"].shape[0]),
        "source_sha256": str(loaded["port_source_sha256"]),
    }
    if report["stack_input"] != report["input_dim"] * report["pool_step"]:
        raise AssertionError("large stack input shape does not match frame stacking")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
