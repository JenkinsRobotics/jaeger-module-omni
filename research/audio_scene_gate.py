#!/usr/bin/env python3
"""Evaluate the neural audio-scene head on source-file-disjoint folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.context_distill import refuse_gpu_contention
from training.perception_model import build, load_arrays
from training.train_scene_head import metrics, rows


def report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    refuse_gpu_contention()

    arrays = load_arrays(args.adapter)
    adapter, _ = build(arrays)
    validation = metrics(adapter, rows("validation"))
    test = metrics(adapter, rows("test"))
    result = {
        "kind": "jaeger_audio_scene_gate",
        "adapter": report_path(args.adapter),
        "classes": ["silence", "speech", "music", "noise"],
        "split": "source-file-disjoint",
        "teacher_forced": False,
        "validation": validation,
        "test": test,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
