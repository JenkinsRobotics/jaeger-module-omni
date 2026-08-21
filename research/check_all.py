"""Run every self-check across the repo. One command.

    python research/check_all.py

Covers the package contracts, the research experiments, and the training
policy. Everything runs offline in a couple of seconds.

stage0_access.py is excluded — it needs real weights, so it stays manual.
"""

import importlib
import sys
import time
from pathlib import Path

# Package contracts import as jaeger_agent_omni.*; research/training modules
# import flat. Both are exercised here so one command covers the repo.
PACKAGE_MODULES = [
    "jaeger_agent_omni.performance_spec",
    "jaeger_agent_omni.outputs",
    "jaeger_agent_omni.capture",
    "jaeger_agent_omni.streaming",
    "jaeger_agent_omni.prompt",
    "jaeger_agent_omni.runtime.gemma_audio",
    "jaeger_agent_omni.runtime.session",
    "jaeger_agent_omni.runtime.duplex",
    "jaeger_agent_omni.neural.base",
    "jaeger_agent_omni.neural.audio",
    "jaeger_agent_omni.neural.heads",
    "jaeger_agent_omni.neural.gate",
    "jaeger_agent_omni.neural.delta",
    "jaeger_agent_omni.neural.speech",
    "jaeger_agent_omni.app",
    "jaeger_agent_omni.artifacts",
]

TRAINING_MODULES = [
    "data",
    "config",
    "distill",
    "dataset",
    "make_corpus",   # skips itself where `say` is unavailable
    "import_audio",
    "evaluate",
    "encoder_cache",
    "fetch_dataset",
    "train",         # runs a real 120-step loop on stubs
]

MODULES = [
    "stubs",
    "load_minicpm_projector",   # gguf-optional; degrades to a metric check
    "load_minicpm",
    "overfit_probe",
    "pipeline",      # last: it depends on most of the others
]


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent
    for path in (repo, here, repo / "training"):
        sys.path.insert(0, str(path))

    everything = PACKAGE_MODULES + MODULES + TRAINING_MODULES
    failed = []
    t0 = time.monotonic()
    for name in everything:
        try:
            importlib.import_module(name).demo()
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            failed.append(name)
    elapsed = time.monotonic() - t0

    print(f"\n{len(everything) - len(failed)}/{len(everything)} modules OK in {elapsed:.1f}s")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
