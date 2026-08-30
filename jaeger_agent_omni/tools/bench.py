"""Agent-callable entry point for the safe, non-recursive CPU profile.

Model and release profiles are intentionally CLI-only. Launching another
Gemma session from inside an active voice turn would violate the repository's
one-model-job rule and can corrupt latency measurements.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jaeger_agent_omni.core.agentic.workspace import _require_layout
from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.safety.permissions import (
    PermissionTier, requires_tier,
)


@requires_tier(
    PermissionTier.WRITE_LOCAL,
    skill="bench",
    operation="run_benchmark",
    summary="run the agent self-benchmark against the live pipeline",
)
def run_benchmark(
    tags: str = "",
    limit: int = 0,
    ids: str = "",
    save: bool = True,
    hermetic: bool = True,
) -> dict[str, Any]:
    """Run every CPU-safe Omni contract and return its durable scorecard.

    ``save`` and ``hermetic`` remain in the tool schema for compatibility;
    the CPU profile is always durable and does not mutate agent memory.
    Filtered native-audio corpus runs belong to the external research CLI.
    """
    # The copied implementation used to import JaegerAI's host-only runner
    # here, making the tool fail in the standalone Omni package. A live Omni
    # model cannot recursively launch another model benchmark without
    # violating the one-GPU-job rule, so the agent-callable form runs the
    # comprehensive CPU profile. Model/release profiles are explicit CLI jobs.
    if tags or ids or int(limit or 0):
        return {
            "ok": False,
            "error": "filtered live-model corpus runs use "
                     "research/benchmark_agentic_corpus.py outside an active turn",
        }
    import subprocess
    import sys

    layout = _require_layout()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(layout.logs_dir) / "bench" / "omni-cpu" / stamp
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "jaeger_agent_omni.benchmark",
         "--profile", "cpu", "--out-dir", str(out_dir)],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=180, check=False,
    )
    report_path = out_dir / "report.json"
    if not report_path.is_file():
        return {"ok": False, "error": f"benchmark exit {completed.returncode}",
                "output": completed.stdout[-1000:]}
    report = json.loads(report_path.read_text())
    return {
        "ok": completed.returncode == 0,
        "profile": "cpu",
        "summary": report["summary"],
        "report_dir": str(out_dir),
        "next": "run `python -m jaeger_agent_omni.benchmark --profile release` "
                "outside the active voice turn for model and physical gates",
    }
__all__ = ["run_benchmark"]
