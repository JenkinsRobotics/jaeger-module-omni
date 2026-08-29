"""Versioned, machine-readable production run reports."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1


def latency_summary(values) -> dict:
    """Return stable median/p95/max fields for a latency sample."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "median_ms": None, "p95_ms": None, "max_ms": None}
    # Conservative nearest-rank p95.  Rounded zero-based indexing hides one
    # slow sample in small live sessions (the same defect once hid a 300 ms
    # AEC event behind a reported 100 ms p95).
    rank = max(0, min(len(ordered) - 1,
                      math.ceil(0.95 * len(ordered)) - 1))
    return {
        "n": len(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[rank],
        "max_ms": ordered[-1],
    }


def memory_summary() -> dict:
    """Best-effort process memory for a run report.  NEVER raises.

    RSS alone cannot see MPS/Metal allocations (STATUS.md, measurement
    rules), so the MLX active/peak counters are reported alongside it when
    the runtime is present.  Missing sources stay ``None`` rather than
    aborting a live session.
    """
    import sys

    out = {"rss_peak_mb": None, "mlx_active_mb": None, "mlx_peak_mb": None}
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS and kilobytes on Linux.
        out["rss_peak_mb"] = round(
            peak / (1e6 if sys.platform == "darwin" else 1e3), 1)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import mlx.core as mx

        metal = getattr(mx, "metal", None)
        active = (getattr(mx, "get_active_memory", None) or
                  getattr(metal, "get_active_memory", None))
        peak_fn = (getattr(mx, "get_peak_memory", None) or
                   getattr(metal, "get_peak_memory", None))
        if callable(active):
            out["mlx_active_mb"] = round(active() / 1e6, 1)
        if callable(peak_fn):
            out["mlx_peak_mb"] = round(peak_fn() / 1e6, 1)
    except Exception:                                    # noqa: BLE001
        pass
    return out


def new_run_report(*, phase: str, model_id: str, model_name: str,
                   checkpoint: str, configuration: dict,
                   source: str | None = None) -> dict:
    """Create the immutable header for one production-test result."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "jaeger_duplex_run",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phase": str(phase),
        "model": {
            "id": str(model_id),
            "name": str(model_name),
            "checkpoint": str(checkpoint),
        },
        "source": source,
        "configuration": configuration,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
    }


def finish_run_report(report: dict, *, metrics: dict, replies: list[str],
                      speaker_labels: list[str], interactions: list[dict],
                      errors: list[str], artifacts: dict | None = None) -> dict:
    """Attach final measured state without combining different checkpoints."""
    report = dict(report)
    report.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "replies": list(replies),
        "speaker_labels": list(speaker_labels),
        "interactions": list(interactions),
        "errors": list(errors),
        "artifacts": dict(artifacts or {}),
        "passed_runtime": not errors,
    })
    return report


def default_report_path(root: str | Path, model_id: str, phase: str) -> Path:
    """Choose a collision-resistant local report path."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(root) / "reports" / f"{stamp}-{model_id}-{phase}.json"


def write_report(path: str | Path, report: dict) -> Path:
    """Atomically write one JSON report so partial runs are never mistaken."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False) as tmp:
        json.dump(report, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temporary = Path(tmp.name)
    temporary.replace(destination)
    return destination


def demo() -> None:
    summary = latency_summary([10, 20, 30, 40])
    assert summary == {"n": 4, "median_ms": 25.0,
                       "p95_ms": 40.0, "max_ms": 40.0}
    tail = latency_summary([100] * 11 + [300])
    assert tail["p95_ms"] == 300 and tail["max_ms"] == 300
    report = new_run_report(
        phase="direct-wav", model_id="gemma-native",
        model_name="Gemma Native", checkpoint="built-in",
        configuration={"seconds": 300})
    report = finish_run_report(
        report, metrics={"fed_seconds": 300.0}, replies=[],
        speaker_labels=[], interactions=[], errors=[])
    assert report["schema_version"] == 1 and report["passed_runtime"]
    with tempfile.TemporaryDirectory() as directory:
        saved = write_report(Path(directory) / "run.json", report)
        assert json.loads(saved.read_text())["kind"] == "jaeger_duplex_run"
    memory = memory_summary()
    assert set(memory) == {"rss_peak_mb", "mlx_active_mb", "mlx_peak_mb"}
    assert memory["rss_peak_mb"] is None or memory["rss_peak_mb"] > 0
    print("runtime.reporting: schema, latency tails, memory summary, and "
          "finalization OK")


if __name__ == "__main__":
    demo()
