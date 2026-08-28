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
    print("runtime.reporting: schema, latency tails, and finalization OK")


if __name__ == "__main__":
    demo()
