"""Small, auditable orchestration primitives for the Omni release benchmark.

The individual evaluators remain ordinary repository commands.  This module
adds the part shell scripts usually get wrong: explicit PASS/FAIL/SKIP state,
timeouts, durable logs, incremental JSON, and a release verdict that never
counts missing physical evidence as success.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


Assessment = Callable[["GateSpec", int], tuple[bool, str, dict[str, Any]]]


@dataclass(frozen=True)
class GateSpec:
    id: str
    group: str
    description: str
    command: tuple[str, ...] = ()
    timeout_s: float = 600.0
    required: bool = True
    artifact: Path | None = None
    assess: Assessment | None = None
    skip_reason: str | None = None


@dataclass
class GateResult:
    id: str
    group: str
    description: str
    status: str
    required: bool
    elapsed_s: float = 0.0
    returncode: int | None = None
    command: list[str] = field(default_factory=list)
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact: str | None = None
    log: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def exit_zero(_: GateSpec, returncode: int) -> tuple[bool, str, dict[str, Any]]:
    return returncode == 0, f"exit {returncode}", {}


def run_gate(spec: GateSpec, *, cwd: Path, log_dir: Path,
             env: dict[str, str] | None = None) -> GateResult:
    if spec.skip_reason:
        return GateResult(
            id=spec.id, group=spec.group, description=spec.description,
            status="SKIP", required=spec.required, detail=spec.skip_reason,
            artifact=str(spec.artifact) if spec.artifact else None,
        )
    if not spec.command:
        raise ValueError(f"gate {spec.id} has neither command nor skip reason")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.id}.log"
    started = time.monotonic()
    run_env = dict(os.environ)
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        run_env.update(env)
    returncode: int | None = None
    detail = ""
    try:
        completed = subprocess.run(
            list(spec.command), cwd=cwd, env=run_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=spec.timeout_s, check=False,
        )
        returncode = completed.returncode
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        assessor = spec.assess or exit_zero
        passed, detail, metrics = assessor(spec, completed.returncode)
        status = "PASS" if passed else "FAIL"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        log_path.write_text(output, encoding="utf-8")
        status, metrics = "FAIL", {}
        detail = f"timeout after {spec.timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001 - failures belong in the report
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        status, metrics = "FAIL", {}
        detail = f"{type(exc).__name__}: {exc}"
    return GateResult(
        id=spec.id, group=spec.group, description=spec.description,
        status=status, required=spec.required,
        elapsed_s=round(time.monotonic() - started, 3),
        returncode=returncode, command=list(spec.command), detail=detail,
        metrics=metrics,
        artifact=str(spec.artifact) if spec.artifact else None,
        log=str(log_path),
    )


def summarize(results: list[GateResult]) -> dict[str, Any]:
    counts = {name: sum(row.status == name for row in results)
              for name in ("PASS", "FAIL", "SKIP")}
    required_failures = [r.id for r in results
                         if r.required and r.status == "FAIL"]
    required_skips = [r.id for r in results
                      if r.required and r.status == "SKIP"]
    groups: dict[str, dict[str, int]] = {}
    for row in results:
        slot = groups.setdefault(row.group, {"PASS": 0, "FAIL": 0, "SKIP": 0})
        slot[row.status] += 1
    return {
        **{key.lower(): value for key, value in counts.items()},
        "total": len(results),
        "required_failures": required_failures,
        "required_skips": required_skips,
        "complete": not required_skips,
        "profile_pass": not required_failures and not required_skips,
        "groups": groups,
        "elapsed_s": round(sum(row.elapsed_s for row in results), 3),
    }


def report_payload(*, profile: str, root: Path, results: list[GateResult],
                   git_commit: str, dirty: bool) -> dict[str, Any]:
    summary = summarize(results)
    # Only the release profile contains the long agent corpus, AEC gate, and
    # required physical evidence. A green CPU/model subset must never render
    # as a production-ready verdict.
    summary["release_ready"] = bool(
        profile == "release" and summary["profile_pass"])
    return {
        "schema_version": 1,
        "kind": "jaeger_omni_deep_benchmark",
        "evaluation": "free_running_where_model_is_used",
        "teacher_forced": False,
        "profile": profile,
        "repository": str(root),
        "git_commit": git_commit,
        "dirty_at_start": dirty,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "summary": summary,
        "gates": [asdict(row) for row in results],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    if report["profile"] == "release":
        verdict = "READY" if summary["release_ready"] else (
            "INCOMPLETE" if summary["required_skips"] and
            not summary["required_failures"] else "NOT READY")
    else:
        verdict = "PROFILE PASS" if summary["profile_pass"] else (
            "PROFILE INCOMPLETE" if summary["required_skips"] and
            not summary["required_failures"] else "PROFILE FAIL")
    lines = [
        "# Jaeger Agent Omni deep benchmark", "",
        f"**Verdict: {verdict}**", "",
        f"Profile: `{report['profile']}` · commit `{report['git_commit'][:12]}` · "
        f"{summary['pass']} pass / {summary['fail']} fail / "
        f"{summary['skip']} skip · {summary['elapsed_s']:.1f}s", "",
        "Model-backed gates are free-running. A skipped required gate is "
        "INCOMPLETE, never PASS.", "",
        "| Group | Gate | Status | Required | Seconds | Detail |",
        "|---|---|:---:|:---:|---:|---|",
    ]
    for gate in report["gates"]:
        detail = str(gate.get("detail") or "").replace("|", "\\|")
        lines.append(
            f"| {gate['group']} | `{gate['id']}` | **{gate['status']}** | "
            f"{'yes' if gate['required'] else 'no'} | "
            f"{gate['elapsed_s']:.2f} | {detail} |"
        )
    if summary["required_failures"]:
        lines.extend(("", "Required failures: " + ", ".join(
            f"`{name}`" for name in summary["required_failures"])))
    if summary["required_skips"]:
        lines.extend(("", "Missing required evidence: " + ", ".join(
            f"`{name}`" for name in summary["required_skips"])))
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (directory / "report.md").write_text(
        render_markdown(report), encoding="utf-8")


__all__ = [
    "GateResult", "GateSpec", "exit_zero", "read_json", "render_markdown",
    "report_payload", "run_gate", "summarize", "write_report",
]
