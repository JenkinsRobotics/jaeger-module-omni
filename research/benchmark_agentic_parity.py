"""Verify the embedded agent contract against its pinned read-only source.

This is a structural benchmark, not a model-quality score.  It compares both
81-case routing corpora and the registered tool catalog in isolated Python
processes so imports from one package cannot contaminate the other registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_AGENT_COMMIT = "a3696e9e78d64e1b1996bacc036902b60eafe30e"
LEGACY_OS_COMMIT = "e05dc24eec3ff09af6132b7b0e4fca96d059c494"


PROBE = r"""
import dataclasses, hashlib, json
if {legacy!r}:
    from jaeger_agent.bench import CASES
    from jaeger_agent.bench.cases_b import CASES_B
    import jaeger_agent.tools
    from jaeger_os.core.tools.tool_registry import get_tools
else:
    from jaeger_agent_omni.core.agentic.bench import CASES
    from jaeger_agent_omni.core.agentic.bench.cases_b import CASES_B
    import jaeger_agent_omni.tools
    from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.tools.tool_registry import get_tools
def digest(rows):
    raw=json.dumps([dataclasses.asdict(x) for x in rows],sort_keys=True,separators=(",",":"))
    return hashlib.sha256(raw.encode()).hexdigest()
print(json.dumps({{
    "cases_a": len(CASES), "cases_b": len(CASES_B),
    "cases_a_sha256": digest(CASES), "cases_b_sha256": digest(CASES_B),
    "tools": sorted(t.name for t in get_tools()),
}}, sort_keys=True))
"""


def _commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


def _probe(*, legacy: bool, python: Path, agent: Path, jaeger_os: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if legacy:
        env["PYTHONPATH"] = os.pathsep.join((str(agent), str(jaeger_os)))
    else:
        env["PYTHONPATH"] = str(ROOT)
    output = subprocess.check_output(
        [str(python), "-c", PROBE.format(legacy=legacy)],
        cwd=ROOT, env=env, text=True,
    )
    return json.loads(output.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-agent", type=Path,
        default=Path.home() / "GITHUB/jaeger-module-agent",
    )
    parser.add_argument(
        "--legacy-os", type=Path, default=Path.home() / "GITHUB/JaegerOS",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "research/results/agentic-source-parity.json",
    )
    args = parser.parse_args()

    agent_commit = _commit(args.legacy_agent)
    os_commit = _commit(args.legacy_os)
    legacy = _probe(
        legacy=True, python=Path(sys.executable),
        agent=args.legacy_agent, jaeger_os=args.legacy_os,
    )
    embedded = _probe(
        legacy=False, python=Path(sys.executable),
        agent=args.legacy_agent, jaeger_os=args.legacy_os,
    )
    checks = {
        "pinned_agent_commit": agent_commit == LEGACY_AGENT_COMMIT,
        "pinned_os_commit": os_commit == LEGACY_OS_COMMIT,
        "cases_a_exact": embedded["cases_a_sha256"] == legacy["cases_a_sha256"],
        "cases_b_exact": embedded["cases_b_sha256"] == legacy["cases_b_sha256"],
        "tool_names_exact": embedded["tools"] == legacy["tools"],
    }
    payload = {
        "kind": "agentic_embedded_source_parity",
        "legacy_agent_commit": agent_commit,
        "legacy_os_commit": os_commit,
        "checks": checks,
        "legacy": legacy,
        "embedded": embedded,
        "all_pass": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(
        f"legacy/embedded: {legacy['cases_a']}+{legacy['cases_b']} cases, "
        f"{len(legacy['tools'])} tools"
    )
    print(f"wrote {args.out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
