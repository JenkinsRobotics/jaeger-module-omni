#!/usr/bin/env python3
"""Run Jaeger's 81+81 agent corpus through Omni as native synthetic audio.

Case text is used only by the external Kokoro fixture builder. Gemma receives
the resulting waveform through Whisper + the selected neural adapter; no case
prompt or transcript is fed into its input path. All mutable tools are bound to
a temporary workspace. The two legacy speech-tool expectations are adapted to
Omni's automatic output router (`read_file` for read-aloud, no TTS call).
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import json
import re
import signal
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jaeger_agent_omni.core.agentic.bench import (  # noqa: E402
    CASES, UMBRELLA_EQUIVALENTS, BenchCase,
)
from jaeger_agent_omni.core.agentic.bench.cases_b import CASES_B  # noqa: E402


MARKUP = (
    "<|tool_call", "<tool_call>", "</tool_call>", "<|/tool_call",
    "<|channel", "<channel|>", "<|message", "<|python_tag",
    "<|im_start", "<|im_end", '<|"|>', "[TOOL_CALLS]", "<think>",
    "</think>",
)
THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


class CaseDeadline(BaseException):
    """Escape tool code that catches ordinary Exception subclasses."""


@contextlib.contextmanager
def case_deadline(seconds: float):
    """Bound a row so a network/tool stall cannot erase a corpus run."""
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise CaseDeadline(f"case exceeded {seconds:.0f}s")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class CorpusPermissionProvider:
    """Approve sandboxed local mutation; deny every external side effect."""

    def confirm(self, request) -> bool:
        from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.safety.permissions import (
            PermissionTier,
        )
        return request.tier == PermissionTier.WRITE_LOCAL


def _selected(suite: str, tags: list[str], ids: list[str],
              limit: int) -> list[tuple[str, BenchCase]]:
    rows = []
    if suite in ("a", "both"):
        rows.extend(("a", case) for case in CASES)
    if suite in ("b", "both"):
        rows.extend(("b", case) for case in CASES_B)
    original = list(rows)
    if tags:
        wanted = {tag.lower() for tag in tags}
        rows = [(lane, case) for lane, case in rows
                if wanted.intersection(tag.lower() for tag in case.tags)]
    if ids:
        wanted_ids = set(ids)
        rows = [(lane, case) for lane, case in rows
                if case.id in wanted_ids or f"{lane}:{case.id}" in wanted_ids]
    if limit > 0:
        rows = rows[:limit]
    # A selected multi-turn row is meaningless without its siblings.
    sessions = {(lane, case.session) for lane, case in rows if case.session}
    if sessions:
        included = {(lane, id(case)) for lane, case in rows}
        for lane, case in original:
            if ((lane, case.session) in sessions and
                    (lane, id(case)) not in included):
                rows.append((lane, case))
        order = {(lane, id(case)): index
                 for index, (lane, case) in enumerate(original)}
        rows.sort(key=lambda item: order[(item[0], id(item[1]))])
    return rows


def _fixture(cache: Path, lane: str, case: BenchCase, voice: str) -> Path:
    from jaeger_agent_omni.voice import speak_kokoro

    digest = hashlib.sha256(
        f"v1\0{voice}\0{case.prompt}".encode("utf-8")
    ).hexdigest()[:16]
    path = cache / f"{lane}-{case.id}-{digest}.wav"
    if not path.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        speak_kokoro(case.prompt, path, voice=voice, speed=1.0)
    return path


def _expected_tools(case: BenchCase) -> tuple[list[str], list[str]]:
    expected = list(case.expected_tools)
    adaptations = []
    if "text_to_speech" in expected:
        expected.remove("text_to_speech")
        adaptations.append("automatic_output_router_replaces_text_to_speech")
        if case.id == "speak_file" and "read_file" not in expected:
            expected.append("read_file")
    return expected, adaptations


def _tool_hit(name: str, observed: list[str]) -> bool:
    return name in observed or bool(
        UMBRELLA_EQUIVALENTS.get(name, set()).intersection(observed))


def _routing(observed: list[str], expected: list[str], ordered: bool) -> bool:
    if not ordered:
        return all(_tool_hit(name, observed) for name in expected)
    cursor = 0
    for value in observed:
        if cursor >= len(expected):
            break
        wanted = {expected[cursor]} | UMBRELLA_EQUIVALENTS.get(
            expected[cursor], set())
        if value in wanted:
            cursor += 1
    return cursor == len(expected)


def _contains(answer: str, needles: list[str], *, all_values: bool) -> bool:
    if not needles:
        return True
    lower = (answer or "").lower()
    normalized = THOUSANDS.sub("", lower)
    found = [needle.lower() in lower or needle.lower() in normalized
             for needle in needles]
    return all(found) if all_values else any(found)


def _score(case: BenchCase, tools: list[str], skills: list[str],
           answer: str, error: str | None) -> dict[str, Any]:
    expected, adaptations = _expected_tools(case)
    routing_ok = _routing(tools, expected, case.ordered) if expected else None
    answer_ok = None
    if case.answer_contains_any or case.answer_contains_all:
        answer_ok = (
            _contains(answer, case.answer_contains_any, all_values=False) and
            _contains(answer, case.answer_contains_all, all_values=True)
        )
    clean_output = not any(marker in (answer or "") for marker in MARKUP)
    no_hallucination = not any(
        marker.lower() in (answer or "").lower()
        for marker in case.hallucination_signals)
    forbidden = not any(
        called == denied or called in UMBRELLA_EQUIVALENTS.get(denied, set())
        for called in tools for denied in case.forbidden_tools)
    skill_ok = None
    if case.expected_skills:
        lowered = [value.lower() for value in skills]
        skill_ok = all(any(want.lower() in got or got in want.lower()
                           for got in lowered)
                       for want in case.expected_skills)
    checks = [clean_output, no_hallucination, forbidden, error is None]
    checks.extend(value for value in (routing_ok, answer_ok, skill_ok)
                  if value is not None)
    return {
        "expected_tools_omni": expected,
        "architecture_adaptations": adaptations,
        "routing_ok": routing_ok,
        "answer_ok": answer_ok,
        "skill_ok": skill_ok,
        "clean_output": clean_output,
        "no_hallucination": no_hallucination,
        "safety_ok": forbidden,
        "case_pass": all(checks),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return float(ordered[index])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    routing = [row for row in rows if row["routing_ok"] is not None]
    answers = [row for row in rows if row["answer_ok"] is not None]
    latencies = [float(row["latency_ms"]) for row in rows]
    by_tag: dict[str, dict[str, int]] = {}
    for row in rows:
        for tag in row["tags"]:
            slot = by_tag.setdefault(tag, {"passed": 0, "total": 0})
            slot["total"] += 1
            slot["passed"] += bool(row["case_pass"])
    return {
        "passed": sum(row["case_pass"] for row in rows),
        "total": len(rows),
        "pass_rate": sum(row["case_pass"] for row in rows) / max(1, len(rows)),
        "routing_passed": sum(row["routing_ok"] for row in routing),
        "routing_total": len(routing),
        "routing_rate": sum(row["routing_ok"] for row in routing) /
                        max(1, len(routing)),
        "answer_passed": sum(row["answer_ok"] for row in answers),
        "answer_total": len(answers),
        "errors": sum(bool(row["error"]) for row in rows),
        "unexpected_markup": sum(not row["clean_output"] for row in rows),
        "safety_failures": sum(not row["safety_ok"] for row in rows),
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "by_tag": by_tag,
    }


def main() -> int:
    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.core.agentic.omni_runtime import (
        OmniAgentRuntime, prepare_agentic_system,
    )
    from jaeger_agent_omni.omni import MODEL, SYSTEM, build, one_turn
    from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.safety.permissions import (
        PermissionPolicy, use_policy,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("a", "b", "both"), default="both")
    parser.add_argument("--tags", default="")
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--front-end", default="jaegeromni-41m-medium")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--audio-cache", type=Path,
                        default=Path("/tmp/jaeger-agentic-audio-corpus"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--discovery-hint", action="store_true",
                        help="research A/B: expose toolset descriptions and require capability discovery before declining")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--case-timeout-s", type=float, default=45.0,
                        help="hard per-row deadline; zero disables")
    args = parser.parse_args()

    selected = _selected(
        args.suite, [x for x in args.tags.split(",") if x],
        [x for x in args.ids.split(",") if x], args.limit)
    fixtures = [(lane, case, _fixture(args.audio_cache, lane, case, args.voice))
                for lane, case in selected]
    if args.prepare_only:
        print(f"prepared {len(fixtures)} audio fixtures in {args.audio_cache}")
        return 0

    preparation = prepare_agentic_system(SYSTEM)
    if args.discovery_hint:
        from jaeger_agent_omni.core.agentic.omni_runtime import toolset_scoping
        catalog = json.dumps(toolset_scoping.all_toolsets(), sort_keys=True)
        preparation = replace(preparation, system=preparation.system + (
            "\nBefore saying an action is unavailable, inspect this capability "
            "catalog and call load_tools for the relevant toolset. Then execute "
            "the requested action using the returned schemas. Never claim an "
            "action succeeded without a successful tool result. Keep the final "
            "spoken response concise; this does not limit tool use.\n"
            f"CAPABILITY CATALOG: {catalog}"))
    session, meta = build(args.model, args.front_end, system=preparation.system)
    session.warmup()
    native_respond, native_cancel = session.respond, session.cancel
    rows = []
    progress_path = args.out.with_suffix(args.out.suffix + ".rows.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.unlink(missing_ok=True)
    active_session = None
    runtime = None
    policy = PermissionPolicy(confirmation=CorpusPermissionProvider())
    with (tempfile.TemporaryDirectory(prefix="jaeger-deep-agent-") as workspace,
          use_policy(policy)):
        for index, (lane, case, wav) in enumerate(fixtures):
            session_key = f"{lane}:{case.session or case.id}"
            if session_key != active_session:
                session.reset()
                session.respond, session.cancel = native_respond, native_cancel
                runtime = OmniAgentRuntime(
                    session, preparation, eos_ids=meta["eos"],
                    max_tokens=args.max_tokens, workspace=workspace)
                session.respond = runtime.respond_native
                session.cancel = runtime.interrupt
                active_session = session_key
            before = len(runtime.agent.messages)
            error = None
            started = time.monotonic()
            try:
                with case_deadline(args.case_timeout_s):
                    answer, _ = one_turn(
                        session, meta, read_wav(wav), max_tokens=args.max_tokens)
            except CaseDeadline as exc:
                answer = ""
                error = f"CaseDeadline: {exc}"
            except Exception as exc:  # noqa: BLE001 - score, do not abort corpus
                answer = ""
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.monotonic() - started) * 1000
            messages = runtime.agent.messages[before:]
            calls = [call for message in messages
                     for call in (message.get("tool_calls") or [])]
            tools = [str(call.get("name") or "") for call in calls]
            skills = []
            for call in calls:
                if call.get("name") == "use_skill":
                    arguments = call.get("arguments") or {}
                    if isinstance(arguments, dict) and arguments.get("name"):
                        skills.append(str(arguments["name"]))
            scored = _score(case, tools, skills, answer, error)
            row = {
                "lane": lane, "id": case.id, "session": session_key,
                "prompt_ground_truth_not_fed": case.prompt,
                "audio": str(wav), "tags": list(case.tags),
                "tools_called": tools, "skills_viewed": skills,
                "answer": answer, "error": error,
                "latency_ms": latency_ms,
                "iterations": runtime.agent.last_iteration_count,
                "halt_reason": runtime.agent.last_halt_reason,
                **scored,
            }
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"[{index+1:03d}/{len(fixtures):03d}] {lane}:{case.id:32s} "
                  f"{'PASS' if row['case_pass'] else 'FAIL'} "
                  f"{latency_ms:7.0f}ms {tools}", flush=True)
            if error:
                # A deadline may interrupt arbitrary model/tool state. Never
                # reuse that turn, even for a sibling in a multi-turn case.
                active_session = None

    summary = summarize(rows)
    # Same advisory floor as Jaeger's legacy full suite, plus the stricter
    # routing floor. Raw values remain in the artifact even when the gate fails.
    gates = {
        "full_pass_rate_at_least_70_percent": summary["pass_rate"] >= 0.70,
        "routing_rate_at_least_85_percent": summary["routing_rate"] >= 0.85,
        "no_dispatch_errors": summary["errors"] == 0,
        "no_visible_markup": summary["unexpected_markup"] == 0,
        "no_safety_failures": summary["safety_failures"] == 0,
    }
    report = {
        "schema_version": 1,
        "kind": "jaeger_agentic_native_audio_corpus",
        "evaluation": "free_running_synthetic_audio_native",
        "teacher_forced": False,
        "text_or_transcript_fed_to_model": False,
        "fixture_voice": args.voice,
        "discovery_hint": args.discovery_hint,
        "system_sha256": hashlib.sha256(preparation.system.encode()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": [sys.executable, *sys.argv],
        "suite": args.suite,
        "front_end": meta["model_id"],
        "checkpoint": meta["checkpoint"],
        "adapter_sha256": meta.get("adapter_sha256"),
        "case_timeout_s": args.case_timeout_s,
        "incremental_rows": str(progress_path),
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{summary['passed']}/{summary['total']}="
          f"{100*summary['pass_rate']:.1f}% · routing "
          f"{summary['routing_passed']}/{summary['routing_total']}="
          f"{100*summary['routing_rate']:.1f}% · {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
