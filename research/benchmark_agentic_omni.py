"""Free-running native-audio regression gate with the agent loop enabled.

This deliberately reuses ``research/cases12`` and its grader so the only
experimental variable is the agentic system prompt and loop. No transcript is
decoded or fed to Gemma. Ordinary factual questions should remain ordinary
answers and should not spuriously dispatch tools.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from grading import graded
    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.core.agentic.omni_runtime import (
        OmniAgentRuntime,
        omni_tools,
        prepare_agentic_system,
    )
    from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.tools.tool_registry import (
        get_tools,
    )
    from jaeger_agent_omni.core.skill_registry.playbook_skills import (
        available_playbooks,
    )
    from jaeger_agent_omni.omni import MODEL, SYSTEM, build, one_turn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "research/cases12")
    parser.add_argument("--front-end", default="jaegeromni-41m-medium")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--ids", nargs="*", default=(),
        help="optional case keys for a focused diagnostic run",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "research/results/agentic-omni-cases12.json",
    )
    args = parser.parse_args()

    cases = json.loads((args.cases / "cases.json").read_text())
    if args.ids:
        wanted = set(args.ids)
        cases = [case for case in cases if case["key"] in wanted]
    preparation = prepare_agentic_system(SYSTEM)
    session, meta = build(
        args.model, args.front_end, system=preparation.system,
    )
    session.warmup()

    rows = []
    with tempfile.TemporaryDirectory(prefix="jaeger-agentic-bench-") as workspace:
        runtime = OmniAgentRuntime(
            session,
            preparation,
            eos_ids=meta["eos"],
            max_tokens=args.max_tokens,
            workspace=workspace,
        )
        session.respond = runtime.respond_native
        session.cancel = runtime.interrupt

        for case in cases:
            before = len(runtime.agent.messages)
            answer, latency_ms = one_turn(
                session,
                meta,
                read_wav(args.cases / f"{case['key']}.wav"),
                max_tokens=args.max_tokens,
            )
            turn_messages = runtime.agent.messages[before:]
            tool_calls = [
                call.get("name", "")
                for message in turn_messages
                for call in (message.get("tool_calls") or [])
            ]
            ok = graded(answer, case["answers"])
            rows.append({
                "case": case["key"],
                "ok": ok,
                "latency_ms": latency_ms,
                "answer": answer,
                "tool_calls": tool_calls,
                "iterations": runtime.agent.last_iteration_count,
                "halt_reason": runtime.agent.last_halt_reason,
                "generated_tokens": len(getattr(session, "output_ids", ())),
                "raw_response": runtime.adapter.last_raw_response,
            })
            print(
                f"  {case['key']} {'OK ' if ok else 'MISS'} "
                f"{latency_ms:7.0f}ms iter={runtime.agent.last_iteration_count} "
                f"tok={len(getattr(session, 'output_ids', ()))} "
                f"tools={tool_calls} {answer[:70]!r}",
                flush=True,
            )

    correct = sum(row["ok"] for row in rows)
    allowed_tools = {
        "q00": {"calculate"},
        "q04": {"calculate"},
        "q07": {"calculate"},
    }
    unexpected = [
        row["case"] for row in rows
        if any(name not in allowed_tools.get(row["case"], set())
               for name in row["tool_calls"])
    ]
    payload = {
        "kind": "agentic_omni_native_audio_regression",
        "front_end": meta["model_id"],
        "checkpoint": meta["checkpoint"],
        "adapter_sha256": meta.get("adapter_sha256"),
        "correct": correct,
        "n": len(rows),
        "median_ms": statistics.median(row["latency_ms"] for row in rows),
        "unexpected_tool_cases": unexpected,
        "allowed_tool_cases": sorted(allowed_tools),
        "registered_tools": len(get_tools()),
        "omni_safe_tools": len(omni_tools(visible_only=False)),
        "initial_visible_tools": len(preparation.initial_tools),
        "skills": len(available_playbooks()),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"\nagentic omni {correct}/{len(rows)}; median "
        f"{payload['median_ms']:.0f} ms; unexpected tools {len(unexpected)}",
    )
    print(f"wrote {args.out}")
    return 0 if correct == len(rows) and not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
