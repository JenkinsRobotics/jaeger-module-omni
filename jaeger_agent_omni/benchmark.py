"""Run the repository-owned, multi-layer Jaeger Agent Omni benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

from .core.benchmarking import (
    GateSpec, read_json, report_payload, run_gate, write_report,
)
from .core.model_catalog import DEFAULT_FRONT_END
from .omni import MODEL


ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
GPU_PATTERNS = (
    r"training/(?:train|context_|cleanroom_|stock_whisper|precompute|port_)",
    r"research/(?:latency_breakdown|streaming_stability|causal_adapter_check|"
    r"perception_gate|audio_scene_gate|benchmark_agentic_(?:omni|corpus)|duplex_check)",
    r"eval/benchmark_suite\.py|train_audio_adapter\.py",
)


def _json_assessor(check):
    def assess(spec: GateSpec, returncode: int):
        if returncode != 0 and not spec.artifact:
            return False, f"exit {returncode}", {}
        if spec.artifact is None or not spec.artifact.is_file():
            return False, f"missing artifact {spec.artifact}", {}
        payload = read_json(spec.artifact)
        ok, detail, metrics = check(payload)
        return bool(ok), detail, metrics
    return assess


def _plain_audio(payload):
    metrics = {key: payload.get(key) for key in ("correct", "n", "median_ms")}
    ok = payload.get("correct") == payload.get("n") == 12
    return ok, f"{payload.get('correct')}/{payload.get('n')} · " \
        f"{payload.get('median_ms', 0):.0f} ms", metrics


def _agentic_audio(payload):
    unexpected = payload.get("unexpected_tool_cases") or []
    ok = payload.get("correct") == payload.get("n") == 12 and not unexpected
    metrics = {key: payload.get(key) for key in (
        "correct", "n", "median_ms", "unexpected_tool_cases",
        "registered_tools", "omni_safe_tools", "initial_visible_tools")}
    return ok, f"{payload.get('correct')}/{payload.get('n')} · " \
        f"{payload.get('median_ms', 0):.0f} ms · " \
        f"unexpected={len(unexpected)}", metrics


def _source_parity(payload):
    checks = payload.get("checks") or {}
    return payload.get("all_pass") is True, \
        f"{sum(bool(v) for v in checks.values())}/{len(checks)} exact checks", checks


def _duplex(payload):
    cancel = payload.get("cancel_latency_ms")
    ok = (payload.get("passed") is True and payload.get("audio_dropped") == 0
          and payload.get("interrupts", 0) >= 1 and cancel is not None
          and float(cancel) <= 50.0)
    metrics = {key: payload.get(key) for key in (
        "turns", "interrupts", "audio_dropped", "max_lag_ms",
        "cancel_latency_ms", "consumed_seconds")}
    return ok, f"interrupt {cancel if cancel is not None else 'n/a'} ms · " \
        f"drops {payload.get('audio_dropped')}", metrics


def _companion(kind: str):
    def check(payload):
        summary = payload.get("summary") or {}
        if kind == "ambient":
            ok = summary.get("false_responses") == 0
            detail = f"false responses {summary.get('false_responses')}/" \
                     f"{summary.get('clips')}"
        else:
            ok = float(summary.get("respond_recall", 0)) >= 0.90
            detail = f"address recall {100*float(summary.get('respond_recall', 0)):.1f}%"
        return ok, detail, summary
    return check


def _vad(payload):
    summary = payload.get("summary") or {}
    ok = (summary.get("speech_opened") == summary.get("speech_total") and
          summary.get("speech_ended") == summary.get("speech_total") and
          summary.get("false_opens") == 0 and
          int(summary.get("nonspeech_total", 0)) >= 5)
    return ok, f"opened {summary.get('speech_opened')}/" \
        f"{summary.get('speech_total')} · ended {summary.get('speech_ended')}/" \
        f"{summary.get('speech_total')} · false opens " \
        f"{summary.get('false_opens')}/{summary.get('nonspeech_total')}", summary


def _aec(payload):
    gates = payload.get("gates") or {}
    failed = [name for name, ok in gates.items() if not ok]
    measured = payload.get("reference_aware") or {}
    return payload.get("passed") is True, \
        ("all promotion gates" if not failed else "failed: " + ", ".join(failed)), \
        {"gates": gates,
         "false_barges": measured.get("false_barges"),
         "recall_percent": measured.get("recall_percent"),
         "median_delay_ms": measured.get("median_delay_ms"),
         "p95_delay_ms": measured.get("p95_delay_ms")}


def _agentic_corpus(payload):
    summary = payload.get("summary") or {}
    gates = payload.get("gates") or {}
    return payload.get("passed") is True, \
        f"{summary.get('passed')}/{summary.get('total')} · routing " \
        f"{summary.get('routing_passed')}/{summary.get('routing_total')}", \
        {"summary": summary, "gates": gates}


def _kokoro_wav(path: Path):
    def assess(_: GateSpec, returncode: int):
        if returncode != 0 or not path.is_file():
            return False, f"exit {returncode}; wav={path.is_file()}", {}
        with wave.open(str(path)) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
        seconds = frames / max(1, rate)
        ok = rate == 24_000 and channels == 1 and seconds >= 0.25
        return ok, f"{seconds:.2f}s · {rate} Hz · {channels}ch", {
            "seconds": seconds, "sample_rate": rate, "channels": channels,
        }
    return assess


def successful_physical_tool_results(agentic):
    """Positive results for the scripted clock/math/file actions, not dispatch.

    This is a necessary gate. Artifact/claim review still verifies whether the
    successful operation fulfilled the user's actual request.
    """
    successful = []
    file_flags = {'write_file': 'written', 'append_file': 'appended',
                  'read_file': 'read', 'delete_file': 'deleted'}
    for row in agentic.get('tool_results', []):
        try:
            result = json.loads(row.get('content', ''))
        except (TypeError, ValueError):
            continue
        if not isinstance(result, dict) or result.get('error') or result.get('ok') is False:
            continue
        name = row.get('name')
        passed = ((name == 'calculate' and 'result' in result) or
                  (name == 'get_time' and bool(result.get('iso'))) or
                  (name in file_flags and result.get(file_flags[name]) is True))
        if passed:
            successful.append(row.get('tool_call_id') or name)
    return list(dict.fromkeys(successful))


def _physical_payload(payload):
    metrics = payload.get("metrics") or {}
    config = payload.get("configuration") or {}
    end_audio = metrics.get("end_to_audio") or {}
    aec = metrics.get("aec") or {}
    companion = metrics.get("companion") or {}
    agentic = metrics.get("agentic") or {}
    speech = metrics.get('speech_delivery') or {}
    shutdown = payload.get('audio_shutdown') or {}
    checks = {
        "physical room": payload.get("phase") == "physical-room",
        "five minute capture": float(metrics.get("fed_seconds", 0)) >= 295.0,
        "zero loop drops": float(metrics.get("dropped_seconds", 1)) == 0.0,
        "zero mic overflow": int(metrics.get("mic_queue_overflow_blocks", 1)) == 0,
        "neural VAD": (metrics.get("vad") or {}).get("mode") == "neural",
        "full duplex": config.get("duplex_mode") == "full",
        "AEC exercised": (aec.get("available") is True and
                          float(aec.get("seconds_cancelled", 0)) > 0),
        "turn and spoken reply": (int(metrics.get("turns", 0)) >= 1 and
                                  int(metrics.get("completed_replies", 0)) >= 1 and
                                  int(end_audio.get("n", 0)) >= 1 and
                                  end_audio.get("median_ms") is not None),
        "barge-in exercised": (int(metrics.get("barge_events", 0)) >= 1 and
                               (int(metrics.get("barge_stops", 0)) >= 1 or
                                int(metrics.get("interrupts", 0)) >= 1)),
        "context rolled": int(metrics.get("context_rolls", 0)) >= 1,
        "companion both decisions": (companion.get("enabled") is True and
                                     int(companion.get("respond", 0)) >= 1 and
                                     int(companion.get("listen", 0)) >= 1),
        "agent tools exercised": (agentic.get("enabled") is True and
                                  int(agentic.get("tool_calls", 0)) >= 3),
        "successful action tool results": len(successful_physical_tool_results(agentic)) >= 3,
        "uninterrupted spoken reply": speech.get('completed_uninterrupted_replies', 0) >= 1,
        "audio shutdown completed": shutdown.get('completed') is True and shutdown.get('errors') == [],
        "no runtime errors": not payload.get("errors"),
    }
    failed = [name for name, ok in checks.items() if not ok]
    detail = ("all physical gates" if not failed else
              "failed: " + ", ".join(failed))
    return all(checks.values()), detail, {"checks": checks, **metrics}


def _physical(path: Path | None):
    return _json_assessor(_physical_payload) if path else None


def _gpu_conflicts() -> list[str]:
    listing = subprocess.check_output(
        ["ps", "-Ao", "pid=,ppid=,comm=,args="], text=True)
    mine = {os.getpid(), os.getppid()}
    conflicts = []
    for line in listing.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or not fields[0].isdigit() or int(fields[0]) in mine:
            continue
        if "python" not in Path(fields[2]).name.lower():
            continue
        if any(re.search(pattern, fields[3]) for pattern in GPU_PATTERNS):
            conflicts.append(line.strip())
    return conflicts


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, dirty


def build_gates(args, run_dir: Path) -> list[GateSpec]:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    py = str(PYTHON)
    gates = [
        GateSpec("repo_contracts", "cpu", "31 repository module contracts",
                 (py, "research/check_all.py"), timeout_s=90),
        GateSpec("transcript_ban", "cpu", "native input-path AST audit",
                 (py, "-m", "jaeger_agent_omni.omni", "--selfcheck"), 30),
        GateSpec("package_tests", "cpu", "all package regression tests",
                 (py, "-m", "pytest", "-q", "jaeger_agent_omni/tests"), 90),
        GateSpec("agentic_selfcheck", "agent", "19 tool/skill/dialect checks",
                 (py, "-m", "jaeger_agent_omni.core.agentic.selfcheck", "--json"), 90),
        GateSpec("parity_artifacts", "evidence", "checkpoint-pinned MiniCPM scorecard",
                 (py, "research/parity_scorecard.py", "--check"), 30),
        GateSpec("live_assembly", "output", "live AEC/Kokoro/clause wiring",
                 (py, "-m", "jaeger_agent_omni.live", "--demo"), 30),
        GateSpec("voice_contract", "output", "resident voice-node contract",
                 (py, "-m", "jaeger_agent_omni.voice", "--demo"), 30),
        GateSpec("companion_scoring", "companion", "admission grader selfcheck",
                 (py, "research/companion_gate.py", "--selfcheck"), 30),
    ]
    legacy = Path.home() / "GITHUB/jaeger-agent"
    legacy_os = Path.home() / "GITHUB/JaegerOS"
    parity = artifacts / "agentic-source-parity.json"
    if legacy.is_dir() and legacy_os.is_dir():
        gates.append(GateSpec(
            "agentic_source_parity", "agent",
            "pinned 81+81 corpora and 97-tool source parity",
            (py, "research/benchmark_agentic_parity.py", "--out", str(parity)),
            60, artifact=parity, assess=_json_assessor(_source_parity)))
    else:
        gates.append(GateSpec(
            "agentic_source_parity", "agent",
            "pinned 81+81 corpora and 97-tool source parity",
            required=False, skip_reason="read-only legacy repositories absent"))

    if args.profile in ("model", "release"):
        plain = artifacts / "omni-cases12.json"
        agentic = artifacts / "agentic-cases12.json"
        duplex = artifacts / "duplex-barge.json"
        kokoro = artifacts / "kokoro.wav"
        ambient = artifacts / "companion-ambient.json"
        addressed_dir = artifacts / "addressed-audio"
        addressed_manifest = artifacts / "jaeger-companion-addressed.jsonl"
        addressed = artifacts / "companion-addressed.json"
        vad = artifacts / "vad-cases12.json"
        gates.extend((
            GateSpec("omni_cases12", "model", "frozen neural ear answer gate",
                     (py, "-m", "jaeger_agent_omni.omni", "--bench",
                      "research/cases12", "--front-end", args.front_end,
                      "--model", args.model, "--out", str(plain)),
                     300, artifact=plain, assess=_json_assessor(_plain_audio)),
            GateSpec("agentic_cases12", "agent", "native-audio loop + tool dispatch",
                     (py, "research/benchmark_agentic_omni.py", "--front-end",
                      args.front_end, "--model", args.model, "--out", str(agentic)),
                     360, artifact=agentic, assess=_json_assessor(_agentic_audio)),
            GateSpec("duplex_barge", "duplex", "stream integrity and <=50ms cancel",
                     (py, "research/duplex_check.py", "--barge-in", "--seconds", "8",
                      "--front-end", args.front_end, "--model", args.model,
                      "--json", str(duplex)), 180, artifact=duplex,
                     assess=_json_assessor(_duplex)),
            GateSpec("kokoro_pcm", "output", "resident Kokoro neural synthesis",
                     (py, "-m", "jaeger_agent_omni.voice", "--say",
                      "Jaeger neural speech output is ready.", "--out", str(kokoro)),
                     120, artifact=kokoro, assess=_kokoro_wav(kokoro)),
            GateSpec("companion_ambient", "companion", "unaddressed speech suppression",
                     (py, "research/companion_gate.py", "--manifest",
                      "research/companion_ambient_cases12.jsonl", "--front-end",
                      args.front_end, "--model", args.model, "--out", str(ambient)),
                     300, artifact=ambient, assess=_json_assessor(_companion("ambient"))),
            GateSpec("build_addressed_audio", "companion",
                     "fixed Kokoro named-address corpus",
                     (py, "research/make_companion_addressed.py", "--dir",
                      str(addressed_dir)), 180),
            GateSpec("companion_addressed", "companion", "named-address response recall",
                     (py, "research/companion_gate.py", "--manifest",
                      str(addressed_manifest), "--front-end", args.front_end,
                      "--model", args.model, "--out", str(addressed)),
                     300, artifact=addressed,
                     assess=_json_assessor(_companion("addressed"))),
            GateSpec("neural_vad_cases12", "perception", "speech open/end on cases12",
                     (py, "research/vad_gate.py", "--speech-dir", "research/cases12",
                      "--out", str(vad)), 120, artifact=vad,
                     assess=_json_assessor(_vad)),
        ))

    if args.profile == "release":
        aec = artifacts / "aec-stress.json"
        corpus = artifacts / "agentic-native-audio-corpus.json"
        gates.append(GateSpec(
            "agentic_full_corpus", "agent",
            "81+81 routing/multistep/memory/recovery/safety cases as audio",
            (py, "research/benchmark_agentic_corpus.py", "--suite", "both",
             "--front-end", args.front_end, "--model", args.model,
             "--case-timeout-s", "20",
             "--audio-cache", str(Path.home() / "models/jaeger-benchmark-audio"),
             "--out", str(corpus)), 3600, artifact=corpus,
            assess=_json_assessor(_agentic_corpus)))
        gates.append(GateSpec(
            "aec_room_stress", "aec", "43-room echo/barge promotion gate",
            (py, "research/aec_stress.py", "--cases", "43", "--json", str(aec)),
            300, artifact=aec, assess=_json_assessor(_aec)))
        if args.physical_report:
            physical = args.physical_report.resolve()
            gates.append(GateSpec(
                "physical_five_minute", "physical",
                "real mic/AEC/turn/barge/context-roll evidence",
                (py, "-c", "import pathlib,sys;sys.exit(0 if pathlib.Path(sys.argv[1]).is_file() else 1)",
                 str(physical)), 10, artifact=physical,
                assess=_physical(physical)))
        else:
            gates.append(GateSpec(
                "physical_five_minute", "physical",
                "real mic/AEC/turn/barge/context-roll evidence",
                required=True, skip_reason="pass --physical-report from a real 5-minute run"))
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cpu", "model", "release"),
                        default="cpu")
    parser.add_argument("--front-end", default=DEFAULT_FRONT_END)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--physical-report", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (args.out_dir or ROOT / "research/results/deep-benchmark" / stamp).resolve()
    commit, dirty = _git_state()
    results = []
    if args.profile in ("model", "release"):
        conflicts = _gpu_conflicts()
        if conflicts:
            print("REFUSING: another known GPU/model job is active:", file=sys.stderr)
            print("\n".join(f"  {line}" for line in conflicts), file=sys.stderr)
            return 2
    gates = build_gates(args, run_dir)
    for index, gate in enumerate(gates, 1):
        print(f"[{index:02d}/{len(gates):02d}] {gate.id} ...", flush=True)
        result = run_gate(gate, cwd=ROOT, log_dir=run_dir / "logs")
        results.append(result)
        print(f"         {result.status} {result.detail}", flush=True)
        report = report_payload(
            profile=args.profile, root=ROOT, results=results,
            git_commit=commit, dirty=dirty)
        write_report(report, run_dir)
    summary = report["summary"]
    print(f"\n{summary['pass']} pass · {summary['fail']} fail · "
          f"{summary['skip']} skip · {run_dir}")
    if summary["required_failures"]:
        return 1
    if summary["required_skips"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
