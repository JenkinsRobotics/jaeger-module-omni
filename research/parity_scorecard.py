#!/usr/bin/env python3
"""Generate a checkpoint-safe Jaeger versus MiniCPM parity scorecard.

The scorecard never combines historical best values from different weights.
Every model row is pinned to one SHA and every runtime result must carry the
same model ID and checkpoint prefix.  Missing or incomparable measurements
stay visibly unmeasured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "research/parity_registry.json"
DEFAULT_JSON = ROOT / "research/results/parity_scorecard.json"
DEFAULT_MARKDOWN = ROOT / "PARITY_SCORECARD.md"


def read_json(relative: str | None) -> dict | None:
    if relative is None:
        return None
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def checkpoint_of(payload: dict | None) -> str | None:
    if payload is None:
        return None
    return payload.get("adapter_sha256")


def verify_checkpoint(expected: str, payload: dict | None,
                      source: str | None) -> None:
    actual = checkpoint_of(payload)
    if expected == "built-in" or payload is None:
        return
    if actual != expected:
        raise ValueError(
            f"checkpoint mismatch in {source}: expected {expected}, got {actual}")


def compare(value: float | None, target: float, direction: str,
            tolerance: float) -> str:
    """Return surpassed/parity/behind without promoting measurement noise."""
    if value is None:
        return "unmeasured"
    if direction == "higher":
        if value > target * (1 + tolerance):
            return "surpassed"
        if value >= target * (1 - tolerance):
            return "parity"
        return "behind"
    if direction == "lower":
        if value < target * (1 - tolerance):
            return "surpassed"
        if value <= target * (1 + tolerance):
            return "parity"
        return "behind"
    raise ValueError(f"unknown direction: {direction}")


def latest_runtime_report(directory: str, model_id: str,
                          checkpoint: str) -> tuple[dict | None, str | None]:
    root = ROOT / directory
    candidates = []
    if root.is_dir():
        for path in root.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
                model = payload.get("model", {})
                report_checkpoint = str(model.get("checkpoint", ""))
                checkpoint_match = (
                    checkpoint == "built-in" and report_checkpoint == "built-in"
                ) or (
                    bool(report_checkpoint) and
                    checkpoint.startswith(report_checkpoint)
                )
                if (payload.get("kind") == "jaeger_duplex_run" and
                        model.get("id") == model_id and checkpoint_match):
                    candidates.append((path.stat().st_mtime, path, payload))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if not candidates:
        return None, None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return payload, str(path.relative_to(ROOT))


def load_model(entry: dict, reference: dict, tolerance: float,
               report_directory: str) -> dict:
    cases = read_json(entry["cases12"])
    clean = read_json(entry["devclean"])
    ami = read_json(entry["ami"])
    testclean = read_json(entry.get("testclean"))
    expected = entry["checkpoint"]
    for payload, source in ((cases, entry["cases12"]),
                            (clean, entry["devclean"]),
                            (ami, entry["ami"]),
                            (testclean, entry.get("testclean"))):
        verify_checkpoint(expected, payload, source)
    manifest = read_json(entry["node_manifest"])
    if manifest is not None:
        manifest_sha = manifest["adapter"]["sha256"]
        if manifest_sha != expected:
            raise ValueError(
                f"node manifest {entry['node_manifest']} is {manifest_sha}, "
                f"not {expected}")

    correct = int(cases["correct"]) if cases else None
    count = int(cases["n"]) if cases else None
    accuracy = correct / count if cases else None
    answer_ms = float(cases["median_ms"]) if cases else None
    runtime, runtime_source = latest_runtime_report(
        report_directory, entry["id"], expected)
    runtime_metrics = runtime.get("metrics", {}) if runtime else {}
    end_audio = runtime_metrics.get("end_to_audio", {})
    first_audio = end_audio.get("median_ms")
    fed = runtime_metrics.get("fed_seconds")
    dropped = runtime_metrics.get("dropped_seconds")
    errors = runtime.get("errors", []) if runtime else []
    five_minute = "unmeasured"
    # A direct-WAV soak validates pacing/drop/latency mechanics but cannot
    # validate the physical-room AEC path.  Do not turn that useful partial
    # evidence into a false physical-room failure.
    if runtime is not None and runtime.get("phase") == "physical-room":
        five_minute = "pass" if (
            fed is not None and fed >= 300.0 and
            dropped == 0 and not errors and
            runtime_metrics.get("context_rolls", 0) >= 1
        ) else "fail"

    ref_metrics = reference["metrics"]
    answer_status = compare(
        answer_ms, float(ref_metrics["answer_latency_ms"]), "lower", tolerance)
    accuracy_status = compare(
        accuracy,
        ref_metrics["cases12_correct"] / ref_metrics["cases12_n"],
        "higher", tolerance)
    paired_first_audio = bool(
        runtime and runtime.get("configuration", {}).get(
            "minicpm_paired_harness"))
    first_audio_status = (
        compare(first_audio, float(ref_metrics["first_audio_ms"]),
                "lower", tolerance)
        if paired_first_audio else
        ("not-comparable" if first_audio is not None else "unmeasured")
    )
    core_ok = accuracy_status in ("parity", "surpassed") and \
        answer_status in ("parity", "surpassed")
    if accuracy_status == "unmeasured" and testclean is not None:
        core_status = "broad corpus winner; answer unmeasured"
    elif not core_ok:
        core_status = "behind"
    elif answer_status == "surpassed":
        core_status = "parity accuracy + faster"
    else:
        core_status = "core ear parity"
    return {
        "id": entry["id"], "name": entry["name"],
        "checkpoint": expected,
        "cases12": {"correct": correct, "n": count,
                    "accuracy": accuracy, "status": accuracy_status,
                    "source": entry["cases12"]},
        "answer_latency_ms": {"value": answer_ms, "status": answer_status,
                              "source": entry["cases12"]},
        "devclean_wer": ({"value": float(clean["wer"]),
                           "n": int(clean["n"]),
                           "source": entry["devclean"]}
                          if clean else None),
        "ami_wer": ({"value": float(ami["wer"]), "n": int(ami["n"]),
                      "source": entry["ami"]} if ami else None),
        "testclean_wer": ({"value": float(testclean["wer"]),
                            "n": int(testclean["n"]),
                            "words": int(testclean["reference_words"]),
                            "source": entry["testclean"]}
                           if testclean else None),
        "first_audio_ms": {"value": first_audio,
                           "status": first_audio_status,
                           "source": runtime_source},
        "five_minute_physical": {"status": five_minute,
                                 "source": runtime_source},
        "core_ear_status": core_status,
    }


def status_label(status: str) -> str:
    return {
        "surpassed": "SURPASSED",
        "parity": "PARITY",
        "behind": "BEHIND",
        "unmeasured": "UNMEASURED",
        "parity accuracy + faster": "PARITY + FASTER",
        "core ear parity": "CORE PARITY",
        "broad corpus winner; answer unmeasured":
            "BROAD CORPUS WINNER; ANSWER UNMEASURED",
        "validated": "VALIDATED",
        "not-comparable": "NOT COMPARABLE",
        "pass": "PASS",
        "fail": "FAIL",
    }.get(status, status.upper())


def percent(metric: dict | None) -> str:
    return "—" if metric is None else f"{100*metric['value']:.2f}%"


def milliseconds(metric: dict) -> str:
    value = metric["value"]
    return "—" if value is None else f"{value:.0f} ms"


def ratio(metric: dict) -> str:
    if metric["correct"] is None or metric["n"] is None:
        return "—"
    return f"{metric['correct']}/{metric['n']}"


def build_markdown(scorecard: dict) -> str:
    reference = scorecard["reference"]
    lines = [
        "# MiniCPM parity scorecard", "",
        "Generated from checkpoint-pinned artifacts. A two-percent tolerance "
        "prevents timing noise from being called a win. WER columns are "
        "Jaeger quality gates because no same-corpus MiniCPM WER baseline is "
        "saved yet. First-audio results are comparable only when a run is "
        "explicitly marked as a paired MiniCPM harness.", "",
        "## Reference", "",
        "| System | cases12 | answer latency | first audio | memory | vision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (f"| {reference['name']} | "
         f"{reference['metrics']['cases12_correct']}/"
         f"{reference['metrics']['cases12_n']} | "
         f"{reference['metrics']['answer_latency_ms']:.0f} ms | "
         f"{reference['metrics']['first_audio_ms']:.0f} ms | "
         f"{reference['metrics']['memory_gb']:.1f} GB | "
         f"{reference['metrics']['vision_ms_per_frame']:.0f} ms/frame |"),
        "", "## Audio-ear comparison", "",
        "| Model | checkpoint | cases12 | vs reference | answer latency | "
        "vs reference | first audio | vs reference | clean WER | AMI WER | "
        "test-clean lockbox | physical 5 min | conclusion |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | "
        "---: | --- | --- |",
    ]
    for model in scorecard["models"]:
        lines.append(
            f"| {model['name']} | `{model['checkpoint'][:12]}` | "
            f"{ratio(model['cases12'])} | "
            f"{status_label(model['cases12']['status'])} | "
            f"{milliseconds(model['answer_latency_ms'])} | "
            f"{status_label(model['answer_latency_ms']['status'])} | "
            f"{milliseconds(model['first_audio_ms'])} | "
            f"{status_label(model['first_audio_ms']['status'])} | "
            f"{percent(model['devclean_wer'])} | "
            f"{percent(model['ami_wer'])} | "
            f"{percent(model['testclean_wer'])} | "
            f"{status_label(model['five_minute_physical']['status'])} | "
            f"{status_label(model['core_ear_status'])} |")
    system = scorecard["system"]
    lines.extend([
        "", "## System-level parity", "",
        "| Gate | Jaeger now | MiniCPM reference | State |",
        "| --- | --- | --- | --- |",
        f"| AEC/double-talk | {system['aec']['summary']} | no comparable saved "
        f"baseline | {status_label(system['aec']['status'])} |",
        f"| Speaker identity/overlap | {system['speaker']['summary']} | no "
        f"comparable saved baseline | {status_label(system['speaker']['status'])} |",
        "| Speech output | decoded text → resident Kokoro | native speech "
        "generation | BEHIND |",
        "| Vision | not connected in Jaeger GUI | 379 ms/frame | BEHIND |",
        "| Physical five-minute matrix | no checkpoint-matched JSON report "
        "yet | complete reference system | UNMEASURED |",
        "| Memory | no checkpoint-matched full-stack measurement | 4.9 GB | "
        "UNMEASURED |",
        "", "## Current verdict", "",
        scorecard["verdict"], "",
        "Jaeger may be called *surpassed* only after one checkpoint retains "
        "cases12 parity, is faster on the same harness, passes the locked "
        "clean/meeting gates, and passes repeated physical five-minute "
        "duplex runs. Full Omni superiority additionally requires native "
        "expressive speech and connected vision.", "",
        "Regenerate with:", "", "```bash",
        "./.venv/bin/python research/parity_scorecard.py", "```", "",
    ])
    return "\n".join(lines)


def generate(registry_path: Path) -> dict:
    registry = json.loads(registry_path.read_text())
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported parity registry schema")
    reference = registry["reference"]
    tolerance = float(registry["comparison_tolerance_fraction"])
    report_directory = registry["system_evidence"]["duplex_reports"]
    models = [load_model(entry, reference, tolerance, report_directory)
              for entry in registry["models"]]
    aec_stress = read_json(registry["system_evidence"]["aec_stress"])
    speaker = read_json(registry["system_evidence"]["speaker_overlap"])
    aec_passed = bool(aec_stress and aec_stress.get("passed"))
    speaker_passed = bool(
        speaker and speaker.get("passed_turn_level_identity") and
        speaker.get("word_level_overlap_attribution_supported"))
    leading = next(model for model in models
                   if model["id"] == "gemmaomni-41m-medium")
    jaeger = next(model for model in models
                  if model["id"] == "jaegeromni-41m-medium")
    causal = next(model for model in models
                  if model["id"] == "jaegeromni-41m-causal-balanced")
    verdict = (
        f"{leading['name']} is the current core-ear leader: "
        f"{leading['cases12']['correct']}/{leading['cases12']['n']} matches "
        f"MiniCPM and {leading['answer_latency_ms']['value']:.0f} ms is faster "
        f"than the {reference['metrics']['answer_latency_ms']:.0f} ms "
        "reference. JaegerOmni 41M independently matches cases12 at "
        f"{jaeger['cases12']['correct']}/{jaeger['cases12']['n']}, but its "
        f"{jaeger['answer_latency_ms']['value']:.0f} ms answer latency is "
        "behind the reference. The new causal-balanced Jaeger checkpoint "
        f"retains {causal['cases12']['correct']}/{causal['cases12']['n']} at "
        f"{causal['answer_latency_ms']['value']:.0f} ms and reduces AMI to "
        f"{100*causal['ami_wer']['value']:.2f}%, but its perception heads "
        "still require post-blend calibration. The frozen Jaeger 41M remains "
        "the broad lockbox leader at "
        f"{100*jaeger['testclean_wer']['value']:.2f}% free-running WER and "
        "has a zero-drop direct five-minute context-roll run. "
        "Jaeger has not surpassed MiniCPM as a complete system: "
        "physical five-minute reports are absent, speaker separation is "
        "failed, the broad AEC stress gate fails, speech still crosses "
        "decoded text into Kokoro, and Jaeger vision is not connected.")
    return {
        "schema_version": 1,
        "reference": reference,
        "models": models,
        "system": {
            "aec": {
                "status": "validated" if aec_passed else "behind",
                "summary": (
                    f"{aec_stress['echo_attenuation_median_db']:.1f} dB "
                    "room-stress attenuation, "
                    f"{aec_stress['reference_aware']['false_barges']}/"
                    f"{aec_stress['cases']} false, "
                    f"{aec_stress['reference_aware']['recall_percent']:.0f}% "
                    "recall, "
                    f"{aec_stress['reference_aware']['p95_delay_ms']:.0f} ms p95"
                    if aec_stress else "unmeasured"),
                "source": registry["system_evidence"]["aec_stress"],
            },
            "speaker": {
                "status": "validated" if speaker_passed else "behind",
                "summary": (
                    f"{100*speaker['continuity']['continuity_accuracy']:.0f}% "
                    f"continuity; separation "
                    f"{'pass' if speaker['continuity']['separated_identities'] else 'fail'}; "
                    "word attribution unavailable"
                    if speaker else "unmeasured"),
                "source": registry["system_evidence"]["speaker_overlap"],
            },
        },
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true",
                        help="validate and print without writing")
    args = parser.parse_args()
    scorecard = generate(args.registry.resolve())
    markdown = build_markdown(scorecard)
    if args.check:
        print(markdown)
        return 0
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(scorecard, indent=2) + "\n")
    args.markdown.write_text(markdown)
    print(scorecard["verdict"])
    print(f"json: {args.json}")
    print(f"markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
