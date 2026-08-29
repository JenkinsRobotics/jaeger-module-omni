#!/usr/bin/env python3
"""Free-running corpus gate for always-on companion admission.

The manifest supplies audio and only the expected action; no transcript is
decoded or passed into Gemma.  Rows sharing ``session_id`` run in order with a
persistent cache so follow-up behavior is measured rather than simulated.

Example JSONL row::

  {"audio":"direct.wav","expected":"respond","session_id":"direct-1"}
  {"audio":"followup.wav","expected":"respond","session_id":"direct-1"}
  {"audio":"overheard.wav","expected":"listen","category":"overheard"}

Run this only when the one accelerator slot is assigned::

  python research/companion_gate.py --manifest research/companion_gate.jsonl \
    --front-end jaegeromni-41m-medium --out research/results/companion.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from jaeger_agent_omni.omni import MODEL, build
from jaeger_agent_omni.runtime.companion import (
    CompanionState, LISTEN, RESPOND, parse_decision, system_prompt,
)


RATE = 16_000


def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:
        count = max(1, round(len(audio) * RATE / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, count),
            np.arange(len(audio)), audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        expected = str(row.get("expected", "")).lower()
        if expected not in (RESPOND, LISTEN):
            raise ValueError(
                f"{path}:{line_number}: expected must be respond or listen")
        audio = Path(row["audio"]).expanduser()
        if not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        rows.append({**row, "expected": expected, "audio": str(audio)})
    if not rows:
        raise ValueError(f"{path}: no evaluation rows")
    return rows


def summarize(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(row["expected"] == row["predicted"] for row in results)
    tp = sum(row["expected"] == RESPOND and row["predicted"] == RESPOND
             for row in results)
    fp = sum(row["expected"] == LISTEN and row["predicted"] == RESPOND
             for row in results)
    fn = sum(row["expected"] == RESPOND and row["predicted"] == LISTEN
             for row in results)
    latency = [row["latency_ms"] for row in results]
    by_category = {}
    for row in results:
        category = str(row.get("category", "unspecified"))
        group = by_category.setdefault(category, {"n": 0, "correct": 0})
        group["n"] += 1
        group["correct"] += row["expected"] == row["predicted"]
    return {
        "clips": total,
        "correct": correct,
        "accuracy": correct / total,
        "respond_precision": tp / max(1, tp + fp),
        "respond_recall": tp / max(1, tp + fn),
        "false_responses": fp,
        "missed_addresses": fn,
        "latency_ms": {
            "median": statistics.median(latency),
            "max": max(latency),
        },
        "by_category": by_category,
    }


def evaluate(args) -> dict:
    rows = read_manifest(Path(args.manifest))
    session, meta = build(
        args.model, args.front_end, system=system_prompt())
    session.warmup()
    state = None
    active_session = None
    results = []
    for index, row in enumerate(rows):
        session_id = str(row.get("session_id", f"independent-{index}"))
        if session_id != active_session:
            session.reset()
            state = CompanionState()
            active_session = session_id
        audio = load_audio(Path(row["audio"]))
        state.speech_started()
        session.open_turn()
        for offset in range(0, len(audio), 1600):
            session.feed_audio(audio[offset:offset + 1600])
        session.end_of_turn(close=False)
        session.feed_text(state.turn_metadata(row.get("speaker")))
        session.close_turn()
        started = time.monotonic()
        raw = session.respond(max_tokens=96, eos_ids=meta["eos"])
        latency_ms = (time.monotonic() - started) * 1000
        decision = parse_decision(raw)
        state.observe(decision)
        results.append({
            **row,
            "predicted": RESPOND if decision.should_respond else LISTEN,
            "valid_control": decision.valid,
            "reason": decision.reason,
            "response": decision.response,
            "raw_control_output": raw,
            "latency_ms": latency_ms,
        })
    return {
        "kind": "jaeger_companion_admission_gate",
        "evaluation": "free_running_audio_native",
        "teacher_forced": False,
        "front_end": args.front_end,
        "model": args.model,
        "checkpoint": meta["checkpoint"],
        "manifest": str(Path(args.manifest).resolve()),
        "summary": summarize(results),
        "results": results,
    }


def selfcheck() -> None:
    rows = [
        {"expected": RESPOND, "predicted": RESPOND, "latency_ms": 10,
         "category": "direct"},
        {"expected": LISTEN, "predicted": RESPOND, "latency_ms": 20,
         "category": "overheard"},
        {"expected": LISTEN, "predicted": LISTEN, "latency_ms": 30,
         "category": "noise"},
    ]
    report = summarize(rows)
    assert report["accuracy"] == 2 / 3
    assert report["false_responses"] == 1
    assert report["respond_precision"] == 0.5
    assert report["latency_ms"]["median"] == 20
    print("companion_gate: corpus accuracy, false-response, recall, latency OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest")
    parser.add_argument("--front-end", default="jaegeromni-41m-medium")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="research/results/companion-gate.json")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if not args.manifest:
        parser.error("--manifest is required unless --selfcheck is used")
    report = evaluate(args)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    print(f"{summary['correct']}/{summary['clips']}="
          f"{summary['accuracy']:.1%} · false responses "
          f"{summary['false_responses']} · missed addresses "
          f"{summary['missed_addresses']} · {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
