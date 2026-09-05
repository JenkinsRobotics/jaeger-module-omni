#!/usr/bin/env python3
"""Matched cases12 answers at the one-generation converse HEARD/REPLY seam."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jaeger_agent_omni import converse
from jaeger_agent_omni.app import read_wav
from grading import graded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=ROOT/"research/cases12")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from training.context_distill import refuse_gpu_contention
    refuse_gpu_contention()
    spec = importlib.util.spec_from_file_location("answer_ear", args.node/"runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ear = getattr(module, "build_converse", module.build)()
    cases = json.loads((args.cases/"cases.json").read_text())
    rows = []
    for case in cases:
        path = args.cases/f"{case['key']}.wav"
        start = time.monotonic()
        raw, metrics = ear.respond(read_wav(path), system=converse.NN_SYSTEM,
                                   history=[], max_tokens=140)
        heard, reply = converse.parse_heard_reply(raw)
        row = {"case": case["key"], "heard": heard, "answer": reply,
               "ok": graded(reply, case["answers"]), "raw": raw,
               "latency_ms": 1000*(time.monotonic()-start), "metrics": metrics,
               "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        rows.append(row); print(json.dumps(row), flush=True)
    report = {"command": [sys.executable, *sys.argv], "node": args.node.name,
              "correct": sum(r["ok"] for r in rows), "total": len(rows), "n": len(rows), "rows": rows,
              "median_ms": statistics.median(r["latency_ms"] for r in rows),
              "scorer_sha256": hashlib.sha256((ROOT/"research/grading.py").read_bytes()).hexdigest(),
              "cases_sha256": hashlib.sha256((args.cases/"cases.json").read_bytes()).hexdigest(),
              "adapter_sha256": hashlib.sha256((args.node/"weights/adapter.npz").read_bytes()).hexdigest(),
              "evaluation_mode": "converse_heard_reply_fresh_history_no_context"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"{report['correct']}/{report['total']}", flush=True)
    return 0 if report["correct"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
