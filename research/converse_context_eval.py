#!/usr/bin/env python3
"""Speaker-sensitive context A/B with oracle speaker IDs and real audio/Gemma.

Tests context utilization separately from speaker-ID accuracy. Current route
injects the current speaker but drops previous speaker IDs from memory; a third
arm measures retaining that same text context in history. No runtime changes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jaeger_agent_omni import converse

SCENARIOS = [
    ("fruit", [(1, "My favorite fruit is mango."), (2, "My favorite fruit is peach.")],
     "What is my favorite fruit?", {1: "mango", 2: "peach"}),
    ("destination", [(2, "My travel destination is Denver."),
                     (1, "My travel destination is Boston.")],
     "What is my travel destination?", {1: "boston", 2: "denver"}),
    ("pet", [(1, "My pet is a rabbit."), (2, "My pet is a turtle.")],
     "What kind of pet do I have?", {1: "rabbit", 2: "turtle"}),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speaker-labels", choices=("oracle", "ecapa"), default="oracle")
    args = p.parse_args()
    from training.context_distill import refuse_gpu_contention
    refuse_gpu_contention()
    from kokoro import KPipeline
    import torch
    torch.set_num_threads(2)
    pipe = KPipeline(lang_code="a", device="cpu")
    cache = REPO / "jaeger_agent_omni/tests/simcache"
    cache.mkdir(exist_ok=True)

    def audio(speaker, text):
        voice = "am_adam" if speaker == 1 else "bm_george"
        key = hashlib.sha256((voice+text).encode()).hexdigest()[:20]
        path = cache / f"context-{key}.npy"
        if not path.exists():
            pieces = [np.asarray(r.audio, np.float32) for r in pipe(text, voice=voice)
                      if r.audio is not None]
            np.save(path, resample_poly(np.concatenate(pieces), 2, 3).astype(np.float32))
        return np.load(path)

    ear = converse.load_trained_semantic("half")
    speaker_node = converse.load_node("para-speaker-embed-v0")
    speaker_node.counts = []
    ecapa = converse.load_node("para-speaker-ecapa-v1") if args.speaker_labels == "ecapa" else None
    if ecapa:
        ecapa.load()
    rows = []
    for arm in ("off", "current_only", "current_and_history"):
        for name, facts, question, expected in SCENARIOS:
            for target in (1, 2):
                history = []
                if ecapa:
                    ecapa._reg = type(ecapa._reg)()
                for index, (sid, words) in enumerate([*facts, (target, question)]):
                    waveform = audio(sid, words)
                    label = ecapa.label_turn(waveform) if ecapa else {"speaker": sid, "conf": 1.0}
                    ctx = speaker_node.context(label)
                    system = converse.NN_SYSTEM
                    if arm != "off" and ctx:
                        system += f"\n\n[context: {ctx}]"
                    started = time.monotonic()
                    raw, stats = ear.respond(waveform, system=system,
                                             history=history, max_tokens=140)
                    heard, reply = converse.parse_heard_reply(raw)
                    history_words = heard
                    if arm == "current_and_history" and ctx:
                        history_words = f"[context: {ctx}]\n{heard}"
                    history.extend([{"role": "user", "content": history_words},
                                    {"role": "assistant", "content": reply}])
                    row = {"arm": arm, "scenario": name, "speaker": sid,
                           "target_speaker": target, "turn": index,
                           "perception_label": label, "context": ctx,
                           "source": words, "heard": heard, "reply": reply,
                           "elapsed_s": time.monotonic()-started, "metrics": stats}
                    if index == len(facts):
                        other = expected[3-target]
                        row.update(expected=expected[target],
                                   passed=expected[target] in reply.lower()
                                   and other not in reply.lower())
                        print(json.dumps(row), flush=True)
                    rows.append(row)
    summary = {arm: {"passed": sum(r["passed"] for r in rows
                                   if r["arm"] == arm and "passed" in r), "total": 6}
               for arm in ("off", "current_only", "current_and_history")}
    report = {"command": [sys.executable, *sys.argv], "summary": summary, "rows": rows,
              "speaker_labels": args.speaker_labels,
              "scope": "12 synthetic two-voice fixtures; oracle utilization or real ECAPA turn labels as selected",
              "node": converse.semantic_node_id("half"),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runtime_changed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
