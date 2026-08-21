"""Decide the production architecture on evidence, not preference.

Runs the streaming native-Gemma pipeline over a spoken benchmark with
per-case answer keys, and reports the numbers the decision actually turns on:
accuracy, wait time, realtime factor, and where the time goes.

Why the other two candidates are not head-to-head here — both honest, both
recorded so the decision is reviewable:

    MiniCPM-o    its English assistant prompt lives behind `duplex_mode`,
                 which `llama-omni-cli` never exposes (only the server does).
                 The CLI's non-duplex branch hardcodes a Chinese persona and
                 treats the input clip as a VOICE-CLONING REFERENCE, so its
                 "answers" are a self-introduction. Wiring the server is a
                 build task, not a measurement.
    Ultravox     its published model does not load on transformers v5 (v4-era
                 loader dies in tie_weights on meta tensors). Its TRAINED
                 PROJECTOR does load into our modules and is measured
                 separately in research/load_ultravox.py — which is the part
                 that informs our own bridge design.

    python research/final_decision.py --cases <dir>
"""

from __future__ import annotations

import argparse
import re
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from grading import graded                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model",
                    default="/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.utils import load

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.runtime import GemmaAudioPrefiller, StreamingSession

    cases = json.load(open(Path(args.cases) / "cases.json"))
    m, proc = load(args.model)
    cfg = json.load(open(Path(args.model) / "config.json"))
    eos = {proc.tokenizer.eos_token_id}

    def fresh():
        return StreamingSession(
            m, proc, GemmaAudioPrefiller(m, proc), cache_factory=KVCache,
            config=cfg,
            system="You are a helpful assistant. Answer in one short sentence.")

    fresh().warmup()          # compile before timing anything

    rows = []
    print(f"{len(cases)} spoken questions, "
          f"{sum(c['duration_s'] for c in cases):.1f}s of audio\n")
    for c in cases:
        s = fresh()
        s.reset()
        s.open_turn()
        audio = read_wav(Path(args.cases) / f"{c['key']}.wav")
        for i in range(0, len(audio), 1600):        # as a mic delivers it
            s.feed_audio(audio[i:i + 1600])
        t0 = time.monotonic()                       # clock starts at silence
        s.end_of_turn()
        reply = s.respond(max_tokens=48, eos_ids=eos)
        wait = (time.monotonic() - t0) * 1000
        ok = graded(reply, c["answers"])
        rows.append({"case": c["key"], "ok": ok, "wait_ms": wait,
                     "rt": wait / 1000 / c["duration_s"],
                     "hidden_ms": s.hidden_ms, "tail_ms": s.timing.tail_ms,
                     "decode_ms": s.timing.decode_ms,
                     "reply": reply.strip()[:70]})
        print(f"  {c['key']:4s} {'OK' if ok else '..'} {wait:6.0f}ms "
              f"({rows[-1]['rt']:.2f}x)  {reply.strip()[:56]!r}")

    n_ok = sum(r["ok"] for r in rows)
    waits = sorted(r["wait_ms"] for r in rows)
    print(f"\naccuracy      {n_ok}/{len(rows)}  ({100*n_ok/len(rows):.0f}%)")
    print(f"wait median   {statistics.median(waits):.0f}ms   "
          f"p90 {waits[int(len(waits)*0.9)]:.0f}ms   max {waits[-1]:.0f}ms")
    print(f"realtime      {statistics.median(r['rt'] for r in rows):.2f}x median")
    print(f"time split    tail {statistics.median(r['tail_ms'] for r in rows):.0f}ms "
          f"+ decode {statistics.median(r['decode_ms'] for r in rows):.0f}ms | "
          f"hidden under speech {statistics.median(r['hidden_ms'] for r in rows):.0f}ms")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def demo():
    assert graded("The answer is four.", ["4", "four"])
    assert not graded("It is Berlin.", ["paris"])
    # multi-answer needs 2 of 3, so one colour is not enough
    assert graded("red, blue and green", ["red", "blue", "yellow"])
    assert not graded("only red here", ["red", "blue", "yellow"])
    print("final_decision: grader OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
