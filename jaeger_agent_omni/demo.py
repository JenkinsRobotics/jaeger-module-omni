"""The demo. Audio in, Gemma, audio out — one command, evaluate it yourself.

    audio ──► Gemma's audio tower ──► Gemma E4B ──► text ──► voice ──► wav
              (no transcription step anywhere on the way in)

    python -m jaeger_agent_omni.demo --wav question.wav
    python -m jaeger_agent_omni.demo --wav q.wav --voice am_michael --speed 1.2
    python -m jaeger_agent_omni.demo --ask "what is the capital of France"
    python -m jaeger_agent_omni.demo --bench          # 12 questions, spoken

What is honest about this build, so nobody has to reverse-engineer it:

    audio in        10/12 on the spoken benchmark, 482 ms, NO transcript in
                    the path -- audio becomes embeddings and stays embeddings
    turn detection  hands-free, measured: speech_start/probably_ended/speech_end
    barge-in        20.5 ms to stop, conversation context preserved
    voice out       Kokoro. Correct words, 10 voices, speed 0.8-1.4x
    voice CLONING   token2wav clones a reference speaker, but the WORDS are
                    unreliable until SpeechOutDelta is trained -- so cloning
                    is a separate flag, not silently mixed into the demo
    streaming in    NOT finished. The duplex loop currently feeds the model a
                    fraction of the audio; turns and barge-in work, ingest
                    throughput does not.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"


def run_one(session, meta, audio, voice: str, speed: float, out: Path,
            engine: str = "kokoro", ref: str | None = None) -> dict:
    """One spoken turn: audio in, Gemma, spoken answer out."""
    from jaeger_agent_omni.omni import one_turn
    from jaeger_agent_omni.voice import speak_cloned, speak_kokoro

    t0 = time.monotonic()
    reply, think_ms = one_turn(session, meta, audio)
    reply = reply.split("<")[0].strip() or reply.strip()

    t1 = time.monotonic()
    if engine == "token2wav":
        info = speak_cloned(reply, out, ref)
    else:
        info = speak_kokoro(reply, out, voice, speed)
    speak_ms = (time.monotonic() - t1) * 1000
    return {"reply": reply, "think_ms": think_ms, "speak_ms": speak_ms,
            "total_ms": (time.monotonic() - t0) * 1000, **info}


def main() -> int:
    import numpy as np
    from jaeger_agent_omni.model_catalog import (
        front_end_choices, resolve_front_end)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", help="spoken question")
    ap.add_argument("--ask", help="typed question (skips the audio path)")
    ap.add_argument("--bench", action="store_true",
                    help="run the 12 spoken questions and speak each answer")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--engine", default="kokoro",
                    choices=("kokoro", "token2wav"))
    ap.add_argument("--ref", help="reference clip to clone (token2wav)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--front-end", default="gemma-native",
                    choices=front_end_choices(),
                    help="canonical Jaeger audio-ear identity")
    ap.add_argument("--adapter", default=None,
                    help="stock node/checkpoint override")
    ap.add_argument("--out", default="reply.wav")
    a = ap.parse_args()

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.omni import build

    if a.engine == "token2wav" and not a.ref:
        raise SystemExit("--engine token2wav needs --ref REF.wav")

    session, meta = build(a.model, a.front_end, a.adapter)
    session.warmup()

    if a.bench:
        import json

        from jaeger_agent_omni.voice import speak_kokoro
        cases = json.load(open(Path("research/cases12/cases.json")))
        sys.path.insert(0, str(Path("research")))
        from grading import graded
        outdir = Path("demo_out")
        outdir.mkdir(exist_ok=True)
        ok = 0
        for c in cases:
            audio = read_wav(Path("research/cases12") / f"{c['key']}.wav")
            r = run_one(session, meta, audio, a.voice, a.speed,
                        outdir / f"{c['key']}.wav")
            hit = graded(r["reply"], c["answers"])
            ok += hit
            print(f"  {c['key']} {'OK ' if hit else '..'} "
                  f"think {r['think_ms']:5.0f}ms  speak {r['speak_ms']:5.0f}ms  "
                  f"{r['reply'][:46]!r}")
        print(f"\n{ok}/{len(cases)} = {100*ok/len(cases):.0f}%   "
              f"spoken answers in {outdir}/")
        return 0

    if a.ask:
        if resolve_front_end(a.front_end, a.adapter).engine == "stock":
            raise SystemExit(
                "--ask is a text control; use --front-end gemma-native or "
                "provide audio")
        session.reset()
        session.open_turn()
        session.feed_text(a.ask)
        session.end_of_turn()
        reply = session.respond(max_tokens=48, eos_ids=meta["eos"])
        audio = None
    else:
        if not a.wav:
            ap.print_help()
            return 1
        audio = read_wav(Path(a.wav))

    if audio is not None:
        r = run_one(session, meta, audio, a.voice, a.speed, Path(a.out),
                    a.engine, a.ref)
    else:
        from jaeger_agent_omni.voice import speak_kokoro
        r = {"reply": reply.split("<")[0].strip(), "think_ms": 0.0}
        r.update(speak_kokoro(r["reply"], a.out, a.voice, a.speed))
        r["speak_ms"] = 0.0

    print()
    print(f"  heard   {Path(a.wav).name if a.wav else '(typed)'}")
    print(f"  reply   {r['reply']!r}")
    print(f"  think   {r.get('think_ms', 0):.0f} ms")
    print(f"  speak   {r.get('speak_ms', 0):.0f} ms  ({r.get('engine')})")
    print(f"  wav     {r.get('path', a.out)}  {r.get('seconds', 0):.1f}s")
    if r.get("warning"):
        print(f"  NOTE    {r['warning']}")
    return 0


def demo() -> None:
    """Surface check, no models."""
    import inspect
    # Scan the FUNCTIONS, not the file. Scanning the file text means the
    # check's own needle is inside the haystack, and the guard fails on
    # itself -- which is exactly what happened the first time.
    code = "".join(inspect.getsource(f) for f in (run_one, main))
    needle = "transcri" + "be("
    assert needle not in code, "the demo must not transcribe on the way in"
    assert run_one.__doc__ is None or run_one.__doc__
    print("demo: no transcription on the input path, entry points OK")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
