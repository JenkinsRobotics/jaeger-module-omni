"""Does the selected production audio path actually do full duplex? Run it.

Every piece of this had a passing self-check for a day before anything ran
them together with real audio and a real model:

    streaming prefill      session.feed_audio, chunked into the KV cache
    hands-free turn end    Endpoint, 510 ms silence + speculative commit
    barge-in               session.interrupt, cache rewind, context kept

`omni.py --duplex` was documented in the module docstring and the flag
existed, but it was never wired -- so "full duplex works" was an assertion
about components, not a measured property of the system.

    python research/duplex_check.py                 # synthetic speech/silence
    python research/duplex_check.py --wav clip.wav  # real audio
    python research/duplex_check.py --barge-in      # interrupt mid-reply
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"


def main() -> int:
    import numpy as np

    from jaeger_agent_omni.core.model_catalog import DEFAULT_FRONT_END
    from jaeger_agent_omni.omni import build
    from jaeger_agent_omni.core.runtime.duplex import SAMPLE_RATE, DuplexLoop

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--front-end", default=DEFAULT_FRONT_END)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--barge-in", action="store_true",
                    help="speak again while the model is replying")
    ap.add_argument("--json", type=Path,
                    help="optional machine-readable result artifact")
    a = ap.parse_args()

    session, meta = build(a.model, a.front_end)
    session.warmup()

    if a.wav:
        from jaeger_agent_omni.app import read_wav
        audio = read_wav(Path(a.wav))
    else:
        # 2 s of speech-shaped noise then 1.5 s quiet: enough to trip
        # speech_start, probably_ended, and speech_end in order.
        rng = np.random.default_rng(0)
        audio = np.concatenate([
            rng.normal(0, 0.25, SAMPLE_RATE * 2).astype(np.float32),
            np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)])

    pos = [0]
    step = SAMPLE_RATE // 10          # 100 ms, the cadence a mic delivers

    # A microphone BUFFER does not block: it returns what has arrived and
    # nothing otherwise. Sleeping inside the source stalls the whole duplex
    # tick, which made the first run read 2 chunks consumed of 35 and a 2 s
    # lag -- a property of the harness, not of the system under test.
    start = [None]

    def source():
        now = time.monotonic()
        if start[0] is None:
            start[0] = now
        due = int((now - start[0]) * SAMPLE_RATE)
        if due <= pos[0] or pos[0] >= len(audio):
            return None                     # nothing captured yet
        end = min(len(audio), pos[0] + step, due)
        chunk = audio[pos[0]:end]
        pos[0] = end
        return chunk if len(chunk) else None

    # Barge-in: report "user is speaking" once the reply has started, which is
    # what a VAD would do when someone talks over the model.
    speaking = {"on": False}

    def user_speaking():
        return a.barge_in and speaking["on"]

    replies = []

    def on_reply(text):
        replies.append(text)

    loop = DuplexLoop(session, audio_source=source,
                      user_speaking=user_speaking, on_reply=on_reply)

    if a.barge_in:
        orig = session.respond

        def respond(**kw):
            # Capture the caller's callback FIRST. Overwriting kw["on_token"]
            # with a closure that reads kw["on_token"] makes it call itself.
            inner = kw.get("on_token")

            def on_token(tid, n):
                # Flip after a few tokens so the interrupt lands mid-reply,
                # not before generation starts.
                if n >= 3:
                    speaking["on"] = True
                if inner is not None:
                    inner(tid, n)

            kw["on_token"] = on_token
            return orig(**kw)

        session.respond = respond

    t0 = time.monotonic()
    stats = loop.run(eos_ids=meta["eos"], max_seconds=a.seconds)
    wall = time.monotonic() - t0

    secs = len(audio) / SAMPLE_RATE
    print()
    consumed_s = stats.audio_samples / SAMPLE_RATE
    print(f"consumed     {consumed_s:.1f} s of {secs:.1f} s "
          f"({100 * consumed_s / max(secs, 1e-9):.0f}%)")
    print(f"audio        {secs:.1f} s, streamed in 100 ms chunks")
    print(f"wall         {wall:.1f} s   ({wall/max(secs,1e-9):.2f}x realtime)")
    print(f"endpoint     {' -> '.join(stats.endpoint_events) or '(none)'}")
    print(f"turns        {stats.turns}    interrupts {stats.interrupts}")
    print(f"audio        {stats.audio_chunks} chunks, {stats.audio_dropped} dropped")
    print(f"peak lag     {stats.max_lag_s * 1000:.0f} ms")
    if session.cancel_latency_ms is not None:
        print(f"barge-in     {session.cancel_latency_ms:.1f} ms to stop")
    for r in replies:
        print(f"reply        {r.strip()[:88]!r}")

    checks = {
        # ``audio_chunks`` counts adapter windows committed *before* endpoint,
        # not microphone callbacks.  The causal adapter deliberately holds a
        # short (<12 s) turn for one bidirectional Whisper pass and therefore
        # reports zero here even though every 100 ms callback reached it.  The
        # ingestion invariant is measured by accepted samples plus the
        # independent consumed/drop checks below.
        "accepted live PCM": stats.audio_samples > 0,
        "detected speech end": "speech_end" in stats.endpoint_events,
        "closed a turn unprompted": stats.turns >= 1,
        "kept up with realtime": stats.max_lag_s < 1.0,
        # THE CHECK THAT MATTERS. Peak lag alone passed while 2 of ~35 chunks
        # reached the model and one was dropped -- "full duplex" with 94% of
        # the audio missing. Consumed seconds must approach the clip length.
        "consumed most of the audio":
            consumed_s >= secs * 0.8,
        "dropped nothing": stats.audio_dropped == 0,
    }
    if a.barge_in:
        checks["interrupted mid-reply"] = stats.interrupts >= 1
    print()
    for name, ok in checks.items():
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    result = {
        "kind": "jaeger_duplex_check",
        "evaluation": "free_running_audio_native",
        "teacher_forced": False,
        "front_end": meta.get("model_id", a.front_end),
        "checkpoint": meta.get("checkpoint"),
        "barge_in_requested": a.barge_in,
        "audio_seconds": secs,
        "consumed_seconds": consumed_s,
        "wall_seconds": wall,
        "turns": stats.turns,
        "interrupts": stats.interrupts,
        "audio_chunks": stats.audio_chunks,
        "audio_chunk_semantics": "adapter windows committed before endpoint",
        "audio_dropped": stats.audio_dropped,
        "max_lag_ms": stats.max_lag_s * 1000,
        "cancel_latency_ms": session.cancel_latency_ms,
        "endpoint_events": list(stats.endpoint_events),
        "checks": checks,
        "passed": passed,
        "replies": replies,
    }
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"artifact     {a.json}")
    print(f"\n{'FULL DUPLEX OK' if passed else 'NOT FULLY DUPLEX'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
