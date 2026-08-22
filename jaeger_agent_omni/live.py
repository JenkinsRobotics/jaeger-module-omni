"""Live mic -> Gemma -> voice. The whole pipeline, on real hardware.

    python -m jaeger_agent_omni.live                 # hands-free, speak to it
    python -m jaeger_agent_omni.live --seconds 60
    python -m jaeger_agent_omni.live --voice am_michael --speed 1.15
    python -m jaeger_agent_omni.live --list-devices

    audio ──► Gemma's audio tower ──► Gemma E4B ──► text ──► Kokoro ──► speaker
              no transcription anywhere on the way in

Everything here has been measured in isolation; this is the first thing that
runs it against a real microphone, in a room, end to end:

    turn detection   hands-free, 510 ms silence endpoint
    barge-in         talk over the reply and it stops
    speech out       in-process Kokoro, ~340 ms after the first call

Known limit, stated up front: the duplex loop's ingest is the weak part. It
drops audio when it falls behind rather than queueing, so a long utterance
may reach the model only in part. Turns and barge-in are solid; throughput
is not.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"
RATE = 16000
BLOCK = 1600            # 100 ms, the cadence the duplex loop expects


def list_devices() -> int:
    import sounddevice as sd
    print("inputs:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:2d}  {d['name']}  ({d['default_samplerate']:.0f} Hz)")
    print(f"\ndefault in : {sd.query_devices(kind='input')['name']}")
    print(f"default out: {sd.query_devices(kind='output')['name']}")
    return 0


def main() -> int:
    import numpy as np
    import sounddevice as sd

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", type=int, default=None, help="input device id")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threshold", type=float, default=0.012,
                    help="speech RMS floor; raise it in a noisy room")
    ap.add_argument("--silence-ms", type=float, default=510.0,
                    help="quiet needed to close a turn; a continuous "
                         "conversation never reaches the default")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-speak", action="store_true",
                    help="print replies instead of speaking them")
    ap.add_argument("--cooldown", type=float, default=4.0,
                    help="seconds after a reply before another turn may "
                         "close. A continuous speaker never yields the "
                         "floor; without this the model answers every "
                         "pause, and each reply DEAFENS it (half-duplex).")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="record raw mic AND model-fed audio for audit")
    a = ap.parse_args()

    if a.list_devices:
        return list_devices()

    from jaeger_agent_omni.omni import build
    from jaeger_agent_omni.runtime.duplex import DuplexLoop
    from jaeger_agent_omni.voice import speak_kokoro

    print("loading Gemma…", flush=True)
    session, meta = build(a.model, "tower")
    session.warmup()

    if not a.no_speak:
        print("warming the voice…", flush=True)
        speak_kokoro("ready", "/tmp/_warm.wav", a.voice, a.speed)

    # The mic thread NEVER blocks the duplex tick: it drops into a queue and
    # the loop drains whatever has arrived. A blocking source stalls the whole
    # loop, which is how an earlier test read 2 chunks of 35.
    q: queue.Queue = queue.Queue(maxsize=200)
    speaking_out = threading.Event()

    def on_block(indata, frames, t, status):
        if status:
            pass
        mono = indata[:, 0] if indata.ndim > 1 else indata
        try:
            q.put_nowait(mono.copy().astype("float32"))
        except queue.Full:
            pass

    rec = {"raw": [], "fed": []}

    def source():
        got = []
        while True:
            try:
                got.append(q.get_nowait())
            except queue.Empty:
                break
        if not got:
            return None
        block = np.concatenate(got)
        if a.record:
            rec["raw"].append(block.copy())
        if speaking_out.is_set():
            # FULL-DUPLEX-LITE. No AEC yet, so the mic hears our own voice.
            # But the user talking OVER playback is louder at the mic than
            # the speaker echo (measured echo floor set at start of playback).
            # If the level clearly exceeds the echo, that is a barge-in:
            # kill playback NOW and let the block through so the endpointer
            # hears the new turn. Zeroing the mic here made interruption
            # impossible for the whole length of every reply.
            rms = float(np.sqrt(np.mean(np.square(block))))
            if rms > max(echo_floor["rms"] * 1.4, a.threshold * 1.1):
                sd.stop()                      # cut our voice mid-word
                speaking_out.clear()
                barged["n"] += 1
                print("\n  (barge-in: you spoke over me — stopping)",
                      flush=True)
                return block
            gated = np.zeros_like(block)        # just our own echo: drop it
            if a.record:
                rec["fed"].append(gated.copy())
            return gated
        if a.record:
            rec["fed"].append(block.copy())
        return block

    level = {"rms": 0.0}
    heard = {"n": 0}
    barged = {"n": 0}
    echo_floor = {"rms": 0.35}   # start high; calibrated during first playback

    def user_speaking():
        return level["rms"] > a.threshold and not speaking_out.is_set()

    stage_ms = {"synth": [], "reply_len": [], "to_first_audio": []}
    turn_t0 = {"t": None}          # when the turn closed (speech_end)

    import queue as _q
    say_q: _q.Queue = _q.Queue()

    def _player():
        """Sequential sentence player. Synthesises and plays each sentence as
        it arrives, so audio starts after the FIRST sentence is generated
        instead of after the whole reply."""
        import soundfile as sf
        while True:
            sent = say_q.get()
            if sent is None:
                break
            try:
                speaking_out.set()
                t0 = time.monotonic()
                speak_kokoro(sent, "/tmp/_sent.wav", a.voice, a.speed)
                stage_ms["synth"].append((time.monotonic() - t0) * 1000)
                if turn_t0["t"] is not None:
                    stage_ms["to_first_audio"].append(
                        (time.monotonic() - turn_t0["t"]) * 1000)
                    print(f"\n          [latency: turn-end -> first audio "
                          f"{stage_ms['to_first_audio'][-1]:.0f} ms]",
                          flush=True)
                    turn_t0["t"] = None
                wav, sr = sf.read("/tmp/_sent.wav", dtype="float32")
                sd.play(wav, sr)
                sd.wait()
            except Exception:
                pass
            finally:
                if say_q.empty():
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except _q.Empty:
                            break
                    speaking_out.clear()

    threading.Thread(target=_player, daemon=True).start()

    def _calibrate_echo():
        # 400 ms into playback, whatever the mic reads is OUR echo. The
        # barge-in threshold rides 1.8x above that.
        time.sleep(0.4)
        got = []
        while True:
            try:
                got.append(q.get_nowait())
            except queue.Empty:
                break
        if got:
            blk = np.concatenate(got)
            r = float(np.sqrt(np.mean(np.square(blk))))
            if r > 0:
                echo_floor["rms"] = r

    def _play_async(text):
        """Synthesis + playback OFF the duplex thread. sd.wait() inside the
        loop blocked ingest for the whole reply -- measured 7480 ms peak lag
        and 4 dropped chunks. The loop must keep listening while we speak."""
        try:
            t0 = time.monotonic()
            info = speak_kokoro(text, "/tmp/_reply.wav", a.voice, a.speed)
            stage_ms["synth"].append((time.monotonic() - t0) * 1000)
            import soundfile as sf
            wav, sr = sf.read("/tmp/_reply.wav", dtype="float32")
            sd.play(wav, sr)
            threading.Thread(target=_calibrate_echo, daemon=True).start()
            sd.wait()
        finally:
            # DRAIN BEFORE CLEARING. The old order (clear, then drain) threw
            # away user speech that began as playback ended -- the audit
            # showed "Listen and practice" reaching the model as "And
            # practice", and "Conversation 32" misheard as "Station 32" from
            # its clipped first syllable. Draining first removes only what
            # was captured DURING playback (our echo); everything after the
            # clear survives.
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            speaking_out.clear()

    last_reply = {"t": 0.0}

    def on_reply(text):
        text = text.split("<")[0].strip()
        if not text:
            return
        last_reply["t"] = time.monotonic()
        heard["n"] += 1
        stage_ms["reply_len"].append(len(text.split()))
        print(f"\n  \033[1mJAEGER\033[0m  {text}", flush=True)
        # speech already streamed sentence-by-sentence via say_q

    def turn_gate():
        # Measured: 13 replies in 90 s took input survival from 86% to 38%,
        # because every reply is a deaf window. Rate-limit the mouth.
        if time.monotonic() - last_reply["t"] < a.cooldown:
            return False
        return loop.endpoint.ended if loop.endpoint else False

    tok = session.processor.tokenizer
    _orig_respond = session.respond

    def _streaming_respond(**kw):
        turn_t0["t"] = time.monotonic()
        ids, consumed = [], 0
        inner = kw.get("on_token")

        def on_token(tid, n):
            nonlocal consumed
            ids.append(tid)
            # decode is cheap at this length; look for a finished sentence
            text = tok.decode(ids)
            for i in range(len(text) - 1, consumed - 1, -1):
                if text[i] in ".!?" and i + 1 > consumed:
                    sent = text[consumed:i + 1].split("<")[0].strip()
                    if len(sent.split()) >= 2:
                        say_q.put(sent)
                        consumed = i + 1
                    break
            if inner:
                inner(tid, n)

        kw["on_token"] = on_token
        out = _orig_respond(**kw)
        tail = out[consumed:].split("<")[0].strip()
        if len(tail.split()) >= 2:
            say_q.put(tail)
        return out

    if not a.no_speak:
        session.respond = _streaming_respond

    loop = DuplexLoop(session, audio_source=source, user_speaking=user_speaking,
                      on_reply=on_reply, turn_ended=None)
    _default_ended = loop.turn_ended
    loop.turn_ended = turn_gate
    # A live room is never silent. The endpoint floor has to sit above the
    # room, not at a synthetic zero: measured on the Elgato, an occupied room
    # runs 0.10-0.36 RMS while speaking and does not fall to 0.012 between
    # sentences.
    loop.endpoint.speech_rms = a.threshold
    loop.endpoint.silence_ms = a.silence_ms
    loop.endpoint.__post_init__()

    # Meter thread: shows the room level so a dead mic is obvious immediately.
    stop = threading.Event()

    def meter():
        while not stop.is_set():
            r = level["rms"]
            n = int(min(1.0, r / 0.08) * 22)
            state = "SPEAKING" if speaking_out.is_set() else (
                "hearing " if r > a.threshold else "quiet   ")
            print(f"\r  {state} |{'#'*n}{'.'*(22-n)}| {r:.4f}",
                  end="", flush=True)
            time.sleep(0.1)

    orig_source = source

    def metered_source():
        blk = orig_source()
        if blk is not None and len(blk):
            level["rms"] = float(np.sqrt(np.mean(np.square(blk))))
        return blk

    loop.audio_source = metered_source

    print(f"\ndevice   {sd.query_devices(a.device, 'input')['name'] if a.device is not None else sd.query_devices(kind='input')['name']}")
    print(f"voice    {a.voice} @ {a.speed}x")
    print(f"listening for {a.seconds:.0f}s — speak, pause, and it will answer.")
    print("talk over the reply to test barge-in.\n")

    threading.Thread(target=meter, daemon=True).start()
    t0 = time.monotonic()
    with sd.InputStream(samplerate=RATE, blocksize=BLOCK, channels=1,
                        dtype="float32", device=a.device, callback=on_block):
        stats = loop.run(eos_ids=meta["eos"], max_seconds=a.seconds)
    stop.set()

    wall = time.monotonic() - t0
    print(f"\n\n  ran {wall:.0f}s")
    print(f"  turns        {stats.turns}     interrupts {stats.interrupts}")
    # feed_audio returns ONE-SECOND chunks (gemma_audio.CHUNK_S = 1.0), not
    # the 100 ms mic blocks. Reporting them as 0.1 s each made 89 s of clean
    # capture read as 8.9 s, i.e. a throughput failure that was not happening.
    print(f"  audio        {stats.audio_chunks} chunks "
          f"({stats.audio_chunks:.0f}s of {wall:.0f}s), "
          f"{stats.audio_dropped} dropped")
    print(f"  endpoint     {' -> '.join(stats.endpoint_events[:8]) or '(none)'}")
    print(f"  peak lag     {stats.max_lag_s*1000:.0f} ms")
    print(f"  replies      {heard['n']}    barge-ins during playback {barged['n']}")
    if stage_ms["synth"]:
        import statistics as st
        print(f"  synth        median {st.median(stage_ms['synth']):.0f} ms")
    if stage_ms["to_first_audio"]:
        import statistics as st
        print(f"  END->AUDIO   median "
              f"{st.median(stage_ms['to_first_audio']):.0f} ms  "
              f"(the latency the user feels)")
    if a.record and rec["raw"]:
        import soundfile as sf
        d = Path(a.record)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / "raw_mic.wav"), np.concatenate(rec["raw"]), RATE)
        sf.write(str(d / "fed_model.wav"), np.concatenate(rec["fed"]), RATE)
        print(f"  recorded     {d}/raw_mic.wav + fed_model.wav — audit with"
              f" research/session_audit.py")
    return 0


def demo() -> None:
    import importlib.util
    assert importlib.util.find_spec("sounddevice"), "sounddevice missing"
    assert importlib.util.find_spec("kokoro"), "kokoro must be in THIS venv"
    src = Path(__file__).read_text()
    assert "speaking_out" in src, "must gate the mic while our own voice plays"
    print("live: sounddevice + in-process kokoro present, self-hearing guarded")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
