#!/usr/bin/env python3
"""Gemma Omni V2 converse harness — wires the parallel and series channels.

Imported from VoiceLLM-Playground "Omni Model/gemma omni v2" run_v2.py at
its 27/27 campaign state (playground commit 9f1103b, 2026-09-03). Node
folders travel unmodified under the shared manifest contract; this file only
changed its docstring. Nodes resolve from the sibling nodes/ directory and
the engine core from core/ (flat modules), exactly as upstream.

    python -m jaeger_agent_omni.converse --selftest            # node contracts, no models
    python -m jaeger_agent_omni.converse --enroll NAME F.wav   # one-shot wake enrollment
    python -m jaeger_agent_omni.converse --wav F.wav           # deterministic full pipeline
    python -m jaeger_agent_omni.converse --mic                 # live microphone
    python -m jaeger_agent_omni.converse --converse half|quasi|full [--barge stop|continue]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
FRAME = 480                                   # 30 ms at 16 kHz


def load_node(node_id):
    p = HERE / "nodes" / node_id / "runtime.py"
    spec = importlib.util.spec_from_file_location(f"n_{node_id}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"n_{node_id}"] = m
    spec.loader.exec_module(m)
    return m.build()


def write_stats(node_id, node):
    p = HERE / "nodes" / node_id / "stats"
    p.mkdir(exist_ok=True)
    (p / f"session_{int(time.time())}.json").write_text(
        json.dumps(node.stats(), indent=2) + "\n")


def run_stream(frames, *, with_stt: bool):
    """Feed 30 ms frames through the parallel nodes; cut utterances on the
    turn-state channel; run series STT per utterance; fuse."""
    spk = load_node("para-speaker-embed-v0")
    turn = load_node("para-turn-state-v0")
    wake = load_node("para-wake-spotter-v0")
    fuse = load_node("fusion-context-v0")
    stt = load_node("semantic-whisper-v0") if with_stt else None

    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    out = open(logs / f"v2_{time.strftime('%Y-%m-%d')}.jsonl", "a",
               buffering=1)
    utter: list[np.ndarray] = []
    wake_hit = None
    records = []
    t = 0.0
    for fr in frames:
        t += 0.03
        s_out = spk.process(fr, t)
        t_out = turn.process(fr, t)
        w_out = wake.process(fr, t)
        if w_out.get("wake"):
            wake_hit = w_out["wake"]
            print(f"[wake] «{wake_hit}» dist={w_out['dist']}", flush=True)
        if t_out["in_turn"] or t_out["speaking"]:
            utter.append(np.asarray(fr, np.float32).reshape(-1))
        if t_out["turn_ended"] and utter:
            audio = np.concatenate(utter)
            utter = []
            text = (stt.process(audio, t)["text"] if stt else
                    f"<{len(audio)/16000:.1f}s utterance>")
            rec = fuse.process({"text": text, "speaker": s_out,
                                "turn": t_out, "wake": wake_hit}, t)
            wake_hit = None
            out.write(json.dumps(rec) + "\n")
            print(f"{rec['context_line']}  {text}", flush=True)
            records.append(rec)
    spk.save()
    for nid, n in [("para-speaker-embed-v0", spk), ("para-turn-state-v0", turn),
                   ("para-wake-spotter-v0", wake), ("fusion-context-v0", fuse)]:
        write_stats(nid, n)
    if stt:
        write_stats("semantic-whisper-v0", stt)
    return records


def wav_frames(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a[:, 0]
    if sr != 16000:
        idx = np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr))
        a = a[idx.astype(int)].astype(np.float32)
    n = len(a) // FRAME * FRAME
    return [a[i:i + FRAME] for i in range(0, n, FRAME)]


def mic_frames():
    import queue
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=FRAME,
                        callback=lambda ind, f, ti, st: q.put(ind[:, 0].copy())):
        print("listening — Ctrl-C to stop", flush=True)
        try:
            while True:
                yield q.get()
        except KeyboardInterrupt:
            return


# ── selftest: node contracts, synthetic audio, no models ─────────────────
def _voice(f0, gain=0.08, secs=1.5, mod=3.0):
    t = np.arange(int(16000 * secs)) / 16000
    sig = sum(a * np.sin(2 * np.pi * f0 * k * t)
              for k, a in [(1, 1.0), (2, 0.5), (3, 0.25)])
    return (gain * sig / 1.75 * (1 + 0.3 * np.sin(2 * np.pi * mod * t))
            ).astype(np.float32)


def _frames(sig):
    n = len(sig) // FRAME * FRAME
    return [sig[i:i + FRAME] for i in range(0, n, FRAME)]


def selftest() -> int:
    checks = []
    ok = lambda c, w: checks.append((bool(c), w))

    spk = load_node("para-speaker-embed-v0")
    spk.centroids, spk.counts = [], []          # ignore persisted state
    t = 0.0
    for fr in _frames(_voice(125, secs=2.0)):
        t += 0.03
        a = spk.process(fr, t)
    for fr in _frames(_voice(210, secs=2.0)):
        t += 0.03
        b = spk.process(fr, t)
    ok(a["speaker"] == 1 and b["speaker"] == 2,
       "parallel speaker channel separates two voices")

    turn = load_node("para-turn-state-v0")
    ended = []
    t = 0.0
    for fr in _frames(_voice(125, secs=1.0)) + [np.zeros(FRAME, np.float32)] * 25:
        t += 0.03
        ended.append(turn.process(fr, t)["turn_ended"])
    ok(any(ended), "turn-state channel endpoints speech through silence")
    ok(sum(ended) == 1, "one utterance ends exactly once")

    import tempfile
    _tmp = Path(tempfile.mkdtemp()) / "templates.npz"
    wk = load_node("para-wake-spotter-v0")
    wk._wfile = lambda: _tmp          # HERMETIC: the selftest once clobbered
    wk.templates = {}                 # real enrollments (review 2026-09-02 #1)
    phrase = _voice(180, secs=1.0, mod=7.0)      # distinctive cadence
    wk.enroll("test-wake", phrase)
    ok(_tmp.exists(),
       "one-shot enrollment persists — into scratch, never the real weights")
    t, hit = 0.0, None
    for fr in _frames(np.concatenate([np.zeros(16000, np.float32), phrase])):
        t += 0.03
        r = wk.process(fr, t)
        if r.get("wake"):
            hit = r["wake"]
    ok(hit == "test-wake", "the enrolled waveform is spotted — no STT involved")
    wk2 = load_node("para-wake-spotter-v0")      # fresh instance
    wk2._wfile = lambda: _tmp
    wk2.templates = {}
    wk2._load()                                  # from scratch file only
    t, hit2 = 100.0, None
    for fr in _frames(_voice(95, secs=1.2, mod=1.0)):
        t += 0.03
        if wk2.process(fr, t).get("wake"):
            hit2 = True
    ok(hit2 is None, "a different voice/cadence does not false-wake")

    fuse = load_node("fusion-context-v0")
    rec = fuse.process({"text": "hello there.",
                        "speaker": {"speaker": 2, "conf": 0.8,
                                    "pitch_hz": 200.0},
                        "turn": {"in_turn": False, "turn_ended": True},
                        "wake": "test-wake"}, 1.0)
    ok(rec["context_line"].startswith("[context: speaker=S2")
       and rec["wake"] == "test-wake" and rec["complete"],
       "fusion converges the channels into one record")

    ok(text_wake("hey computer what time is it") is not None
       and text_wake("computer, status report")[0] == "computer"
       and text_wake("i bought a new computer yesterday") is None,
       "text-wake fallback: two-token, address-position, no mid-sentence")
    ok(is_dismissal("okay goodbye now") and not is_dismissal("good bag"),
       "dismissal closes; lookalikes don't")
    ok(is_garble({"wpm": 900, "text": "one two three four five six"})
       and not is_garble({"wpm": 900, "text": "yes please"}),
       "garble guard runs on graph evidence, short bursts exempt")
    txt_n = load_node("text-input-v0")
    txt_n.submit("  hello  ")
    ok(txt_n.poll() == "hello" and txt_n.poll() is None,
       "text channel is a declared node with a clean contract")
    rec = load_node("fusion-context-v0").process(
        {"text": "hello there friend", "speaker": {"speaker": 1, "conf": 0.9,
         "pitch_hz": 120.0, "energy_db": -20.0},
         "turn": {"in_turn": False, "turn_ended": True}, "wake": None,
         "duration_s": 1.5}, 0.0)
    ok(rec["wpm"] == 120.0 and rec["energy_db"] == -20.0,
       "fusion carries wpm+energy so guards run on graph evidence")

    h, rp = parse_heard_reply(
        "HEARD: what time is it\nREPLY: It is ten o'clock.")
    ok(h == "what time is it" and rp == "It is ten o'clock.",
       "one generation carries transcript AND answer")
    h2, rp2 = parse_heard_reply("just an answer with no format")
    ok(h2 == "" and rp2 == "just an answer with no format",
       "format miss degrades to reply-only, never crashes")

    hspec = importlib.util.spec_from_file_location(
        "heads_math", HERE / "nodes/para-perception-heads-v1/runtime.py")
    hmod = importlib.util.module_from_spec(hspec)
    hspec.loader.exec_module(hmod)
    summ = hmod.summarize_heads({
        "directed": np.full(20, 3.0), "turn_end": np.linspace(-4, 4, 20),
        "overlap": np.full(20, -5.0), "speaker_change": np.full(20, -5.0)})
    ok(summ["directed_p"] > 0.9 and summ["turn_end_p"] > 0.9
       and summ["overlap_p"] < 0.05 and summ["speaker_change_frames"] == [],
       "head pooling math: confident logits -> confident probabilities")

    ok(is_stock("Thank you.", 45) and not is_stock("Thank you.", 150)
       and not is_stock("please check the calendar", 45),
       "stock artifact guard: density-gated, real speech survives (gap 3)")

    h13, t13 = stream_split(np.zeros(13 * 16000, np.float32))
    h1, t1 = stream_split(np.zeros(1 * 16000, np.float32))
    ok(abs(h13.size - 11 * 16000) < 5 and abs(t13.size - 2 * 16000) < 5
       and h1.size == 0 and t1.size == 16000,
       "stream_split: long turn commits head, keeps 2s tail; short turn whole")

    for good, what in checks:
        print(f"  {'PASS' if good else 'FAIL'}  {what}")
    bad = sum(1 for g, _ in checks if not g)
    print(f"\n{len(checks)-bad}/{len(checks)} checks passed")
    return 1 if bad else 0


# ── conversation layer: the para channel RUNS turn-taking ─────────────────
WAKE_PREFIXES = ("ok", "okay", "hey", "yo", "hi")
WAKE_NAMES = ("whisper", "jaeger", "assistant", "computer", "google",
              "siri", "alexa", "jarvis", "gemini", "claude",
              "yeager", "yager", "jager", "wisper")
_DISMISS = ("goodbye", "good bye", "bye", "that's all", "thats all",
            "stop listening", "go to sleep", "never mind", "nevermind")
NN_SYSTEM = ("You are a helpful voice assistant. The user's message arrives "
             "as speech. Reply with EXACTLY two lines:\n"
             "HEARD: <verbatim transcription of the speech>\n"
             "REPLY: <your answer in one or two short conversational "
             "sentences>")


def parse_heard_reply(raw: str) -> tuple[str, str]:
    """One generation carries transcript AND answer (probed 2026-09-02:
    5/5, warm turns 1.4–1.6 s vs ~5.5 s for the two-model cascade)."""
    heard, reply = "", raw.strip()
    for ln in raw.splitlines():
        t = ln.strip()
        if t.upper().startswith("HEARD:"):
            heard = t.split(":", 1)[1].strip()
        elif t.upper().startswith("REPLY:"):
            reply = t.split(":", 1)[1].strip()
    return heard, reply


SYSTEM_PROMPT = ("You are a helpful voice assistant. Answer in 1-2 short "
                 "sentences of plain conversational English. No markdown, "
                 "no lists. Each user message may begin with one bracketed "
                 "[context: ...] line describing how it was spoken; use it "
                 "to judge tone, never mention it.")


def text_wake(text: str):
    """Text-domain fallback wake: the QbE spotter (waveform, one-shot) is
    preferred when templates are enrolled; this covers the unenrolled case."""
    import re as _re
    words = _re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    for i in range(len(words) - 1):
        if words[i] in WAKE_PREFIXES and words[i + 1] in WAKE_NAMES:
            return " ".join(words[i:i + 2]), " ".join(words[i + 2:])
    for i in (0, len(words) - 1) if words else ():
        if words[i] in WAKE_NAMES:
            rest = words[1:] if i == 0 else words[:-1]
            return words[i], " ".join(rest)
    return None


def is_dismissal(text: str) -> bool:
    t = " ".join(text.lower().split())
    return any(d in t for d in _DISMISS)


_STOCK = {"thank you", "thanks", "thank you for watching",
          "thanks for watching", "you", "bye", "okay", "thank you bye",
          "good luck"}


def is_stock(heard: str, wpm) -> bool:
    """Whisper-family silence artifact, gap 3: stock phrases at implausibly
    low density are residue, not speech (family lesson, live-verified)."""
    t = " ".join(heard.lower().split()).strip(" .!?")
    return t in _STOCK and (wpm or 0) < 80


def is_garble(rec) -> bool:
    """Cascade lesson: AEC residue decodes at impossible density."""
    return rec.get("wpm", 0) > 550 and len(rec["text"].split()) >= 5


STREAM_COMMIT_S = 12.0       # a single turn longer than this commits its head
STREAM_KEEP_S = 2.0          # mid-stream and keeps only the tail, so an
                             # unbounded monologue never grows without limit
                             # (family 04 lesson). RARE by design: real and
                             # synthetic speech both carry sub-500 ms pauses,
                             # so turns almost always endpoint before 12 s —
                             # this is the safety valve for the ones that do
                             # not. Verified by selftest, not the live path.


def stream_split(audio_16k, keep_s: float = STREAM_KEEP_S):
    """Split a long in-progress turn into (head_to_commit, tail_to_keep).
    Pure — the caller transcribes the head and re-seeds the buffer with the
    tail. Short turns pass through whole (nothing to commit yet)."""
    keep = int(keep_s * 16000)
    if audio_16k.size <= keep:
        return np.zeros(0, np.float32), audio_16k
    return audio_16k[:-keep], audio_16k[-keep:]


def load_trained_semantic():
    """The POINT of this folder: audio -> whisper ENCODER -> trained adapter
    -> Gemma. No text seam, no whisper decoder. The v0 text path exists only
    as an explicitly-labeled fallback.

    Consolidated 2026-09-03 onto jaegeromni-41m-converse (the repository's
    causal-balanced 41M at this seam, prompt-aligned: 5.03% vs 5.65%
    devclean64). gemmaomni-v2-stream-bargein-93h remains packaged, disabled."""
    p = HERE / "nodes/jaegeromni-41m-converse/runtime.py"
    spec = importlib.util.spec_from_file_location("nn_semantic", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["nn_semantic"] = m
    spec.loader.exec_module(m)
    return m.build()


def converse(mode: str, barge: str, with_vision: bool,
             semantic: str = "nn", reply_mode: str = "nn") -> int:
    """--converse half|quasi|full: a functioning agent on the v2 graph.
    Turn-taking lives on the PARALLEL channel (eot_pressure endpoints,
    QbE wake); the duplex ladder mirrors the verified cascade family:
    half = mic closed while speaking, quasi = open mic + AEC + barge,
    full = quasi + streaming semantics (forming preview, mid-turn commit).
    This is the scaffold every trained head drops into."""
    import sounddevice as sd
    spk = load_node("para-speaker-embed-v0")
    turn = load_node("para-turn-state-v0")
    wake = load_node("para-wake-spotter-v0")
    fuse = load_node("fusion-context-v0")
    ecapa = load_node("para-speaker-ecapa-v1")
    try:
        ecapa.load()
        print("[converse] speaker+wake embeddings: ECAPA (trained NN)",
              flush=True)
    except Exception as exc:
        ecapa = None
        print(f"[converse] ECAPA unavailable ({exc}) — hand fingerprint "
              f"fallback", flush=True)
    if semantic == "nn":
        print("[converse] semantic channel: TRAINED adapter path "
              "(encoder→adapter→Gemma, no text seam)…", flush=True)
        nn = load_trained_semantic()
        stt = None
        try:
            hspec = importlib.util.spec_from_file_location(
                "node_heads",
                HERE / "nodes/para-perception-heads-v1/runtime.py")
            hm = importlib.util.module_from_spec(hspec)
            sys.modules["node_heads"] = hm
            hspec.loader.exec_module(hm)
            heads = hm.build(transcriber=nn)
            print("[converse] perception heads: TRAINED (directed / "
                  "turn_end / overlap / speaker_change on the trunk)",
                  flush=True)
        except Exception as exc:
            heads = None
            print(f"[converse] perception heads unavailable: {exc}",
                  flush=True)
    else:
        print("[converse] semantic channel: TEXT FALLBACK (whisper decoder) "
              "— not the omni path", flush=True)
        nn = None
        heads = None
        stt = load_node("semantic-whisper-v0")
    llm = None                     # loaded LAZILY: nn-reply voice turns
                                   # never need the second Gemma (~5 GB);
                                   # typed/image turns load it on first use

    def get_llm():
        nonlocal llm
        if llm is None:
            llm = load_node("llm-gemma-v0")
            llm.load(vis.handler)
            llm.history = nn_hist          # ONE memory across models (gap 1)
        return llm
    tts = load_node("tts-kokoro-v0")
    txt = load_node("text-input-v0")
    vis = load_node("semantic-vision-v0")

    vis.load(with_vision)
    if reply_mode != "nn" or semantic != "nn":
        get_llm()
    tts.load()
    if stt is not None:
        stt.process(np.zeros(16000, np.float32), 0.0)  # warm whisper
    else:
        nn.transcribe(np.zeros(16000, np.float32))     # warm the NN path

    io = None
    if mode in ("quasi", "full"):
        dx = importlib.util.spec_from_file_location(
            "node_dx", HERE / "nodes/audio-duplex-io-v0/runtime.py")
        dxm = importlib.util.module_from_spec(dx)
        sys.modules["node_dx"] = dxm
        dx.loader.exec_module(dxm)
        if dxm.pyaec is None:
            raise SystemExit("pyaec required for --converse quasi/full")
        dxm.BARGE_MODE = barge
        io = dxm.AudioIO().__enter__()
        monitor_cls = dxm.BargeMonitor
    print(f"[converse] {mode} duplex · barge={barge} · say 'hey computer …' "
          f"or type (Enter). Ctrl-C to quit.", flush=True)

    state = {"followup": 0.0, "utt": [], "hold": None, "speaking": False,
             "image": None, "pending": [], "last_wake": None,
             "committed": [], "preview_s": 0.0}
    q: queue.Queue = queue.Queue()

    def pump(fr, t):
        """ONE owner for perception events, whoever holds the queue — the
        playback watcher used to consume endpoints and drop them (review
        2026-09-02, #4). Endpointed utterances land in state['pending'];
        only the main loop answers them."""
        s_out = spk.process(fr, t)
        t_out = turn.process(fr, t)
        w_out = wake.process(fr, t)
        if t_out["in_turn"] or t_out["speaking"]:
            state["utt"].append(np.asarray(fr, np.float32).reshape(-1))
        if w_out.get("wake"):
            state["last_wake"] = w_out["wake"]
        if t_out["turn_ended"] and state["utt"]:
            audio = np.concatenate(state["utt"])
            state["utt"] = []
            state["preview_s"] = 0.0
            state["pending"].append((audio, s_out, state["last_wake"]))
            state["last_wake"] = None

    def drain_pending():
        while state["pending"] and not state["speaking"]:
            audio, s_out, w_hit = state["pending"].pop(0)
            on_utterance(audio, s_out, w_hit)

    def speak(reply: str) -> bool:
        state["speaking"] = True
        finished = True
        played24: list = []
        try:
            if io is None:                      # HALF: device closed
                for ch in tts.synth(reply):
                    played24.append(ch)
                    sd.play(ch, 24000); sd.wait()
                time.sleep(0.1)
                with q.mutex:
                    q.queue.clear()
            else:                               # FULL: live mic + barge
                io.clear_barge()
                io.drain()                       # barge considers audio
                # DURING playback only — stale frames buffered before the
                # agent started speaking must not sit ahead of live speech
                # in the queue, or the watcher spends the whole (short)
                # playback window draining backlog and never reaches the
                # real interruption (sim 2026-09-03: barge frames answered
                # as turns AFTER playback instead of cutting it).
                mon = monitor_cls()
                stop = {"x": False, "barged": False}
                def _watch():
                    while not stop["x"]:
                        try:
                            fr = io.q.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        pump(fr.reshape(-1), time.monotonic())
                        if (dxm.BARGE_MODE == "stop" and not stop["barged"]
                                and mon.feed(fr)):
                            io.interrupt_playback()
                            stop["barged"] = True
                    return
                w = threading.Thread(target=_watch, daemon=True)
                w.start()
                for ch in tts.synth(reply):
                    if stop["barged"]:
                        break
                    played24.append(ch)
                    io.play_chunk(ch)
                time.sleep(0.1)
                while io.is_playing() and not stop["barged"]:
                    time.sleep(0.02)
                time.sleep(0.1)
                stop["x"] = True
                w.join(timeout=0.5)
                finished = not stop["barged"]
        finally:
            state["speaking"] = False
            state["speak_end"] = time.time()
            if ecapa is not None and played24:
                a16 = resample_poly(np.concatenate(played24),
                                    up=2, down=3).astype(np.float32)
                state["self_vec"] = ecapa.embed(a16)
        return finished

    def respond(user_text, rec, image):
        line = (rec["context_line"] + "\n" + user_text) if rec else user_text
        content = vis.content(line, image)
        reply = get_llm().answer(content, SYSTEM_PROMPT)
        get_llm().remember(user_text, reply)
        print(f"  ⏵ agent: {reply}", flush=True)
        finished = speak(reply)
        if not finished:
            print("  ✂ barged", flush=True)
        state["followup"] = time.time() + (32.0 if reply.rstrip()
                                           .endswith("?") else 20.0)

    nn_hist: list = []

    def remember(user_text, reply):
        nn_hist.append({"role": "user", "content": user_text})
        nn_hist.append({"role": "assistant", "content": reply})
        del nn_hist[:-16]

    def answer_text(user_text, rec=None):
        """ONE text-turn answerer for every non-audio entrance (typed,
        hold expiry/merge, streamed long turns) — same model, same memory."""
        if nn is not None and reply_mode == "nn":
            reply = nn.respond_text(user_text, SYSTEM_PROMPT, nn_hist,
                                    max_tokens=140)
            remember(user_text, reply)
            print(f"  ⏵ agent: {reply}", flush=True)
            speak(reply)
            state["followup"] = time.time() + (32.0 if reply.rstrip()
                                               .endswith("?") else 20.0)
        else:
            respond(user_text, rec, None)

    def transcribe_any(audio):
        a = np.asarray(audio, np.float32).reshape(-1)
        if nn is not None:
            t, _ = nn.transcribe(a)
            return (t or "").strip()
        return (stt.process(a, time.time())["text"] or "").strip()

    def is_self_echo(audio) -> bool:
        """ECAPA self-voice rejection, gap 2: the agent enrolls its OWN
        Kokoro output after each reply and drops sound-alike turns near
        playback (the cascade's thank-you-loop lesson, learned space)."""
        if ecapa is None or state.get("self_vec") is None:
            return False
        dt = time.time() - state.get("speak_end", 0.0)
        if dt > 4.0:
            return False
        v = ecapa.embed(np.asarray(audio, np.float32).reshape(-1))
        cos = float(np.dot(v, state["self_vec"])) if v is not None else -1.0
        return v is not None and cos >= 0.45

    def nn_identity(audio, s_out, w_hit):
        """Trained embeddings replace hand features at the turn seam:
        ECAPA identity overrides the v0 fingerprint; enrolled wake
        templates match in the LEARNED space."""
        if ecapa is None:
            return s_out, w_hit
        lab = ecapa.label_turn(audio)
        if lab["speaker"]:
            s_out = {**s_out, **lab}
        if w_hit is None:
            m = ecapa.wake_match(audio)
            if m is not None:
                w_hit = m[0]
                print(f"  [wake·ecapa] «{m[0]}» sim={m[1]}", flush=True)
        return s_out, w_hit

    def on_utterance_nn_reply(audio, s_out, w_hit):
        if is_self_echo(audio):
            print("  [self-voice dropped · ecapa]", flush=True)
            return
        s_out, w_hit = nn_identity(audio, s_out, w_hit)
        """THE OMNI PATH: one model, one generation — audio embeddings in,
        transcript + answer out. Paralinguistic gating first (QbE wake /
        window); text guards run on HEARD afterwards and can discard the
        reply (a guard trip costs one wasted generation, measured rare)."""
        in_win = time.time() < state["followup"]
        if state["committed"]:            # full tier: turn already streamed
            tail = transcribe_any(audio)
            full = " ".join(state["committed"] + ([tail] if tail else []))
            state["committed"] = []
            if w_hit is None and text_wake(full) is None and not in_win:
                return                    # overheard monologue
            print(f"  ⏵ heard (streamed): {full}", flush=True)
            answer_text(full)
            return
        if not in_win and w_hit is None:
            # QbE wake is ADDITIVE, not authoritative: a miss must NOT
            # suppress the text-wake fallback. The v0 spotter is crude and
            # speaker-biased (its own node notes; probe 2026-09-02 measured
            # best-window edist 0.97–1.9 vs the 0.55 gate — it never fires),
            # so hard-returning on `wake.templates` made enrollment DISABLE
            # the more reliable wake path. When w_hit is set (a trained QbE
            # or ECAPA hit) we never reach here. Otherwise the cheap directed
            # head rejects clearly-overheard audio; the real addressing
            # decision (w_hit OR text_wake on the transcript) is the post-gate.
            if heads is not None:
                # TRAINED addressing: the directed head decides whether
                # unenrolled speech was meant for us — one cheap trunk
                # pass instead of a wasted full generation
                hp = heads.analyze(audio)
                # CONSERVATIVE threshold: smoke 2026-09-02 measured thin
                # margins on synthetic audio (addressed 0.602 vs overheard
                # 0.549 — right direction, weak calibration; the heads were
                # trained on real room audio). 0.35 cannot false-block
                # either measured case; full-strength gating is gated on
                # the per-head eval against real data.
                if hp.get("directed_p", 1.0) < 0.35:
                    print(f"  [overheard · directed_p="
                          f"{hp.get('directed_p')}]", flush=True)
                    return
        raw, met = nn.respond(np.asarray(audio, np.float32).reshape(-1),
                              system=NN_SYSTEM, history=nn_hist,
                              max_tokens=140)
        heard, reply = parse_heard_reply(raw)
        if not heard.strip():
            print("  [unintelligible — dropped]", flush=True)
            return                        # never hold or answer empty audio
        rec = fuse.process({"text": heard, "speaker": s_out,
                            "turn": {"in_turn": False, "turn_ended": True},
                            "wake": w_hit,
                            "duration_s": len(audio) / 16000.0}, time.time())
        print(f"  {rec['context_line']}  {heard}"
              f"   [nn {met['total_s']:.2f}s ttft {met['ttft_s']:.2f}s]",
              flush=True)
        if is_garble(rec):
            print(f"  [garble dropped] {heard[:60]}", flush=True)
            return
        if is_stock(heard, rec.get("wpm")):
            print(f"  [stock artifact dropped] {heard[:40]}", flush=True)
            return
        hit = w_hit or text_wake(heard)
        if state["hold"] and (in_win or hit is not None):
            merged = state["hold"]["text"] + " " + heard
            state["hold"] = None          # regenerate on the WHOLE thought —
            print(f"  ⏵ merged: {merged}", flush=True)
            answer_text(merged)           # one wasted generation, same cost
            return                        # profile as a guard trip
        if hit is None and not in_win:
            return                        # overheard; reply discarded
        if not rec["complete"] and hit is None:
            state["hold"] = {"text": heard, "rec": rec,
                             "deadline": time.time() + 8.0}
            print("  [holding — finish your thought]", flush=True)
            return
        remember(heard or "[speech]", reply)
        print(f"  ⏵ agent: {reply}", flush=True)
        if not speak(reply):
            print("  ✂ barged", flush=True)
        if is_dismissal(heard):
            state["followup"] = 0.0
            nn_hist.clear()
            print("  [conversation closed]", flush=True)
            return
        state["followup"] = time.time() + (32.0 if reply.rstrip()
                                           .endswith("?") else 20.0)

    def on_utterance(audio, s_out, w_hit):
        if nn is not None and reply_mode == "nn":
            return on_utterance_nn_reply(audio, s_out, w_hit)
        s_out, w_hit = nn_identity(audio, s_out, w_hit)
        if nn is not None:
            text, _nn_metrics = nn.transcribe(
                np.asarray(audio, np.float32).reshape(-1))
        else:
            text = stt.process(audio, time.time())["text"]
        if not text:
            return
        rec = fuse.process({"text": text, "speaker": s_out,
                            "turn": {"in_turn": False, "turn_ended": True},
                            "wake": w_hit,
                            "duration_s": len(audio) / 16000.0}, time.time())
        if is_garble(rec):
            print(f"  [garble dropped] {text[:60]}", flush=True)
            return
        hit = w_hit or text_wake(text)
        in_win = time.time() < state["followup"]
        print(f"  {rec['context_line']}  {text}", flush=True)
        if state["hold"] and (in_win or hit):
            text = state["hold"]["text"] + " " + text
            state["hold"] = None
        if hit is None and not in_win:
            return                              # overheard
        if not rec["complete"] and hit is None:
            state["hold"] = {"text": text, "rec": rec,
                             "deadline": time.time() + 8.0}
            print("  [holding — finish your thought]", flush=True)
            return
        if is_dismissal(text):
            respond(text, rec, None)            # farewell THEN close
            state["followup"] = 0.0
            llm.clear()
            print("  [conversation closed]", flush=True)
            return
        image, state["image"] = state["image"], None
        respond(text, rec, image)

    frames_src = None
    if io is None:
        frames_src = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", blocksize=480,
            callback=lambda ind, n, ti, st: q.put(ind[:, 0].copy()))
        frames_src.start()
    try:
        while True:
            typed = txt.poll()
            if typed:
                if state["speaking"] and io is not None and barge == "stop":
                    io.interrupt_playback()
                if typed.strip() == "/quit":
                    print("bye", flush=True)
                    break
                print(f"  you (typed): {typed}", flush=True)
                if (nn is not None and reply_mode == "nn"
                        and state["image"] is None):
                    answer_text(typed)    # gap 1: SAME model and memory
                else:
                    respond(typed, None, state["image"])
                    state["image"] = None
            src_q = io.q if io is not None else q
            if mode == "full" and not state["speaking"] and state["utt"]:
                buf_s = sum(a.size for a in state["utt"]) / 16000.0
                if buf_s >= 2.0 and buf_s - state["preview_s"] >= 1.5:
                    state["preview_s"] = buf_s
                    ptxt = transcribe_any(np.concatenate(state["utt"]))
                    if ptxt:
                        print(f"  ⋯ forming: {ptxt[-70:]}", flush=True)
                if buf_s >= STREAM_COMMIT_S:          # commit-and-trim
                    cur = np.concatenate(state["utt"])
                    head, tail = stream_split(cur)
                    ctxt = transcribe_any(head) if head.size else ""
                    if ctxt:
                        state["committed"].append(ctxt)
                        print(f"  ▸ committed: {ctxt}", flush=True)
                    state["utt"][:] = [tail]
                    state["preview_s"] = STREAM_KEEP_S
            if (state["hold"]
                    and time.time() > state["hold"]["deadline"]):
                h, state["hold"] = state["hold"], None
                if h["text"].strip():
                    print("  [hold expired — answering]", flush=True)
                    answer_text(h["text"], h["rec"])  # review #6: never leak
                else:
                    print("  [hold expired — nothing held]", flush=True)
            if state["speaking"]:
                # The barge watcher inside speak() is the SOLE owner of the
                # duplex queue during playback. The main loop must NOT race
                # it for io.q — a stolen-and-discarded frame is one the
                # BargeMonitor never sees, and enough of them starve barge
                # detection entirely (sim 2026-09-02: 0 frames reached the
                # watcher). Half-duplex has no watcher and no io, so its
                # queue still drains here.
                if io is not None:
                    time.sleep(0.02)
                    continue
                try:
                    src_q.get(timeout=0.2)      # half: discard while speaking
                except queue.Empty:
                    pass
                continue
            try:
                fr = src_q.get(timeout=0.2)
            except queue.Empty:
                drain_pending()
                continue
            pump(np.asarray(fr).reshape(-1), time.monotonic())
            drain_pending()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        if frames_src is not None:
            frames_src.stop()
        if io is not None:
            io.__exit__(None, None, None)
        stat_nodes = [("para-speaker-embed-v0", spk),
                      ("para-turn-state-v0", turn),
                      ("para-wake-spotter-v0", wake),
                      ("fusion-context-v0", fuse)]
        if stt is not None:
            stat_nodes.append(("semantic-whisper-v0", stt))
        for nid, n in stat_nodes:
            write_stats(nid, n)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--enroll", nargs=2, metavar=("NAME", "WAV"))
    ap.add_argument("--wav")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--no-stt", action="store_true",
                    help="parallel channels only — no whisper load")
    ap.add_argument("--converse", choices=["half", "quasi", "full"],
                    help="run the agent on the node graph: half = mic "
                         "closed while speaking; quasi = open mic + AEC + "
                         "barge; full = quasi + streaming (live forming "
                         "preview, long turns commit mid-speech)")
    ap.add_argument("--barge", choices=["stop", "continue"], default="stop")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--reply", choices=["nn", "cascade"], default="nn",
                    help="nn = the SAME Gemma answers the audio embeddings "
                         "directly, one generation (probed: 1.4–1.6 s warm "
                         "vs ~5.5 s cascade); cascade = transcript -> "
                         "second Gemma")
    ap.add_argument("--semantic", choices=["nn", "text"], default="nn",
                    help="nn = the trained adapter path (DEFAULT — the point "
                         "of this folder); text = whisper-decoder fallback")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.converse:
        return converse(args.converse, args.barge, args.vision,
                        args.semantic, args.reply)
    if args.enroll:
        name, wav = args.enroll
        audio = np.concatenate(wav_frames(wav))
        wk = load_node("para-wake-spotter-v0")
        wk.enroll(name, audio)
        try:
            ec = load_node("para-speaker-ecapa-v1")
            ec.load()
            ec.enroll(name, audio)
            print(f"enrolled «{name}» in BOTH spaces: hand-feature "
                  f"(frame-rate latch) + ECAPA (learned, turn-level)")
        except Exception as exc:
            print(f"enrolled «{name}» (hand features only — ECAPA: {exc})")
        return 0
    if args.wav:
        run_stream(wav_frames(args.wav), with_stt=not args.no_stt)
        return 0
    if args.mic:
        run_stream(mic_frames(), with_stt=not args.no_stt)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
