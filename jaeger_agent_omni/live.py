"""Live mic -> Gemma -> voice. The whole pipeline, on real hardware.

    python -m jaeger_agent_omni.live                 # hands-free, speak to it
    python -m jaeger_agent_omni.live --seconds 60
    python -m jaeger_agent_omni.live --voice am_michael --speed 1.15
    python -m jaeger_agent_omni.live --list-devices

    audio ──► selected neural ear ──► Gemma E4B ──► text ──► Kokoro ──► speaker
              no transcription or ASR decoder anywhere on the way in

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
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"
RATE = 16000
BLOCK = 1600            # 100 ms, the cadence the duplex loop expects
MEETING_SYSTEM = (
    "You are a silent third-person meeting observer. Track the discussion by "
    "speaker label. After each completed audio turn, return compact cumulative "
    "meeting minutes with: topics, decisions, action items with owners when "
    "known, and open questions. Preserve concrete facts; do not greet the room "
    "or claim something was decided when it was only discussed."
)


def _ready_chunks(text: str, consumed: int) -> tuple[list[str], int]:
    """Return newly speakable clauses and the updated character offset.

    Sentence punctuation is always a boundary.  A comma, semicolon, colon, or
    em dash becomes a boundary after five words. Some adapter replies contain
    no punctuation, so a whitespace boundary after twelve words provides a
    final latency bound without cutting through a word or feeding Kokoro
    one-word scraps.
    """
    chunks = []
    start = consumed
    for i in range(consumed, len(text)):
        terminal = text[i] in ".!?"
        clause = text[i] in ",;:—"
        words = len(text[start:i + 1].split())
        cadence = text[i].isspace() and words >= 12
        if ((terminal and words >= 2) or (clause and words >= 5) or
                cadence):
            chunk = text[start:i + 1].split("<")[0].strip()
            if chunk:
                chunks.append(chunk)
            start = i + 1
    return chunks, start


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
    from jaeger_agent_omni.core.model_catalog import front_end_choices

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", type=int, default=None, help="input device id")
    ap.add_argument("--model", default=MODEL)
    from jaeger_agent_omni.core.model_catalog import DEFAULT_FRONT_END
    ap.add_argument("--front-end", default=DEFAULT_FRONT_END,
                    choices=front_end_choices(),
                    help="canonical Jaeger audio-ear identity")
    ap.add_argument("--adapter", default=None,
                    help="explicit research override for the selected stock "
                         "node (reported as custom)")
    ap.add_argument("--threshold", type=float, default=0.012,
                    help="barge-in RMS floor and legacy --vad energy floor")
    ap.add_argument("--vad", choices=("neural", "energy"), default="neural",
                    help="turn-opening gate; neural uses the packaged Silero "
                         "network, energy is the legacy RMS fallback")
    ap.add_argument("--vad-open", type=float, default=0.30,
                    help="Silero probability needed to open a speech turn")
    ap.add_argument("--vad-close", type=float, default=0.125,
                    help="Silero probability that holds an open turn")
    ap.add_argument("--silence-ms", type=float, default=510.0,
                    help="quiet needed to close a turn; a continuous "
                         "conversation never reaches the default")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-speak", action="store_true",
                    help="print replies instead of speaking them")
    ap.add_argument("--max-context", type=int, default=6000,
                    help="cache tokens before the session ROLLS: audio costs "
                         "25 tok/s, so unbounded listening slows every "
                         "subsequent prefill and eventually overflows. "
                         "Rolling keeps the model in real time forever.")
    ap.add_argument("--cooldown", type=float, default=4.0,
                    help="seconds after a reply before another turn may "
                         "close in direct-assistant mode; companion uses its "
                         "semantic RESPOND/LISTEN decision instead")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="record raw mic AND model-fed audio for audit")
    ap.add_argument("--no-aec", action="store_true",
                    help="disable SpeexDSP acoustic echo cancellation")
    ap.add_argument("--aec-reference-mode",
                    choices=("wallclock", "contiguous"), default="wallclock",
                    help="EXPERIMENTAL: contiguous feeds Speex one unbroken "
                         "reply-level far-end stream (clauses plus real "
                         "silence) instead of per-block wall-clock slices. "
                         "wallclock remains the measured baseline until a "
                         "physical A/B supports promotion.")
    ap.add_argument("--barge-policy", choices=("stop", "continue"),
                    default="stop",
                    help="action after speech-during-output notification")
    ap.add_argument("--interaction-mode",
                    choices=("assistant", "companion"), default="assistant",
                    help="assistant answers every endpointed turn; companion "
                         "uses Gemma's audio-native RESPOND/LISTEN decision")
    ap.add_argument("--duplex-mode", choices=("half", "full"),
                    default="full",
                    help="half ignores microphone audio while generating or "
                         "speaking; full keeps listening through AEC")
    ap.add_argument("--speakers", type=int, default=0, metavar="K",
                    help="diarize up to K room talkers and append inline "
                         "[speaker A/B] context tags (0 disables)")
    ap.add_argument("--meeting-minutes", metavar="PATH",
                    help="silent third-person observer; write native "
                         "audio-to-minutes output at the end (no transcript)")
    ap.add_argument("--agentic", action="store_true",
                    help="enable the bundled JaegerAgent tool/skill/memory "
                         "loop over the same native multimodal Gemma cache")
    ap.add_argument("--agentic-workspace", metavar="DIR",
                    help="agent memory/skills/workspace root (default: "
                         "<cwd>/.jaeger_agent_omni)")
    ap.add_argument("--report", metavar="PATH",
                    help="JSON run report (default: GUI runtime/reports)")
    a = ap.parse_args()

    if a.meeting_minutes:
        a.no_speak = True
        if not a.speakers:
            a.speakers = 2

    if a.list_devices:
        return list_devices()

    from jaeger_agent_omni.omni import SYSTEM, build
    from jaeger_agent_omni.core.runtime.companion import (
        CompanionState, parse_decision, streaming_response, system_prompt,
    )
    from jaeger_agent_omni.core.runtime.duplex import DuplexLoop
    from jaeger_agent_omni.core.runtime.neural_vad import configure_endpoint
    from jaeger_agent_omni.voice import speak_kokoro

    companion = CompanionState() if a.interaction_mode == "companion" else None
    selected_system = MEETING_SYSTEM if a.meeting_minutes else (
        system_prompt() if companion is not None else SYSTEM)
    agentic_preparation = None
    if a.agentic:
        if a.meeting_minutes:
            raise SystemExit(
                "--agentic and --meeting-minutes are separate modes; meeting "
                "minutes must remain a silent observation artifact")
        from jaeger_agent_omni.core.agentic.omni_runtime import (
            prepare_agentic_system,
        )
        agentic_preparation = prepare_agentic_system(selected_system)
        selected_system = agentic_preparation.system

    print("loading Gemma…", flush=True)
    session, meta = build(
        a.model, a.front_end, a.adapter,
        system=selected_system)
    session.warmup()
    agentic_runtime = None
    if agentic_preparation is not None:
        from jaeger_agent_omni.core.agentic.omni_runtime import OmniAgentRuntime
        agentic_runtime = OmniAgentRuntime(
            session, agentic_preparation, eos_ids=meta["eos"],
            workspace=a.agentic_workspace)
        session.respond = agentic_runtime.respond_native
        # DuplexLoop calls session.cancel() on barge-in. Route that signal to
        # both the native token generator and the outer agent/tool loop so an
        # interrupted tool-capable turn cannot advance to another iteration.
        session.cancel = agentic_runtime.interrupt
        print(f"agentic  bundled JaegerAgent · "
              f"{len(agentic_runtime.agent.tools)} active tools · skills/memory on",
              flush=True)

    from jaeger_agent_omni.core.runtime.reporting import (
        default_report_path, finish_run_report, latency_summary,
        memory_summary, new_run_report, write_report)
    input_name = (str(sd.query_devices(a.device, "input")["name"])
                  if a.device is not None else
                  str(sd.query_devices(kind="input")["name"]))
    report_path = Path(a.report) if a.report else default_report_path(
        Path("Full DuplexDemo App/runtime"), meta["model_id"],
        "physical-room")
    run_report = new_run_report(
        phase="physical-room", model_id=meta["model_id"],
        model_name=meta["model_name"], checkpoint=meta["checkpoint"],
        source=input_name,
        configuration={
            "seconds": a.seconds, "threshold": a.threshold,
            "vad": a.vad, "vad_open": a.vad_open,
            "vad_close": a.vad_close,
            "silence_ms": a.silence_ms, "cooldown_s": a.cooldown,
            "max_context": a.max_context, "speakers": a.speakers,
            "voice": a.voice, "speed": a.speed,
            "barge_policy": a.barge_policy,
            "interaction_mode": a.interaction_mode,
            "duplex_mode": a.duplex_mode,
            "agentic": bool(a.agentic),
            # Meeting-minutes mode is deliberately a silent observer. AEC is
            # only meaningful while Jaeger is playing a known far-end signal;
            # reporting a requested-but-inactive canceller as active made the
            # physical soak artifact contradict its metrics section.
            "observer_mode": bool(a.meeting_minutes),
            "aec_requested": not a.no_aec,
            "aec": (not a.no_aec and not a.no_speak and
                    a.duplex_mode == "full"),
            "aec_reference_mode": a.aec_reference_mode,
        })

    if not a.no_speak:
        print("warming the voice…", flush=True)
        speak_kokoro("ready", "/tmp/_warm.wav", a.voice, a.speed)

    # The mic thread NEVER blocks the duplex tick: it drops into a queue and
    # the loop drains whatever has arrived. A blocking source stalls the whole
    # loop, which is how an earlier test read 2 chunks of 35.
    q: queue.Queue = queue.Queue(maxsize=200)
    speaking_out = threading.Event()
    generating = threading.Event()
    stop = threading.Event()
    barge_request = threading.Event()
    barge_active = threading.Event()
    barge_latched = threading.Event()
    level = {"rms": 0.0, "raw_rms": 0.0}
    heard = {"n": 0}
    overflow = {"blocks": 0, "samples": 0}
    half_muted = {"blocks": 0, "samples": 0}
    barged = {"events": 0, "stops": 0}
    echo_floor = {"rms": 0.35}
    rec = {"raw": [], "fed": []}
    turn_facts = {"self_speech_overlap": False}
    generation_speech = None
    speaker_recent = deque(maxlen=max(1, round(3 * RATE / BLOCK)))
    speaker_tags = []
    perception_turns: list[dict] = []
    run_errors: list[str] = []
    capture_health = {'last_callback': None, 'failure': None, 'callbacks': 0}
    roll_requested = threading.Event()

    from jaeger_agent_omni.core.runtime.interaction import InteractionPolicy
    interaction_policy = InteractionPolicy(a.barge_policy)

    speaker_tracker = None
    if a.speakers:
        from jaeger_agent_omni.core.runtime.speaker import OnlineSpeakerTracker
        speaker_tracker = OnlineSpeakerTracker(a.speakers)
        print(f"speakers online CampPlus, up to {a.speakers} room talkers",
              flush=True)

    aec = None
    barge_detector = None
    if (not a.no_aec and not a.no_speak and
            a.duplex_mode == "full"):
        from jaeger_agent_omni.core.runtime.aec import (
            EchoAwareBargeIn, SpeexEchoCanceller,
        )
        aec = SpeexEchoCanceller(RATE, reference_mode=a.aec_reference_mode)
        barge_detector = EchoAwareBargeIn(RATE, a.threshold)
        print(f"AEC      SpeexDSP ({aec.library_path}) "
              f"reference={a.aec_reference_mode}", flush=True)

    def notify_barge(stage: str) -> None:
        """Record speech during output, then apply policy independently."""
        if barge_latched.is_set():
            return
        barge_latched.set()
        action = interaction_policy.speech_during_output(
            stage, raw_rms=level["raw_rms"], clean_rms=level["rms"],
            raw_reference_correlation=getattr(
                aec, "raw_reference_correlation", 0.0),
            residual_reference_correlation=getattr(
                aec, "residual_reference_correlation", 0.0))
        barged["events"] += 1
        print(f"\n  (speech during agent output: policy={action})", flush=True)
        if action == "stop":
            barge_active.set()
            barge_request.set()

    def on_block(indata, frames, t, status):
        capture_health['last_callback'] = time.monotonic()
        capture_health['callbacks'] += 1
        if capture_health['failure'] is not None:
            return
        try:
            process_block(indata, frames, t, status)
        except Exception as exc:
            capture_health['failure'] = f'{type(exc).__name__}: {exc}'
            run_errors.append('audio callback failed: ' + capture_health['failure'])

    def process_block(indata, frames, t, status):
        if status:
            run_errors.append(f"audio callback: {status}")
        raw = indata[:, 0] if indata.ndim > 1 else indata
        raw = raw.copy().astype("float32")
        neural_speech = generation_speech.update(raw) if generation_speech is not None else None
        clean = aec.process(raw) if aec is not None else raw.copy()
        raw_rms = float(np.sqrt(np.mean(np.square(raw))))
        clean_rms = float(np.sqrt(np.mean(np.square(clean))))
        level["raw_rms"], level["rms"] = raw_rms, clean_rms
        if a.record:
            rec["raw"].append(raw.copy())
        if speaking_out.is_set() and a.duplex_mode == "full":
            turn_facts["self_speech_overlap"] = True

        if (a.duplex_mode == "half" and
                (speaking_out.is_set() or generating.is_set())):
            # Half duplex is explicit and measurable: capture remains alive,
            # but model input is paused until the agent finishes.  This gives
            # perfect self-rejection at the cost of zero interruptibility.
            half_muted["blocks"] += 1
            half_muted["samples"] += len(raw)
            if a.record:
                rec["fed"].append(np.zeros_like(raw))
            return

        if speaking_out.is_set():
            if aec is None:
                # Preserve the old safe fallback when SpeexDSP is explicitly
                # disabled: do not feed the agent its own uncancelled voice.
                is_barge = raw_rms > max(echo_floor["rms"] * 1.4,
                                         a.threshold * 1.1)
                if not is_barge:
                    echo_floor["rms"] = 0.95 * echo_floor["rms"] + 0.05 * raw_rms
                    clean = np.zeros_like(raw)
                    clean_rms = 0.0
            else:
                # Energy alone cannot distinguish a listener from uncancelled
                # reverberation. Residual still correlated with the exact
                # Kokoro waveform is Jaeger's own voice; decorrelated residual
                # above the learned echo floor is genuine double-talk.
                is_barge = barge_detector.update(aec, clean_rms, len(raw))
            if is_barge:
                notify_barge("playback")
        elif generating.is_set() and (
                neural_speech if neural_speech is not None else
                clean_rms > a.threshold * 1.1):
            # A loud background is not a continuation. In neural mode this
            # evidence comes from a separate raw-audio NN running while the
            # endpoint/inference loop is occupied, before any playback exists.
            notify_barge("generation")

        level["rms"] = clean_rms
        if a.record:
            rec["fed"].append(clean.copy())
        if speaker_tracker is not None:
            speaker_recent.append(clean.copy())
        try:
            q.put_nowait(clean)
        except queue.Full:
            # Distinct from the duplex loop's lag-based drops: this audio
            # never reached the loop at all.  Count it so "zero dropped"
            # in a report is airtight.
            overflow["blocks"] += 1
            overflow["samples"] += len(clean)

    def source():
        if capture_health['failure'] is not None:
            raise RuntimeError('Audio capture failed: ' + capture_health['failure'])
        last = capture_health['last_callback']
        if last is not None and time.monotonic() - last > 2.0:
            raise RuntimeError('Audio device stopped delivering frames for 2 seconds')
        # This runs on the model owner thread, between encode/decode calls.
        # The monitoring thread may request a roll but must never reset a
        # cache while the model is using it.
        if (roll_requested.is_set() and not speaking_out.is_set()
                and not generating.is_set() and not loop.endpoint.speaking):
            roll_requested.clear()
            n = session.n_prefilled
            if n > a.max_context:
                summary = ' '.join(recent_replies[-2:])[:400]
                session.reset()
                session.open_turn()
                if summary:
                    session.feed_text(f' (ongoing conversation; my recent replies: {summary}) ')
                soak['rolls'] += 1
                print(f"\n  [soak] ROLLED context at {n} tok -> "
                      f"{session.n_prefilled} (roll #{soak['rolls']})", flush=True)
        got = []
        while True:
            try:
                got.append(q.get_nowait())
            except queue.Empty:
                break
        if not got:
            return None
        block = np.concatenate(got)
        return block

    def user_speaking():
        return barge_active.is_set()

    def tag_speaker():
        if speaker_tracker is None or not speaker_recent:
            return None
        audio = np.concatenate(tuple(speaker_recent))
        speaker_recent.clear()
        label = speaker_tracker.identify(audio)
        if label is None:
            return None
        tag = f"[speaker {label}]"
        session.feed_text(f" {tag}")
        speaker_tags.append(label)
        print(f"\n          [context: {tag}]", flush=True)
        return label

    def prepare_turn():
        label = tag_speaker()
        perception = getattr(session, "perception_snapshot", lambda: {})()
        scene = perception.get("scene", {})
        if scene.get("calibrated"):
            scene_tag = (
                f" [audio_scene={scene['class']}; "
                f"confidence={scene['probability']:.3f}]"
            )
            session.feed_text(scene_tag)
            print(f"\n          [context: {scene_tag.strip()}]", flush=True)
        if companion is not None:
            metadata = companion.turn_metadata(
                label, self_speech_overlap=turn_facts["self_speech_overlap"])
            session.feed_text(metadata)
            print(f"\n          [context: {metadata.strip()}]", flush=True)
        turn_facts["self_speech_overlap"] = False

    stage_ms = {"synth": [], "reply_len": [], "to_first_audio": []}
    turn_t0 = {"t": None}          # when the turn closed (speech_end)

    import queue as _q
    say_q: _q.Queue = _q.Queue()
    wav_q: _q.Queue = _q.Queue(maxsize=2)
    from jaeger_agent_omni.core.runtime.speech_delivery import SpeechDelivery
    delivery = SpeechDelivery()

    def _drain(q):
        while not q.empty():
            try:
                q.get_nowait()
            except _q.Empty:
                break

    def _barge_watcher():
        while not stop.is_set():
            if not barge_request.wait(0.01):
                continue
            barge_request.clear()
            with delivery.lock:
                delivery.cancel_pending('barge')
                was_playing = speaking_out.is_set()
                if was_playing:
                    sd.stop()
                    speaking_out.clear()
                _drain(say_q)
                _drain(wav_q)
            if was_playing:
                barged["stops"] += 1
                print("\n  (barge-in: residual speech — stopping)", flush=True)

    barge_thread = threading.Thread(target=_barge_watcher, daemon=True)
    barge_thread.start()

    def _synth_worker():
        """Stage 1 of the mouth: synthesise clause N+1 WHILE clause N is
        still playing.  The old single-thread player left a full Kokoro
        synthesis of dead air between clauses."""
        import soundfile as sf
        while True:
            sent = say_q.get()
            if sent is None:
                wav_q.put(None)
                break
            reply_id, sent = sent
            if not delivery.valid(reply_id):
                continue
            try:
                t0 = time.monotonic()
                speak_kokoro(sent, "/tmp/_sent.wav", a.voice, a.speed)
                stage_ms["synth"].append((time.monotonic() - t0) * 1000)
                wav, sr = sf.read("/tmp/_sent.wav", dtype="float32")
                if delivery.valid(reply_id):
                    wav_q.put((reply_id, wav, sr))
            except Exception as exc:
                delivery.fail(reply_id, exc)
                run_errors.append(
                    f"speech synthesis: {type(exc).__name__}: {exc}")

    def _player():
        """Stage 2: sequential playback of pre-synthesised clauses.  The AEC
        reference is registered here, immediately before output, so playback
        timing — not synthesis timing — anchors the far-end timeline."""
        while True:
            item = wav_q.get()
            if item is None:
                break
            reply_id, wav, sr = item
            try:
                with delivery.lock:
                    if not delivery.valid(reply_id):
                        continue
                    if aec is not None:
                        if aec.reference_mode == "contiguous":
                            aec.append_playback(wav, sr)
                        else:
                            aec.start_playback(wav, sr)
                    speaking_out.set()
                    if turn_t0["t"] is not None:
                        stage_ms["to_first_audio"].append(
                            (time.monotonic() - turn_t0["t"]) * 1000)
                        print(f"\n          [latency: turn-end -> first audio "
                              f"{stage_ms['to_first_audio'][-1]:.0f} ms]",
                              flush=True)
                        turn_t0["t"] = None
                    sd.play(wav, sr)
                playback_status = sd.wait()
                if playback_status:
                    raise RuntimeError(f'Audio playback buffer failure: {playback_status}')
                delivery.played(reply_id)
            except Exception as exc:
                delivery.fail(reply_id, exc)
                run_errors.append(
                    f"speech output: {type(exc).__name__}: {exc}")
            finally:
                speaking_out.clear()

    synth_thread = threading.Thread(target=_synth_worker, daemon=True)
    synth_thread.start()
    player_thread = threading.Thread(target=_player, daemon=True)
    player_thread.start()

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
        if (companion is None and
                time.monotonic() - last_reply["t"] < a.cooldown):
            return False
        return loop.endpoint.ended if loop.endpoint else False

    tok = session.processor.tokenizer
    _orig_respond = session.respond

    def _streaming_respond(**kw):
        reply_id = delivery.begin()
        def queue_clause(text):
            if delivery.enqueue(reply_id):
                say_q.put((reply_id, text))
        turn_t0["t"] = time.monotonic()
        barge_active.clear()
        barge_latched.clear()
        if aec is not None:
            # One reply = one playback generation (no-op in wallclock mode).
            aec.begin_reply_playback()
        generating.set()
        ids, consumed = [], 0
        inner = kw.get("on_token")

        def on_token(tid, n):
            nonlocal consumed
            # An agentic turn may first emit executable tool markup. Buffer
            # until the copied loop has parsed/dispatched it and returned the
            # final answer; otherwise Kokoro could literally speak a command.
            if not getattr(session, "agentic_buffering", False):
                ids.append(tid)
                # Decode is cheap at this length. Clause boundaries keep a
                # long first sentence from delaying all audio until its final
                # period.
                text = tok.decode(ids)
                if companion is not None:
                    text = streaming_response(text)
                chunks, consumed = _ready_chunks(text, consumed)
                if not a.no_speak:
                    for chunk in chunks:
                        queue_clause(chunk)
            if inner:
                inner(tid, n)

        kw["on_token"] = on_token
        try:
            raw_out = _orig_respond(**kw)
        except BaseException as exc:
            delivery.fail(reply_id, exc)
            delivery.close(reply_id)
            raise
        finally:
            generating.clear()
        if companion is not None:
            decision = parse_decision(raw_out)
            companion.observe(decision)
            out = decision.response if decision.should_respond else ""
            verdict = "RESPOND" if decision.should_respond else "LISTEN"
            detail = decision.reason or "addressed"
            print(f"\n          [companion: {verdict} · {detail}]", flush=True)
            if not decision.should_respond:
                turn_t0["t"] = None
        else:
            out = raw_out
        snapshot = getattr(session, "perception_snapshot", lambda: {
            "available": False, "controls_policy": False})()
        perception_turns.append(snapshot)
        tail = out[consumed:].split("<")[0].strip()
        if not a.no_speak and tail:
            queue_clause(tail)
        delivery.close(reply_id)
        return out

    if not a.no_speak or companion is not None:
        session.respond = _streaming_respond

    loop = DuplexLoop(session, audio_source=source, user_speaking=user_speaking,
                      on_reply=on_reply, turn_ended=None,
                      # Always offer the selected neural ear a chance to append
                      # calibrated, non-transcript scene metadata.  Ears with
                      # no perception head make this a no-op.
                      before_respond=prepare_turn,
                      on_endpoint_event=(
                          lambda event: companion.speech_started()
                          if companion is not None and event == "speech_start"
                          else None))
    _default_ended = loop.turn_ended
    loop.turn_ended = turn_gate
    loop.endpoint.silence_ms = a.silence_ms
    vad_label = configure_endpoint(
        loop.endpoint, mode=a.vad, energy_threshold=a.threshold,
        open_threshold=a.vad_open, close_threshold=a.vad_close)
    if a.vad == "neural" and a.duplex_mode != "half":
        from jaeger_agent_omni.core.runtime.neural_vad import load_silero, SpeechPresence
        engine, package = load_silero()
        generation_speech = SpeechPresence(
            engine.probability, frame_samples=int(package.manifest["frame_samples"]),
            threshold=a.vad_open)

    def meter():
        while not stop.is_set():
            r = level["rms"]
            n = int(min(1.0, r / 0.08) * 22)
            state = "AEC OUT " if speaking_out.is_set() and aec else (
                "SPEAKING" if speaking_out.is_set() else (
                "hearing " if loop.endpoint.speaking else "quiet   ")
            )
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

    print(f"\ndevice   {input_name}")
    node_label = f" · {meta['node']}" if meta.get("node") else ""
    print(f"input    {meta['model_name']} @{meta['checkpoint']}{node_label}")
    print(f"voice    {a.voice} @ {a.speed}x")
    print(f"policy   {a.interaction_mode} · {a.duplex_mode} duplex")
    print(f"VAD      {vad_label} · {loop.endpoint.frame_ms:.0f}ms frames")
    print(f"listening for {a.seconds:.0f}s — speak, pause, and it will answer.")
    print("talk over the reply to test barge-in.\n")

    # ── REAL-TIME DRIFT WATCH + ROLLING CONTEXT ──────────────────────────
    # The cache grows 25 tokens per second of audio heard. Attention cost
    # grows with it, so prefill-per-chunk creeps up until the loop cannot
    # keep pace -- the model "falls behind" not from load spikes but from
    # its own accumulated history. The soak stats make the creep visible;
    # the roll resets the cache BEFORE it matters, carrying recent replies
    # as text so conversational context survives the trim.
    soak = {"rows": [], "rolls": 0, "last": None}
    recent_replies: list = []
    all_replies: list[str] = []
    meeting_notes: list[str] = []

    _orig_on_reply = on_reply

    def on_reply_keep(text):
        t = text.split("<")[0].strip()
        if t:
            recent_replies.append(t)
            del recent_replies[:-3]
            all_replies.append(t)
            if a.meeting_minutes:
                meeting_notes.append(t)
        _orig_on_reply(text)

    def soak_watch():
        import time as _t
        while not stop.is_set():
            _t.sleep(15)
            if stop.is_set():
                break
            n = session.n_prefilled
            total_n = session.total_prefill_tokens
            pre_ms = session.total_prefill_ms
            now = _t.monotonic()
            if soak["last"] is not None:
                dn = total_n - soak["last"][0]
                dms = pre_ms - soak["last"][1]
                if dn > 0:
                    rate = dms / dn
                    soak["rows"].append((n, rate))
                    print(f"\n  [soak] cache {n} tok · prefill "
                          f"{rate:.1f} ms/tok · lag {stats_lag():.0f} ms",
                          flush=True)
            soak["last"] = (total_n, pre_ms, now)
            if n > a.max_context:
                roll_requested.set()

    def stats_lag():
        return loop.stats.max_lag_s * 1000

    threading.Thread(target=soak_watch, daemon=True).start()
    loop.on_reply = on_reply_keep

    threading.Thread(target=meter, daemon=True).start()
    t0 = time.monotonic()
    # Crash-only teardown: a wedged output device can deadlock PortAudio
    # stream close (measured 2026-08-29: a healthy 300 s run finished, then
    # parked forever inside the cffi close call, losing its report and
    # recordings).  So the stream is closed LAST, after every artifact is on
    # disk, and a hung close forces exit rather than hanging the session.
    mic_stream = sd.InputStream(samplerate=RATE, blocksize=BLOCK, channels=1,
                                dtype="float32", device=a.device,
                                callback=on_block)
    try:
        capture_health['last_callback'] = time.monotonic()
        mic_stream.start()
        stats = loop.run(eos_ids=meta["eos"], max_seconds=a.seconds)
    except Exception as exc:                               # noqa: BLE001
        # A worker exception must still produce the report and recordings —
        # a five-minute physical run is too expensive to lose to one bug.
        run_errors.append(f"duplex loop: {type(exc).__name__}: {exc}")
        stats = loop.stats
    finally:
        stop.set()
        delivery.cancel_pending('shutdown')
    threading.Thread(target=sd.stop, daemon=True).start()
    speaking_out.clear()
    while not say_q.empty():
        try:
            say_q.get_nowait()
        except _q.Empty:
            break
    say_q.put(None)
    synth_thread.join(timeout=5.0)
    player_thread.join(timeout=5.0)
    barge_thread.join(timeout=0.5)
    for worker_name, worker in (('synthesis', synth_thread), ('playback', player_thread),
                                ('barge monitor', barge_thread)):
        if worker.is_alive():
            run_errors.append(f'{worker_name} worker did not stop before the shutdown deadline')

    minutes = None
    if a.meeting_minutes:
        # Final partial audio is still an audio turn.  Flush it, append only the
        # permitted inline diarization tag, and let Gemma produce the artifact
        # directly from its audio cache.  There is no ASR/transcript handoff.
        session.end_of_turn(close=False)
        tag_speaker()
        session.close_turn()
        minutes = session.respond(max_tokens=256, eos_ids=meta["eos"])
        artifact = "# Meeting minutes\n\n" + minutes.strip()
        if meeting_notes:
            artifact += "\n\n## Running notes\n\n" + "\n\n".join(
                f"### Note {i}\n\n{note}" for i, note in
                enumerate(meeting_notes, 1))
        Path(a.meeting_minutes).write_text(artifact.strip() + "\n")
        print(f"\n\n  MEETING MINUTES -> {a.meeting_minutes}\n{artifact.strip()}",
              flush=True)

    wall = time.monotonic() - t0
    print(f"\n\n  ran {wall:.0f}s")
    print(f"  turns        {stats.turns}     interrupts {stats.interrupts}")
    fed_s = stats.audio_samples / RATE
    dropped_s = stats.audio_samples_dropped / RATE
    print(f"  audio        {stats.audio_chunks} model chunks · "
          f"{fed_s:.1f}s fed of {wall:.0f}s · {dropped_s:.1f}s dropped")
    if overflow["blocks"]:
        print(f"  mic queue    OVERFLOW {overflow['blocks']} blocks · "
              f"{overflow['samples']/RATE:.1f}s never reached the loop")
    if half_muted["blocks"]:
        print(f"  half duplex  {half_muted['samples']/RATE:.1f}s intentionally "
              "not fed while Jaeger was active")
    print(f"  endpoint     {' -> '.join(stats.endpoint_events[:8]) or '(none)'}")
    print(f"  peak lag     {stats.max_lag_s*1000:.0f} ms")
    print(f"  replies      {heard['n']}    speech-during-output events "
          f"{barged['events']} · policy stops {barged['stops']}")
    if speaker_tags:
        counts = ", ".join(f"{label}:{speaker_tags.count(label)}"
                           for label in sorted(set(speaker_tags)))
        print(f"  speakers     {counts} turns tagged inline")
    if soak["rows"]:
        first, last = soak["rows"][0], soak["rows"][-1]
        print(f"  drift        prefill {first[1]:.1f} -> {last[1]:.1f} ms/tok "
              f"across {last[0]} cache tokens · rolls {soak['rolls']}")
    if stage_ms["synth"]:
        import statistics as st
        print(f"  synth        median {st.median(stage_ms['synth']):.0f} ms")
    if stage_ms["to_first_audio"]:
        import statistics as st
        print(f"  END->AUDIO   median "
              f"{st.median(stage_ms['to_first_audio']):.0f} ms  "
              f"(the latency the user feels)")
    if aec is not None and aec.samples_cancelled:
        print(f"  AEC           {aec.attenuation_db:.1f} dB aggregate echo reduction "
              f"across {aec.samples_cancelled/RATE:.1f}s of playback/tail")
    if a.record and rec["raw"]:
        import soundfile as sf
        d = Path(a.record)
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / "raw_mic.wav"), np.concatenate(rec["raw"]), RATE)
        sf.write(str(d / "fed_model.wav"), np.concatenate(rec["fed"]), RATE)
        print(f"  recorded     {d}/raw_mic.wav + fed_model.wav — audit with"
              f" research/session_audit.py")
    agentic_calls = []
    if agentic_runtime is not None:
        agentic_calls = [
            str(call.get("name") or "")
            for message in agentic_runtime.agent.messages
            for call in (message.get("tool_calls") or [])
            if call.get("name")
        ]
    run_report = finish_run_report(
        run_report,
        metrics={
            "wall_seconds": wall, "fed_seconds": fed_s,
            "dropped_seconds": dropped_s, "audio_chunks": stats.audio_chunks,
            # Loop-lag drops (above) and mic-queue overflow (below) are
            # different failures: the first reached the loop too late, the
            # second never reached it at all.
            "mic_queue_overflow_blocks": overflow["blocks"],
            "mic_queue_overflow_seconds": overflow["samples"] / RATE,
            "capture_health": dict(capture_health),
            "half_duplex_muted_seconds": half_muted["samples"] / RATE,
            "memory": memory_summary(),
            "turns": stats.turns, "completed_replies": heard["n"],
            "speech_delivery": delivery.summary(),
            "interrupts": stats.interrupts,
            "barge_events": barged["events"],
            "barge_stops": barged["stops"],
            "context_rolls": soak["rolls"],
            "interrupt_rewind_failures": getattr(
                session, "rewind_failures", 0),
            "final_cache_tokens": session.n_prefilled,
            "peak_loop_lag_ms": stats.max_lag_s * 1000,
            "endpoint_events": list(stats.endpoint_events),
            "vad": loop.endpoint.probability_summary(),
            "generation_barge_vad": (generation_speech.summary()
                                     if generation_speech is not None else None),
            "end_to_audio": latency_summary(stage_ms["to_first_audio"]),
            "synthesis": latency_summary(stage_ms["synth"]),
            "aec": {
                "available": aec is not None,
                "reference_mode": (
                    aec.reference_mode if aec is not None else None),
                "attenuation_db": (
                    aec.attenuation_db if aec is not None else None),
                "seconds_cancelled": (
                    aec.samples_cancelled / RATE if aec is not None else 0.0),
                "raw_reference_correlation_last": getattr(
                    aec, "raw_reference_correlation", None),
                "residual_reference_correlation_last": getattr(
                    aec, "residual_reference_correlation", None),
                "reference_underflow_seconds": (
                    getattr(aec, "reference_underflow_samples", 0) / RATE
                    if aec is not None else None),
                "reference_overflow_seconds": (
                    getattr(aec, "reference_overflow_samples", 0) / RATE
                    if aec is not None else None),
                "alignment_corrections": getattr(
                    aec, "alignment_corrections", None),
                "estimated_lag_ms": getattr(aec, "estimated_lag_ms", None),
                "estimated_drift_ppm": getattr(
                    aec, "estimated_drift_ppm", None),
            },
            "companion": {
                "enabled": companion is not None,
                "respond": (sum(d["action"] == "respond" and d["valid"]
                                for d in companion.decisions)
                            if companion is not None else 0),
                "listen": (sum(d["action"] == "listen"
                               for d in companion.decisions)
                           if companion is not None else 0),
                "decisions": (companion.decisions
                              if companion is not None else []),
            },
            "agentic": {
                "enabled": agentic_runtime is not None,
                "tool_calls": len(agentic_calls),
                "tool_names": agentic_calls,
                "tool_results": ([m for m in agentic_runtime.agent.messages
                                  if m.get('role') == 'tool']
                                 if agentic_runtime is not None else []),
                "last_iterations": (
                    agentic_runtime.agent.last_iteration_count
                    if agentic_runtime is not None else 0),
                "last_halt_reason": (
                    agentic_runtime.agent.last_halt_reason
                    if agentic_runtime is not None else None),
            },
            "neural_perception_shadow": perception_turns,
        }, replies=all_replies, speaker_labels=speaker_tags,
        interactions=interaction_policy.as_dicts(), errors=run_errors,
        artifacts={"recording_dir": a.record,
                   "meeting_minutes": a.meeting_minutes})
    import hashlib
    run_report['runtime_sha256'] = {
        str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).parent/'core/runtime/speech_delivery.py')}
    saved_report = write_report(report_path, run_report)
    print(f"  report       {saved_report}")
    if aec is not None:
        aec.close()
    # Every artifact is on disk; only now is a PortAudio hang survivable.
    import os as _os

    def _close_audio():
        try:
            mic_stream.abort()
            mic_stream.close()
        except Exception:                                  # noqa: BLE001
            pass

    closer = threading.Thread(target=_close_audio, daemon=True)
    closer.start()
    closer.join(timeout=5.0)
    if closer.is_alive():
        print("  (audio teardown wedged; artifacts saved — forcing exit)",
              flush=True)
        _os._exit(0)
    return 0


def demo() -> None:
    import importlib.util
    assert importlib.util.find_spec("sounddevice"), "sounddevice missing"
    assert importlib.util.find_spec("kokoro"), "kokoro must be in THIS venv"
    src = Path(__file__).read_text()
    assert "SpeexEchoCanceller" in src, "live input must cancel known playback"
    chunks, end = _ready_chunks(
        "This opening clause is deliberately long, but speech continues.", 0)
    assert chunks == ["This opening clause is deliberately long,",
                      "but speech continues."] and end > 0
    chunks, _ = _ready_chunks(
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen", 0)
    assert chunks == [
        "one two three four five six seven eight nine ten eleven twelve"]
    assert "meeting observer" in MEETING_SYSTEM.lower()
    print("live: Kokoro, Speex AEC, clause streaming, native meeting minutes present")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
