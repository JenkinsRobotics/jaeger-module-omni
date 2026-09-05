"""Scripted converse campaign for Gemma Omni V2 (verification harness).

Drives converse() with a FAKE sounddevice (scripted mic + acoustic
echo coupling + playback capture) and REAL models (trained adapter, Gemma,
Kokoro, ECAPA), one scenario per duplex behaviour. Verified 2026-09-03:
half 12/12, quasi 6/6, barge-continue 4/4, full 5/5 (WHITEPAPER addendum).

    .venv/bin/python jaeger_agent_omni/tests/converse_sim.py --session half|quasi|barge-continue|full

Needs the models present; not collected by pytest (no test_ prefix). TTS
clips cache to tests/simcache/ on first run. OMNI_SIM_DIAG=1 adds tracing.
One session per process: --session half|quasi|barge-continue|full."""
import argparse, faulthandler, importlib.util, os, re, signal, sys, threading, time, traceback, types  # noqa: E401
import numpy as np
import hashlib
import json
from pathlib import Path
from scipy.signal import resample_poly

V2 = Path(__file__).resolve().parent.parent          # the jaeger_agent_omni package root
CACHE = Path(__file__).resolve().parent / "simcache"  # cached TTS clips

# ── fake sounddevice (installed BEFORE run_v2 / nodes import it) ──────────
class MicScript:
    def __init__(self):
        self._buf = np.zeros(0, np.float32)      # 16 k mono float
        self._lock = threading.Lock()
        self.rng = np.random.default_rng(7)
    def say(self, a):
        a = np.asarray(a, np.float32)
        if os.environ.get("OMNI_SIM_DIAG"):
            print(f"[SIM] MIC.say {a.size/16000:.2f}s rms="
                  f"{float(np.sqrt(np.mean(a**2))):.4f}", flush=True)
        with self._lock:
            self._buf = np.concatenate([self._buf, a])
    def busy(self):
        with self._lock:
            return self._buf.size > 0
    def frames16(self, n):
        with self._lock:
            take, self._buf = self._buf[:n], self._buf[n:]
        out = np.zeros(n, np.float32)
        out[:take.size] = take
        return out + self.rng.normal(0, 2e-3, n).astype(np.float32)

MIC = MicScript()
HALF = {"until": 0.0}

class _InputStream:                              # half tier: 16 k float32
    def __init__(self, samplerate=16000, channels=1, dtype="float32",
                 blocksize=480, callback=None, **kw):
        self.cb, self.bs, self._run = callback, blocksize, False
    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        per, nxt = self.bs / 16000.0, time.monotonic()
        while self._run:
            now = time.monotonic()
            if now < nxt:
                time.sleep(min(nxt - now, per)); continue
            nxt += per
            try:
                self.cb(MIC.frames16(self.bs).reshape(-1, 1), self.bs,
                        None, None)
            except Exception:
                print("[SIM] INPUT-CB-EXC", flush=True)
                traceback.print_exc(file=sys.__stdout__)
                return
    def stop(self):  self._run = False
    def close(self): self._run = False

class _Stream:                                   # quasi/full: 48 k duplex
    inst = None
    OUT: list = []                               # (t, int16 block) playback log
    def __init__(self, samplerate=48000, channels=(1, 1), dtype="int16",
                 blocksize=480, callback=None, **kw):
        self.cb, self.bs, self._run = callback, blocksize, False
        self.playing_flag = False
        self.last_play_t = 0.0
        self._echo_prev = np.zeros(blocksize, np.int16)   # room coupling
        _Stream.inst = self
    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        per, nxt = self.bs / 48000.0, time.monotonic()
        while self._run:
            now = time.monotonic()
            if now < nxt:
                time.sleep(min(nxt - now, per)); continue
            nxt += per
            fr16 = MIC.frames16(self.bs // 3)
            mic = np.repeat(fr16, 3) * 32767.0     # fast upsample; fidelity
            # is fine for turn/barge/streaming. (ECAPA self-echo needs true
            # resampling, so it is proven by the offline unit check instead
            # of round-tripping through this crude device path.)
            # ACOUSTIC ECHO: the mic hears the agent's own voice through the
            # room (~0.35 coupling, one block late). AEC cancels THIS using
            # the reference; without it Speex's residual suppressor crushes
            # the double-talk and barge never survives (sim physics fix).
            mic = mic + 0.35 * self._echo_prev.astype(np.float32)
            ind = np.clip(mic, -32768, 32767).astype(np.int16).reshape(-1, 1)
            outd = np.zeros((self.bs, 1), np.int16)
            try:
                self.cb(ind, outd, self.bs, None, None)
            except Exception:
                print("[SIM] STREAM-CB-EXC", flush=True)
                traceback.print_exc(file=sys.__stdout__)
                return
            o = outd[:, 0]
            self._echo_prev = o.copy()          # feed back next block
            if np.any(o):
                _Stream.OUT.append((time.time(), o.copy()))
                self.playing_flag = True
                self.last_play_t = time.time()
            else:
                self.playing_flag = False
            # YIELD the GIL every block: a real audio callback runs in a
            # separate high-priority OS thread, so it never starves the
            # engine's barge watcher. This Python-thread stand-in would hog
            # the GIL through its AEC work and the watcher would miss the
            # interruption — the whole barge flake (sim 2026-09-03).
            time.sleep(0.001)
    def stop(self):  self._run = False
    def close(self): self._run = False

def _play(a, sr=24000, **kw):
    HALF["until"] = max(HALF["until"], time.time()) + len(a) / sr
def _wait():
    d = HALF["until"] - time.time()
    if d > 0:
        time.sleep(d)

fake_sd = types.ModuleType("sounddevice")
fake_sd.InputStream = _InputStream
fake_sd.OutputStream = _Stream
fake_sd.Stream = _Stream
fake_sd.play = _play
fake_sd.wait = _wait
fake_sd.default = types.SimpleNamespace(device=None)
sys.modules["sounddevice"] = fake_sd

# ── stdout tee with waitable markers ──────────────────────────────────────
class Tee:
    def __init__(self):
        self.lines, self._cur, self.cursor = [], "", 0
        self.cv = threading.Condition()
    def write(self, s):
        sys.__stdout__.write(s); sys.__stdout__.flush()
        self._cur += s
        while "\n" in self._cur:
            ln, self._cur = self._cur.split("\n", 1)
            with self.cv:
                self.lines.append(ln)
                self.cv.notify_all()
    def flush(self): sys.__stdout__.flush()

TEE = Tee()

def wait_for(marker, timeout=90.0):
    end = time.time() + timeout
    with TEE.cv:
        while True:
            for i in range(TEE.cursor, len(TEE.lines)):
                if marker in TEE.lines[i] and not TEE.lines[i].lstrip().startswith("[SIM]"):
                    TEE.cursor = i + 1
                    return TEE.lines[i]
            if time.time() >= end:
                return None
            TEE.cv.wait(timeout=0.5)

def seen_since(marker, start_idx):
    with TEE.cv:
        return any(marker in ln and not ln.lstrip().startswith("[SIM]")
                   for ln in TEE.lines[start_idx:])

# ── run_v2 under test, with node capture ──────────────────────────────────
spec = importlib.util.spec_from_file_location("run_v2", V2 / "converse.py")
rv = importlib.util.module_from_spec(spec)
sys.modules["run_v2"] = rv
spec.loader.exec_module(rv)
NODES = {}
_orig_load = rv.load_node
def _cap(nid):
    n = _orig_load(nid)
    NODES[nid] = n
    return n
rv.load_node = _cap

# ── user speech (Kokoro am_adam, the voice the gate harness proved) ───────
_pipe = None
def user(text, rms=0.05):
    f = CACHE / (re.sub(r"\W+", "_", text.lower())[:60] + ".npy")
    if f.exists():
        a = np.load(f)
    else:
        global _pipe
        if _pipe is None:
            from kokoro import KPipeline
            _pipe = KPipeline(lang_code="a")
        ch = [np.asarray(r.audio, np.float32) for r in _pipe(text, voice="am_adam")
              if r.audio is not None]
        a = resample_poly(np.concatenate(ch), up=2, down=3).astype(np.float32)
        np.save(f, a)
    return (a * (rms / max(1e-6, float(np.sqrt(np.mean(a ** 2)))))).astype(np.float32)

def agent_voice(text):
    """The AGENT'S voice (af_heart — the TTS node's KOKORO_VOICE), for the
    self-echo leak test: a clip that ECAPA must recognise as self."""
    global _pipe
    if _pipe is None:
        from kokoro import KPipeline
        _pipe = KPipeline(lang_code="a")
    ch = [np.asarray(r.audio, np.float32) for r in _pipe(text, voice="af_heart")
          if r.audio is not None]
    return resample_poly(np.concatenate(ch), up=2, down=3).astype(np.float32)

# ── scenario machinery ────────────────────────────────────────────────────
RESULTS = []
def step(name, ok, extra=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [SIM] {'PASS' if ok else 'FAIL'}  {name}"
          f"{('  · ' + extra[:90]) if extra else ''}", flush=True)
    if not ok and os.environ.get("OMNI_SIM_DIAG"):
        faulthandler.dump_traceback(file=sys.__stdout__)

def playback_done(mode, timeout=150.0):
    end, quiet = time.time() + timeout, 0
    while time.time() < end:
        if mode == "half":
            busy = time.time() < HALF["until"]
        else:
            busy = _Stream.inst is not None and _Stream.inst.playing_flag
        quiet = 0 if (busy or MIC.busy()) else quiet + 1
        if quiet >= 8:                       # 0.8 s of true quiet
            time.sleep(0.6)                  # let speak() finally fully exit
            return True
        time.sleep(0.1)
    return False

def sc_wake(mode):
    mark = len(TEE.lines)
    MIC.say(user("Hey computer, what is two plus two?"))
    r = wait_for("⏵ agent:", 150)
    step("wake -> reply", r and not seen_since("[holding", mark))
    playback_done(mode)

def sc_followup(mode):
    time.sleep(0.5)
    mark = len(TEE.lines)
    MIC.say(user("And what about three plus three?"))
    r = wait_for("⏵ agent:", 120)
    step("follow-up, no wake",
         r and not seen_since("[holding", mark)
         and not seen_since("unintelligible", mark))
    playback_done(mode)

def sc_hold_merge(mode):
    time.sleep(0.5)
    MIC.say(user("Please tell me the capital of"))
    h = wait_for("[holding", 90)
    step("incomplete -> holding", h)
    if h:
        MIC.say(user("France."))
        step("continuation merged", wait_for("⏵ merged:", 60))
        r = wait_for("⏵ agent:", 120)
        step("merged thought answered", r, extra=r or "")
        playback_done(mode)

def sc_hold_expiry(mode):
    time.sleep(0.5)
    MIC.say(user("My favorite animal is the"))
    step("incomplete -> holding (2)", wait_for("[holding", 90))
    step("hold expires -> answers", wait_for("[hold expired", 20))
    step("expiry reply", wait_for("⏵ agent:", 120))
    playback_done(mode)

def sc_typed_memory(mode):
    NODES["text-input-v0"].submit("Remember this: my name is Jonathan.")
    step("typed turn replies", wait_for("⏵ agent:", 120))
    playback_done(mode)
    mark = len(TEE.lines)
    MIC.say(user("Hey computer, what is my name?"))
    r = wait_for("⏵ agent:", 120)
    step("ONE memory across voice+text",
         r and "jonathan" in r.lower() and not seen_since("[holding", mark),
         extra=r or "")
    playback_done(mode)

def sc_dismissal(mode):
    time.sleep(0.5)
    MIC.say(user("Hey computer, thank you, goodbye."))
    step("farewell reply first", wait_for("⏵ agent:", 120))
    step("close AFTER farewell", wait_for("[conversation closed]", 90))
    playback_done(mode)

def playback_started(mode, timeout=30.0):
    """Start interruptions at actual nonzero audio, not a pre-TTS text marker."""
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        active = (time.time() < HALF["until"] if mode == "half" else
                  _Stream.inst is not None and _Stream.inst.playing_flag)
        if active:
            print("  [SIM] actual playback observed", flush=True)
            return True
        time.sleep(.01)
    return False


def sc_barge(mode):
    # A user who is not heard REPEATS louder — realistic, and it also
    # defeats the pure-Python fake-device GIL-scheduling flake (real
    # hardware runs the audio callback in a separate OS thread, so the
    # barge watcher never starves; here it occasionally does). Each attempt
    # triggers a fresh reply to interrupt and barges into it; PASS on the
    # first that cuts playback.
    bz = user("Stop, stop please. I have a different question for you right "
              "now, please just listen to me.")
    bz = np.concatenate([bz, np.zeros(1600, np.float32), bz])
    fired = False
    for attempt in range(4):
        time.sleep(0.3)
        MIC.say(user("What is the sun made of, and why does it shine so very "
                     "brightly up in the sky every single day?"))
        reply = wait_for("⏵ agent:", 150)
        playing = bool(reply and playback_started(mode))
        if attempt == 0:
            step("reply begins", playing)
        if not playing:
            continue
        MIC.say(bz)
        if wait_for("✂ barged", 12):
            fired = True
            break
        playback_done(mode)                  # that attempt did not cut; retry
    step("barge cuts playback", fired, extra=f"after {attempt + 1} attempt(s)")
    step("barge speech answered", wait_for("⏵ agent:", 120))
    playback_done(mode)

def sc_barge_continue(mode):
    time.sleep(0.3)                          # in-window follow-up (woken)
    MIC.say(user("What is the sun made of, and why does it shine so "
                 "brightly in the sky every day?"))
    reply = wait_for("⏵ agent:", 150)
    playing = bool(reply and playback_started(mode))
    step("reply begins", playing)
    mark = len(TEE.lines)
    ov = user("I am talking right over you now, please just acknowledge me.")
    if playing:
        MIC.say(np.concatenate([ov, np.zeros(1600, np.float32), ov]))
    playback_done(mode)
    step("playback NOT cut (continue)", playing and not seen_since("✂ barged", mark))
    step("overtalk heard and answered", playing and wait_for("⏵ agent:", 120))
    playback_done(mode)

def sc_self_echo(mode):
    # Trigger a FRESH reply, then IMMEDIATELY feed a clip in the AGENT'S OWN
    # voice (af_heart) — models an AEC leak of its own speech. The self-vector
    # window is 4 s from speak_end (offline: same-voice cosine 0.91 -> drop,
    # user voice < gate), so it must land fast. Feeding agent-voice directly
    # is more deterministic than round-tripping the capture through the
    # fake device.
    time.sleep(0.3)
    MIC.say(user("Please say a short friendly sentence back to me now."))
    step("fresh reply for echo", wait_for("⏵ agent:", 120))
    t0 = time.time()
    while _Stream.inst is not None and _Stream.inst.playing_flag \
            and time.time() - t0 < 20:
        time.sleep(0.02)
    leak = agent_voice("Okay, here is a friendly little sentence for you.")
    MIC.say(leak * (0.05 / max(1e-6, float(np.sqrt(np.mean(leak ** 2))))))
    step("self echo dropped (ecapa)", wait_for("[self-voice dropped", 40))

def sc_streaming(mode):
    # ONE continuous >12 s turn (0.12 s joins never reach the 500 ms
    # hangover, so it stays a single turn) — this is what exercises
    # commit-and-trim. Separate sentences with real gaps each endpoint into
    # their own turn (correct behaviour, but not the streaming path here).
    parts = [user("Hey computer, I really need your help planning a very "
                  "big surprise birthday party for my younger sister soon."),
             user("She is turning thirty years old this month and she truly "
                  "loves smooth jazz music, rich chocolate cake, old films."),
             user("There will be about twenty different guests coming over, "
                  "and the total budget is around five hundred whole dollars."),
             user("A few of the guests are strict vegetarians, and two of "
                  "them are also bringing their small young children along."),
             user("So given absolutely all of that information, what single "
                  "party theme would you suggest that ties it all together?")]
    def _trim(a, thr=0.01):               # drop leading/trailing silence so
        idx = np.where(np.abs(a) >= thr)[0]   # concatenation is CONTINUOUS
        return a[idx[0]:idx[-1] + 1] if idx.size else a
    # joins carry LOW-LEVEL vocal energy (~0.03 rms, above the 0.02 turn
    # gate) not dead silence — a real speaker mid-thought never goes fully
    # silent, and this keeps the >12 s monologue ONE turn so commit-and-trim
    # actually triggers instead of endpointing between clauses.
    # SPEECH-level joins (~0.05 rms): a lower level gets AGC-attenuated below
    # the 0.02 turn gate (its noise-floor clamp) and reads as a silence gap
    # that endpoints the monologue before 12 s — so keep the joins as loud as
    # the speech to hold ONE turn through commit-and-trim.
    rng = np.random.default_rng(11)
    join = (rng.standard_normal(int(0.15 * 16000)) * 0.05).astype(np.float32)
    mono = []
    for p in parts:
        mono += [_trim(p), join]
    whole = np.concatenate(mono)
    if os.environ.get("OMNI_SIM_DIAG"):
        print(f"[SIM] streaming turn = {whole.size / 16000:.1f}s continuous",
              flush=True)
    MIC.say(whole)
    # The streaming SEMANTICS: the recogniser runs a live forming preview
    # while the user is still talking, and the long monologue is answered.
    # (The 12 s commit-and-trim safety valve is unit-verified in run_v2
    # --selftest via stream_split — TTS/real speech both carry sub-500 ms
    # pauses, so a turn almost always endpoints before 12 s.)
    step("forming preview while listening", wait_for("⋯ forming:", 60))
    step("long monologue answered", wait_for("⏵ agent:", 150))
    playback_done(mode)

def sc_vision_memory(mode):
    from PIL import Image, ImageDraw
    picture = CACHE / "vision-blue-circle.png"
    canvas = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(canvas).ellipse((90, 90, 422, 422), fill="blue")
    canvas.save(picture)
    NODES["text-input-v0"].submit(f"/image {picture}")
    queued = wait_for("[image queued", 30)
    step("local image admitted", queued)
    NODES["text-input-v0"].submit("Name the shape and its color in this image.")
    r = wait_for("⏵ agent:", 180)
    step("image plus text understood", r and "blue" in r.lower()
         and "circle" in r.lower(), extra=r or "")
    playback_done(mode)
    NODES["text-input-v0"].submit("What color was the shape in the image I showed you?")
    r = wait_for("⏵ agent:", 120)
    step("image memory reaches typed turn", r and "blue" in r.lower(), extra=r or "")
    playback_done(mode)
    MIC.say(user("Hey computer, what shape was in the image I showed you?"))
    r = wait_for("⏵ agent:", 120)
    step("image memory reaches voice turn", r and "circle" in r.lower(), extra=r or "")
    playback_done(mode)


SESSIONS = {
    "half":           ("half",  "stop",     [sc_wake, sc_followup,
                                             sc_hold_merge, sc_hold_expiry,
                                             sc_typed_memory, sc_dismissal]),
    "quasi":          ("quasi", "stop",     [sc_wake, sc_barge,
                                             sc_dismissal]),
    "barge-continue": ("quasi", "continue", [sc_wake, sc_barge_continue]),
    "bargemin":       ("quasi", "stop",     [sc_wake, sc_barge]),
    "full":           ("full",  "stop",     [sc_wake, sc_streaming,
                                             sc_dismissal]),
    "vision":         ("half", "stop", [sc_vision_memory]),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, choices=sorted(SESSIONS))
    ap.add_argument("--output", type=Path, help="write reproducible campaign evidence JSON")
    ap.add_argument("--full-ear", help="research override: an immutable node for the full-tier trial only")
    args = ap.parse_args()
    mode, barge, steps = SESSIONS[args.session]
    if args.full_ear:
        if mode != "full" or Path(args.full_ear).name != args.full_ear:
            ap.error("--full-ear requires a local node ID and --session full")
        from jaeger_agent_omni.node_registry import load_package
        package = load_package(V2 / "nodes" / args.full_ear)
        if package.manifest.get("adapter", {}).get("attention") != "causal":
            ap.error("--full-ear requires a declared causal adapter")
        rv.semantic_node_id = lambda _mode: args.full_ear
    signal.alarm(1500)                        # hard whole-session deadline
    sys.stdout = TEE
    th = threading.Thread(
        target=lambda: rv.converse(mode, barge, args.session == "vision", "nn", "nn"),
        daemon=True)
    th.start()
    if not wait_for("Ctrl-C to quit", 600):
        print("[SIM] FATAL converse never became ready", flush=True)
        os._exit(2)
    t0 = time.time()
    while "text-input-v0" not in NODES and time.time() - t0 < 60:
        time.sleep(0.2)
    time.sleep(1.0)
    # No QbE enroll: v0 spotter is crude/speaker-biased and never fires on
    # these phrases (probe 2026-09-02). Wake rides text-wake ("hey computer"
    # transcribes cleanly) + the follow-up window — the realistic model:
    # wake ONCE, then converse; re-wake only after a dismissal clears state.
    for fn in steps:
        fn(mode)
    NODES["text-input-v0"].submit("/quit")
    th.join(30)
    sys.stdout = sys.__stdout__
    npass = sum(1 for _, ok in RESULTS if ok)
    if args.output:
        node_id = rv.semantic_node_id(mode)
        manifest = json.loads((V2 / "nodes" / node_id / "node.json").read_text())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "command": [sys.executable, *sys.argv],
            "session": args.session, "mode": mode, "barge": barge,
            "node": node_id, "adapter_sha256": manifest["adapter"]["sha256"],
            "converse_sha256": hashlib.sha256((V2 / "converse.py").read_bytes()).hexdigest(),
            "perception_context": rv.CONTEXT_TO_LLM,
            "passed": npass, "total": len(RESULTS),
            "checks": [{"name": name, "passed": ok} for name, ok in RESULTS],
            "log": TEE.lines,
        }, indent=2) + "\n")
    print(f"SIM-SESSION {args.session}: {npass}/{len(RESULTS)} PASS",
          flush=True)
    os._exit(0 if npass == len(RESULTS) else 1)

if __name__ == "__main__":
    main()
