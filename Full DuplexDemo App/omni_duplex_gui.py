"""Jaeger Agent Omni full-duplex demo.

This app owns NO model. It is a PySide6 front-end shaped after the official
https://minicpmo45.modelbest.cn/omni web demo, with a selectable engine:

    MiniCPM-o 4.5    full omni — in-process ctypes engine (libomni_shim), 1Hz
                     model-decided LISTEN/SPEAK, voice out (Token2Wav WAVs)
    Jaeger Agent     Gemma-4 E4B audio tower, rolling conversation cache,
                     Kokoro speech, SpeexDSP AEC, residual barge-in, and
                     speaker tags. Audio reaches Gemma without an ASR decoder
                     or a text bridge.

One window:

    LEFT  = INPUT    camera preview, mic waveform, system-prompt text
    RIGHT = OUTPUT   the model's 1Hz LISTEN/SPEAK decisions, its text, its voice

Jaeger continuously prefills the language-model cache from audio embeddings
while the room speaks. A hands-free endpoint closes turns, clause streaming
starts speech early, and the input callback keeps feeding echo-cancelled audio
during generation and playback. MiniCPM remains an optional 1 Hz reference.

The conversation panel intentionally has no user transcript. It displays the
speaker label inserted between turns and the answer emitted by Gemma. Inline
speaker labels are metadata; microphone input remains audio-native.

The optional MiniCPM reference retains its historical paced-frame self-test.
Jaeger's self-test below instead exercises the exact current GUI worker and
reports fed/dropped audio, cache rolls, interruptions, and END→AUDIO.

Run:
    python demo.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from PySide6.QtCore import QBuffer, QIODevice, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGroupBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget)

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000          # APM (whisper encoder) wants 16k
TTS_SAMPLE_RATE = 24000      # Token2Wav emits 24k
LLM_GGUF = Path(os.environ.get(
    "OMNI_LLM_GGUF",
    Path.home() / "models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf"))
WORK_DIR = HERE / "runtime"
IN_DIR = WORK_DIR / "in"
OUT_DIR = WORK_DIR / "out"
ENGINE_LOG = WORK_DIR / "engine.log"
JAEGER_MODEL = Path(os.environ.get(
    "JAEGER_OMNI_MODEL",
    Path.home() / "models/gemma-4-e4b-it-mlx-full"))
JAEGER_STOCK_NODE = Path(os.environ.get(
    "JAEGER_STOCK_NODE",
    HERE.parent / "nodes/stock-whisper-medium-cekl-blend50"))
JAEGER_CLEANROOM_NODE = Path(os.environ.get(
    "JAEGER_CLEANROOM_NODE",
    HERE.parent / "nodes/stock-whisper-medium-cleanroom-ctx93h-step2000"))


def _find_reference_resource(relative: str, env_var: str) -> Path | None:
    """Locate optional MiniCPM assets without making Jaeger depend on them."""
    override = os.environ.get(env_var)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.exists() else None
    roots = [
        HERE.parent,
        Path.home() / "GITHUB/VoiceLLM-Playground/Omni Model",
    ]
    for root in roots:
        hits = sorted(root.glob(f"*/{relative}"))
        if hits:
            return hits[0]
    return None


SHIM = _find_reference_resource("native/libomni_shim.dylib", "OMNI_SHIM")
REF_AUDIO = _find_reference_resource("assets/ref_english.wav", "OMNI_REF_AUDIO")

MEDIA_AUDIO = 1     # audio only — skips the vision encoder entirely
MEDIA_OMNI = 2      # audio + vision


# ══ section: omni_native ctypes wrapper (duplicated from voice_omni.py) ═════
class OmniResult(ctypes.Structure):
    _fields_ = [
        ("user_seq", ctypes.c_longlong),
        ("frame_id", ctypes.c_longlong),
        ("ok", ctypes.c_int),
        ("is_speak", ctypes.c_int),
        ("ms_total", ctypes.c_double),
        ("text", ctypes.c_char * 2048),
    ]


def _load_shim() -> ctypes.CDLL:
    if SHIM is None or not SHIM.exists():
        raise RuntimeError(
            "libomni_shim.dylib not found next to this app — run "
            "bash '<Minicpm 4.5o>/native/build.sh' first")
    lib = ctypes.CDLL(str(SHIM))
    lib.omni_c_init.restype = ctypes.c_void_p
    lib.omni_c_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_char_p]
    lib.omni_c_duplex_begin.restype = ctypes.c_int
    lib.omni_c_duplex_begin.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                        ctypes.c_char_p]
    lib.omni_c_push_frame.restype = ctypes.c_longlong
    lib.omni_c_push_frame.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_longlong]
    lib.omni_c_wait_frame.restype = ctypes.c_int
    lib.omni_c_wait_frame.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.POINTER(OmniResult)]
    lib.omni_c_stop_speak.restype = ctypes.c_int
    lib.omni_c_stop_speak.argtypes = [ctypes.c_void_p]
    lib.omni_c_duplex_end.argtypes = [ctypes.c_void_p]
    lib.omni_c_free.argtypes = [ctypes.c_void_p]
    lib.omni_c_set_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    try:
        lib.omni_c_set_audio_prompt_parts.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
        ]
        lib._has_audio_prompt_parts = True
    except AttributeError:
        lib._has_audio_prompt_parts = False
    try:
        lib.omni_c_reset_context.argtypes = [ctypes.c_void_p]
        lib._has_context_reset = True
    except AttributeError:
        lib._has_context_reset = False
    try:
        lib.omni_c_configure_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        ]
        lib._has_decode_config = True
    except AttributeError:
        lib._has_decode_config = False
    # simplex surface — bound even though the GUI is duplex-only: an unbound
    # ctypes call truncates the 64-bit ctx pointer and segfaults (been there)
    lib.omni_c_stream_prefill.restype = ctypes.c_int
    lib.omni_c_stream_prefill.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                          ctypes.c_int]
    lib.omni_c_stream_decode.restype = ctypes.c_int
    lib.omni_c_stream_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_int]
    lib.omni_c_pop_text.restype = ctypes.c_int
    lib.omni_c_pop_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int]
    try:
        lib.omni_c_append_system_prompt.argtypes = [ctypes.c_void_p,
                                                    ctypes.c_char_p]
        lib._has_prompt = True
    except AttributeError:
        lib._has_prompt = False
    try:
        lib.omni_c_push_frame_av.restype = ctypes.c_longlong
        lib.omni_c_push_frame_av.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                             ctypes.c_char_p, ctypes.c_longlong]
        lib._has_av = True
    except AttributeError:      # older shim build: audio-only frames
        lib._has_av = False
    return lib


class OmniNative:
    """The engine, in this process. push_frame() never waits for the LLM;
    wait_frame() pops decisions FIFO — that decoupling IS the full duplex."""

    def __init__(self, llm_path: Path, output_dir: Path, media_type: int,
                 n_ctx: int = 4096, language: str = "en",
                 duplex: bool = True, use_tts: bool = True) -> None:
        self.lib = _load_shim()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        # n_ctx must be non-zero: at 0 the engine loads, accepts frames, and
        # generates absolutely nothing, with no error anywhere.
        self.ctx = self.lib.omni_c_init(
            str(llm_path).encode(), media_type, int(use_tts), int(duplex),
            n_ctx, 0, str(self.output_dir).encode())
        if not self.ctx:
            raise RuntimeError(f"omni_init failed for {llm_path}")
        # The engine ships with CHINESE system prompts as the default.
        self.lib.omni_c_set_language(self.ctx, language.encode())

    def append_system_prompt(self, text: str) -> bool:
        """The duplex API's only text channel. Call before begin()."""
        if not text or not self.lib._has_prompt:
            return False
        self.lib.omni_c_append_system_prompt(self.ctx, text.encode())
        return True

    def begin(self, ref_audio: Path | None) -> None:
        voice = str(ref_audio).encode() if ref_audio else None
        if not self.lib.omni_c_duplex_begin(self.ctx, voice,
                                            str(self.output_dir).encode()):
            raise RuntimeError("omni_duplex_session_begin failed")

    def push_frame(self, wav_path: Path, img_path: Path | None = None) -> int:
        self._seq += 1
        if img_path is not None and self.lib._has_av:
            fid = self.lib.omni_c_push_frame_av(
                self.ctx, str(wav_path).encode(), str(img_path).encode(),
                self._seq)
        else:
            fid = self.lib.omni_c_push_frame(self.ctx, str(wav_path).encode(),
                                             self._seq)
        if fid < 0:
            raise RuntimeError("push_frame failed — is the session started?")
        return fid

    def wait_frame(self, timeout_ms: int = 500) -> OmniResult | None:
        out = OmniResult()
        if not self.lib.omni_c_wait_frame(self.ctx, timeout_ms,
                                          ctypes.byref(out)):
            return None
        return out

    def stop_speak(self) -> bool:
        return bool(self.lib.omni_c_stop_speak(self.ctx))

    def close(self) -> None:
        try:
            self.lib.omni_c_duplex_end(self.ctx)
            self.lib.omni_c_free(self.ctx)
        except Exception:
            pass


# ══ section: PersistentPlayer (duplicated from voice_omni.py) ═══════════════
class PersistentPlayer:
    """One output stream for the whole session — per-chunk sd.play() opens a
    fresh CoreAudio stream every second and eventually dies with -9986."""

    def __init__(self, samplerate: int = TTS_SAMPLE_RATE) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self.playing = threading.Event()
        # RMS of the samples the LAST callback actually wrote to the device —
        # the agent's true voice level, not an animation. 0.0 while silent.
        self.level = 0.0
        self._stream = sd.OutputStream(samplerate=samplerate, channels=1,
                                       dtype="float32", blocksize=1024,
                                       callback=self._cb)

    def _cb(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        filled = 0
        while filled < frames:
            if self._buf.size == 0:
                try:
                    self._buf = self._q.get_nowait()
                except queue.Empty:
                    break
            n = min(frames - filled, self._buf.size)
            out[filled:filled + n] = self._buf[:n]
            self._buf = self._buf[n:]
            filled += n
        out[filled:] = 0
        if filled:
            self.playing.set()
            self.level = float(np.sqrt(np.mean(np.square(out[:filled]))))
        else:
            self.level = 0.0
            if self._q.empty() and self._buf.size == 0:
                self.playing.clear()

    def play(self, audio: np.ndarray) -> None:
        self._q.put(np.asarray(audio, dtype=np.float32).reshape(-1))

    def flush(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._buf = np.zeros(0, dtype=np.float32)

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


# ══ engine worker ═══════════════════════════════════════════════════════════
class EngineWorker(QThread):
    """Owns the engine end to end, off the UI thread.

    mic thread → audio_q → [1s accumulate → wav (+jpeg) → push_frame]
    engine     → wait_frame → decision/text signals
    engine     → wav files  → scan → PersistentPlayer
    """

    status = Signal(str)
    ready = Signal()
    decision = Signal(bool, str, float)     # is_speak, text, ms_total
    frame_pushed = Signal(int)
    backlog = Signal(float, int)            # seconds queued, frames dropped
    failed = Signal(str)

    def __init__(self, language: str, send_video: bool, mute_while_speaking: bool,
                 system_prompt: str) -> None:
        super().__init__()
        self.language = language
        self.send_video = send_video
        self.mute_while_speaking = mute_while_speaking
        self.system_prompt = system_prompt.strip()
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.latest_jpeg: bytes | None = None   # set by the UI camera sink
        self._jpeg_lock = threading.Lock()
        self._stop = threading.Event()
        self._dropped = 0
        self.player: PersistentPlayer | None = None
        self.eng: OmniNative | None = None

    def hush(self) -> None:
        """Barge-in: cut generation AND drop what's already queued to play."""
        if self.eng is not None:
            self.eng.stop_speak()
        if self.player is not None:
            self.player.flush()

    def set_jpeg(self, data: bytes) -> None:
        with self._jpeg_lock:
            self.latest_jpeg = data
            self._jpeg_at = time.monotonic()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # noqa: C901 — one linear session, read top-down
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        IN_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # llama.cpp printf-spams fd 1/2; send it to a file so the terminal
        # stays usable. Process-wide, deliberate.
        saved = (os.dup(1), os.dup(2))
        log_fd = os.open(ENGINE_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
        eng = None
        try:
            self.status.emit("loading engine (~12GB, takes a while)…")
            media = MEDIA_OMNI if self.send_video else MEDIA_AUDIO
            eng = OmniNative(LLM_GGUF, OUT_DIR, media_type=media,
                             language=self.language)
            self.eng = eng                      # for barge-in from the UI
            if self.system_prompt:
                if eng.append_system_prompt(self.system_prompt):
                    self.status.emit("system prompt applied")
                else:
                    self.status.emit("shim lacks append_system_prompt — rebuild native/")
            self.status.emit("starting duplex session…")
            eng.begin(REF_AUDIO)
            self.player = PersistentPlayer()
            self.player.start()
            self.ready.emit()
            self.status.emit("session live — model decides at 1Hz")

            consumer = threading.Thread(target=self._consume, args=(eng,))
            consumer.start()
            threading.Thread(target=self._play_new_wavs, daemon=True).start()

            pending = np.zeros(0, dtype=np.float32)
            seq = 0
            while not self._stop.is_set():
                try:
                    frame = self.audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if (self.mute_while_speaking and self.player is not None
                        and self.player.playing.is_set()):
                    # Half-duplex guard (no AEC in this GUI): keep the 1s
                    # cadence but push silence, or the model hears itself and
                    # self-converses. ponytail: vendor SharedAudio's Speex AEC
                    # for true simultaneous talk-over.
                    frame = np.zeros_like(frame)
                pending = np.concatenate([pending, frame.reshape(-1)])
                # Engine slower than realtime -> backlog. Keep at most 10s,
                # drop the OLDEST audio, and report the drop — never silently.
                if pending.size > SAMPLE_RATE * 10:
                    cut = pending.size - SAMPLE_RATE * 10
                    pending = pending[cut:]
                    self._dropped += cut
                self.backlog.emit(pending.size / SAMPLE_RATE,
                                  int(self._dropped / SAMPLE_RATE))
                while pending.size >= SAMPLE_RATE and not self._stop.is_set():
                    piece, pending = pending[:SAMPLE_RATE], pending[SAMPLE_RATE:]
                    seq += 1
                    wav = IN_DIR / f"live_{seq:05d}.wav"
                    sf.write(wav, piece, SAMPLE_RATE)
                    jpeg = None
                    if self.send_video:
                        with self._jpeg_lock:
                            data = self.latest_jpeg
                            if time.monotonic() - getattr(self, "_jpeg_at", 0) > 2.0:
                                data = None      # camera stalled — stale frame
                        if data:
                            jpeg = IN_DIR / f"live_{seq:05d}.jpg"
                            jpeg.write_bytes(data)
                    eng.push_frame(wav, jpeg)
                    self.frame_pushed.emit(seq)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self._stop.set()
            try:
                consumer.join(timeout=2.0)   # may sit inside wait_frame(500ms)
            except (NameError, RuntimeError):
                pass                          # load failed before it started
            if eng is not None:
                eng.close()
            if self.player is not None:
                self.player.close()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)
            self.status.emit("session ended")

    def _consume(self, eng: OmniNative) -> None:
        while not self._stop.is_set():
            r = eng.wait_frame(timeout_ms=500)
            if r is None:
                continue
            text = r.text.decode(errors="replace") if isinstance(r.text, bytes) \
                else (r.text or "")
            self.decision.emit(bool(r.is_speak), text, float(r.ms_total))

    def _play_new_wavs(self) -> None:
        played: set[Path] = set()
        while not self._stop.is_set():
            for w in sorted(OUT_DIR.rglob("wav_*.wav")):
                if w in played:
                    continue
                played.add(w)
                try:
                    pcm, sr = sf.read(w, dtype="float32")
                    if pcm.size and self.player is not None:
                        if sr != TTS_SAMPLE_RATE:   # engine emits 24k; resample if not
                            idx = np.linspace(0, pcm.shape[0] - 1,
                                              int(pcm.shape[0] * TTS_SAMPLE_RATE / sr))
                            pcm = pcm[idx.astype(int)]
                        self.player.play(pcm)
                except Exception:
                    pass                     # mid-write wav — picked up next scan
            time.sleep(0.05)


class PlaybackMonitor:
    """Observable state for Jaeger's real sounddevice output."""

    def __init__(self) -> None:
        self.playing = threading.Event()
        self.level = 0.0

    def flush(self) -> None:
        sd.stop()
        self.playing.clear()
        self.level = 0.0


class JaegerEngineWorker(QThread):
    """The same audio-native path as :mod:`jaeger_agent_omni.live`.

    The sounddevice callback calls :meth:`push_audio`, so AEC and barge-in keep
    running while the model thread is generating. The worker never decodes the
    microphone to words; the only text entering the cache is system context,
    recent assistant replies after a roll, and an inline speaker label.
    """

    status = Signal(str)
    ready = Signal()
    reply = Signal(str)
    speaker = Signal(str)
    metrics = Signal(str)
    barged = Signal()
    failed = Signal(str)

    def __init__(self, system_prompt: str = "", voice: str = "af_heart",
                 speed: float = 1.0, threshold: float = 0.012,
                 silence_ms: float = 700.0, cooldown: float = 4.0,
                 max_context: int = 6000, speakers: int = 2,
                 front_end: str = "tower") -> None:
        super().__init__()
        self.system_prompt = system_prompt.strip()
        self.voice = voice
        self.speed = speed
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.cooldown = cooldown
        self.max_context = max_context
        self.speakers = speakers
        self.front_end = front_end
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.say_q: queue.Queue[str | None] = queue.Queue()
        self._stop_evt = threading.Event()
        self.generating = threading.Event()
        self.barge_active = threading.Event()
        self.player = PlaybackMonitor()
        self.loop = None
        self.session = None
        self.aec = None
        self.raw_level = 0.0
        self.clean_level = 0.0
        self.fed_samples = 0
        self.dropped_samples = 0
        self.barge_count = 0
        self.rolls = 0
        self._last_metric = 0.0
        self._turn_t0: float | None = None
        self.end_audio_ms: list[float] = []
        self._recent_replies: list[str] = []
        self._speaker_audio: deque[np.ndarray] = deque(maxlen=30)
        self._speaker_tracker = None

    def push_audio(self, samples: np.ndarray) -> None:
        """Capture-thread entry: cancel echo and detect speech during output."""
        raw = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        clean = self.aec.process(raw) if self.aec is not None else raw.copy()
        self.raw_level = float(np.sqrt(np.mean(np.square(raw)))) if raw.size else 0.0
        self.clean_level = (float(np.sqrt(np.mean(np.square(clean))))
                            if clean.size else 0.0)

        if self.player.playing.is_set():
            attenuation = 20 * np.log10(
                (self.raw_level + 1e-9) / (self.clean_level + 1e-9))
            is_barge = (self.clean_level > self.threshold * 1.1 and
                        self.aec is not None and
                        (self.aec.playback_age > 1.0 or attenuation < 3.0))
            if is_barge and not self.barge_active.is_set():
                self.hush()
        elif (self.generating.is_set() and
              self.clean_level > self.threshold * 1.1 and
              not self.barge_active.is_set()):
            self.barge_active.set()
            self.barge_count += 1
            self.barged.emit()
            if self.session is not None:
                self.session.cancel()

        self._speaker_audio.append(clean.copy())
        try:
            self.audio_q.put_nowait(clean)
        except queue.Full:
            self.dropped_samples += clean.size

    def hush(self) -> None:
        """Cut both generation and queued/playing speech without losing input."""
        was_active = self.generating.is_set() or self.player.playing.is_set()
        first = not self.barge_active.is_set()
        self.barge_active.set()
        if self.generating.is_set() and self.session is not None:
            self.session.cancel()
        self.player.flush()
        while True:
            try:
                self.say_q.get_nowait()
            except queue.Empty:
                break
        if first and was_active:
            self.barge_count += 1
            self.barged.emit()

    def stop(self) -> None:
        self._stop_evt.set()
        if self.generating.is_set() and self.session is not None:
            self.session.cancel()
        if self.loop is not None:
            self.loop.stop()
        self.player.flush()
        while True:
            try:
                self.say_q.get_nowait()
            except queue.Empty:
                break

    def _source(self):
        blocks = []
        while True:
            try:
                blocks.append(self.audio_q.get_nowait())
            except queue.Empty:
                break
        if not blocks:
            return None
        audio = np.concatenate(blocks)
        self.fed_samples += audio.size
        self._emit_metrics()
        return audio

    def _emit_metrics(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_metric < 1.0:
            return
        self._last_metric = now
        stats = getattr(self.loop, "stats", None)
        cache = getattr(self.session, "n_prefilled", 0)
        lag = getattr(stats, "max_lag_s", 0.0) * 1000 if stats else 0.0
        turns = getattr(stats, "turns", 0) if stats else 0
        interrupts = getattr(stats, "interrupts", 0) if stats else 0
        aec_db = getattr(self.aec, "attenuation_db", 0.0)
        self.metrics.emit(
            f"cache {cache} tok · fed {self.fed_samples/SAMPLE_RATE:.1f}s · "
            f"dropped {self.dropped_samples/SAMPLE_RATE:.1f}s · peak lag "
            f"{lag:.0f}ms · turns {turns} · interrupts {interrupts} · "
            f"barge-ins {self.barge_count} · rolls {self.rolls} · AEC {aec_db:.1f}dB")

    def _tag_speaker(self) -> None:
        if self._speaker_tracker is None or not self._speaker_audio:
            return
        audio = np.concatenate(tuple(self._speaker_audio))
        self._speaker_audio.clear()
        label = self._speaker_tracker.identify(audio)
        if label is None:
            return
        self.session.feed_text(f" [speaker {label}]")
        self.speaker.emit(label)

    def _maybe_roll(self) -> None:
        if (self.session.n_prefilled <= self.max_context or
                self.generating.is_set()):
            return
        summary = " ".join(self._recent_replies[-2:])[:400]
        before = self.session.n_prefilled
        self.session.reset()
        self.session.open_turn()
        if summary:
            self.session.feed_text(
                f" (ongoing conversation; my recent replies: {summary}) ")
        self.rolls += 1
        self.status.emit(
            f"context rolled {before} → {self.session.n_prefilled} tokens")

    def _play(self) -> None:
        import soundfile as sf
        from jaeger_agent_omni.voice import speak_kokoro

        out = WORK_DIR / "jaeger_reply.wav"
        while not self._stop_evt.is_set():
            clause = self.say_q.get()
            if clause is None:
                break
            try:
                t0 = time.monotonic()
                speak_kokoro(clause, out, self.voice, self.speed)
                wav, sr = sf.read(out, dtype="float32")
                synth_ms = (time.monotonic() - t0) * 1000
                # A listener can interrupt while Kokoro is still synthesizing.
                # Never let that now-stale clause begin playing afterward.
                if self.barge_active.is_set() or self._stop_evt.is_set():
                    continue
                if self.aec is not None:
                    self.aec.start_playback(wav, sr)
                self.player.playing.set()
                if self._turn_t0 is not None:
                    first_ms = (time.monotonic() - self._turn_t0) * 1000
                    self.end_audio_ms.append(first_ms)
                    self.metrics.emit(
                        f"END→AUDIO {first_ms:.0f}ms · synth {synth_ms:.0f}ms")
                    self._turn_t0 = None
                sd.play(wav, sr)
                started = time.monotonic()
                while not self._stop_evt.is_set():
                    try:
                        if not sd.get_stream().active:
                            break
                    except Exception:
                        break
                    pos = int((time.monotonic() - started) * sr)
                    sl = wav[max(0, pos - 1024):min(len(wav), pos + 1024)]
                    self.player.level = (float(np.sqrt(np.mean(np.square(sl))))
                                         if sl.size else 0.0)
                    time.sleep(0.03)
                sd.wait()
            except Exception as exc:
                self.status.emit(f"speech output: {exc}")
            finally:
                self.player.playing.clear()
                self.player.level = 0.0

    def run(self) -> None:  # noqa: C901 — one session lifecycle
        try:
            sys.path.insert(0, str(HERE.parent))
            from jaeger_agent_omni.live import _ready_chunks
            from jaeger_agent_omni.omni import SYSTEM, build
            from jaeger_agent_omni.runtime.aec import SpeexEchoCanceller
            from jaeger_agent_omni.runtime.duplex import DuplexLoop
            from jaeger_agent_omni.runtime.speaker import OnlineSpeakerTracker

            self.status.emit("loading Jaeger Gemma-4 E4B…")
            system = SYSTEM
            if self.system_prompt:
                system = f"{SYSTEM} {self.system_prompt}"
            selected_front_end = (
                "stock" if self.front_end in ("stock", "cleanroom")
                else "tower")
            adapter = (
                str(JAEGER_CLEANROOM_NODE) if self.front_end == "cleanroom"
                else str(JAEGER_STOCK_NODE) if self.front_end == "stock"
                else None)
            session, meta = build(
                str(JAEGER_MODEL), selected_front_end, adapter, system=system)
            self.session = session
            warm_ms = session.warmup()

            self.status.emit("warming Kokoro voice and SpeexDSP AEC…")
            from jaeger_agent_omni.voice import speak_kokoro
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            speak_kokoro("ready", WORK_DIR / "jaeger_warm.wav",
                         self.voice, self.speed)
            self.aec = SpeexEchoCanceller(SAMPLE_RATE)
            if self.speakers:
                self._speaker_tracker = OnlineSpeakerTracker(self.speakers)

            player_thread = threading.Thread(target=self._play, daemon=True)
            player_thread.start()
            tok = session.processor.tokenizer
            original_respond = session.respond
            last_reply = {"at": 0.0}

            def streaming_respond(**kw):
                self._turn_t0 = time.monotonic()
                self.barge_active.clear()
                self.generating.set()
                ids, consumed = [], 0
                inner = kw.get("on_token")

                def on_token(tid, n):
                    nonlocal consumed
                    ids.append(tid)
                    text = tok.decode(ids)
                    chunks, consumed = _ready_chunks(text, consumed)
                    for chunk in chunks:
                        self.say_q.put(chunk)
                    if inner:
                        inner(tid, n)

                kw["on_token"] = on_token
                try:
                    answer = original_respond(**kw)
                finally:
                    self.generating.clear()
                tail = answer[consumed:].split("<")[0].strip()
                if len(tail.split()) >= 2:
                    self.say_q.put(tail)
                return answer

            session.respond = streaming_respond

            def on_reply(text: str) -> None:
                answer = text.split("<")[0].strip()
                if not answer:
                    return
                last_reply["at"] = time.monotonic()
                self._recent_replies.append(answer)
                del self._recent_replies[:-3]
                self.reply.emit(answer)

            loop = DuplexLoop(
                session, audio_source=self._source,
                user_speaking=self.barge_active.is_set,
                on_reply=on_reply,
                before_respond=self._tag_speaker if self.speakers else None)
            self.loop = loop

            def turn_gate() -> bool:
                self._maybe_roll()
                if time.monotonic() - last_reply["at"] < self.cooldown:
                    return False
                return bool(loop.endpoint and loop.endpoint.ended)

            loop.turn_ended = turn_gate
            loop.endpoint.speech_rms = self.threshold
            loop.endpoint.silence_ms = self.silence_ms
            loop.endpoint.__post_init__()

            self.ready.emit()
            self.status.emit(
                f"LIVE · {meta['front_end']} audio embeddings · AEC on · "
                f"warmup {warm_ms:.0f}ms")
            loop.run(eos_ids=meta["eos"])
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._stop_evt.set()
            self.player.flush()
            self.say_q.put(None)
            try:
                player_thread.join(timeout=5.0)
            except (NameError, RuntimeError):
                pass
            self._emit_metrics(force=True)
            if self.aec is not None:
                self.aec.close()
            self.status.emit("session ended")


# ══ input widgets ═══════════════════════════════════════════════════════════
class Waveform(QWidget):
    """Rolling REAL level strip. Fed by measured RMS — the mic capture for the
    user, the playback callback for the agent — never by an animation."""

    def __init__(self, color: tuple[int, int, int] = (80, 200, 120)) -> None:
        super().__init__()
        self.color = QColor(*color)
        self.levels: deque[float] = deque(maxlen=120)
        self.setMinimumHeight(64)

    def push(self, level: float) -> None:
        self.levels.append(min(1.0, level * 12.0))
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 20))
        if not self.levels:
            return
        w = self.width() / self.levels.maxlen
        mid = self.height() / 2
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        for i, lv in enumerate(self.levels):
            h = max(1.0, lv * mid)
            p.drawRect(int(i * w), int(mid - h), max(1, int(w) - 1), int(h * 2))


class DecisionTicker(QWidget):
    """The experiment's readout: one square per 1Hz decision.
    grey = LISTEN, green = SPEAK. A green square is news."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: deque[bool | None] = deque(maxlen=60)
        self.setMinimumHeight(28)

    def push(self, is_speak: bool) -> None:
        self.decisions.append(is_speak)
        self.update()

    def interrupt(self) -> None:
        self.decisions.append(None)
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 20))
        w = self.width() / self.decisions.maxlen
        p.setPen(Qt.NoPen)
        for i, spoke in enumerate(self.decisions):
            color = (QColor(220, 165, 60) if spoke is None else
                     QColor(80, 220, 120) if spoke else QColor(70, 70, 80))
            p.setBrush(color)
            p.drawRect(int(i * w) + 1, 4, max(2, int(w) - 2), self.height() - 8)


# ══ main window ═════════════════════════════════════════════════════════════
class OmniDuplexWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Jaeger Agent Omni — Full-Duplex Demo")
        self.resize(1220, 760)
        self.worker = None                       # EngineWorker | JaegerEngineWorker
        self.mic_stream: sd.InputStream | None = None
        self.camera: QCamera | None = None
        self.counts = {"frames": 0, "listen": 0, "speak": 0,
                       "commits": 0, "turns": 0}
        self._mic_level = 0.0
        self._capture_errors = 0
        self._capture_error_text = ""
        self.source_wav: Path | None = None      # None = live microphone
        self._file_timer: QTimer | None = None
        self._rec: list[np.ndarray] = []
        self._paint_timer = QTimer(self)
        self._paint_timer.timeout.connect(self._paint_tick)
        self._paint_timer.start(50)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addLayout(self._build_engine_bar())
        split = QSplitter()
        split.addWidget(self._build_input_panel())
        split.addWidget(self._build_output_panel())
        split.setSizes([560, 640])
        outer.addWidget(split, stretch=1)
        outer.addLayout(self._build_control_bar())
        self._on_engine_changed()
        if not JAEGER_MODEL.exists():
            self._set_status(f"Jaeger model missing: {JAEGER_MODEL}")

    # -- engine selector -----------------------------------------------------
    def _build_engine_bar(self):
        bar = QHBoxLayout()
        title = QLabel("<b>Jaeger Agent Omni</b> — local full duplex")
        bar.addWidget(title)
        bar.addStretch(1)
        self.engine_jaeger = QPushButton("Jaeger · audio-native live")
        self.engine_minicpm = QPushButton("MiniCPM-o 4.5 · reference")
        group = QButtonGroup(self)
        for b in (self.engine_jaeger, self.engine_minicpm):
            b.setCheckable(True)
            group.addButton(b)
            bar.addWidget(b)
        group.setExclusive(True)
        self.engine_jaeger.setChecked(True)
        group.buttonToggled.connect(lambda *_: self._on_engine_changed())
        return bar

    # -- bottom control bar (official-demo order) ------------------------------
    def _build_control_bar(self):
        bar = QHBoxLayout()
        bar.addStretch(1)
        self.hush_btn = QPushButton("Force Listen")
        self.hush_btn.setToolTip("barge-in: cut generation and flush queued speech")
        self.hush_btn.clicked.connect(self._hush)
        self.hush_btn.setEnabled(False)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("stop feeding the engine; capture stays open")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("color:#c33;")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.start_btn = QPushButton("● Start")
        self.start_btn.setStyleSheet(
            "background:#1db954;color:white;font-weight:bold;padding:6px 22px;")
        self.start_btn.clicked.connect(self._start)
        self.rec_check = QCheckBox("Rec")
        self.rec_check.setToolTip(
            "save the exact mic PCM of the session to runtime/capture_*.wav")
        for w in (self.hush_btn, self.pause_btn, self.stop_btn, self.start_btn):
            bar.addWidget(w)
        bar.addWidget(self.rec_check)
        bar.addStretch(1)
        return bar

    def _on_engine_changed(self) -> None:
        minicpm = self.engine_minicpm.isChecked()
        # honest per-engine capability gating — no dead knobs
        self.video_check.setEnabled(minicpm)
        if not minicpm:
            self.video_check.setChecked(False)
        self.mute_check.setEnabled(minicpm)
        self.hush_btn.setText("Force Listen")
        if minicpm:
            self.tts_label.setText(
                "voice out: model's own voice (Token2Wav), always on")
            self.ticker_label.setText(
                "decisions (1Hz) — grey LISTEN · green SPEAK:")
            self._set_status("idle — press Start. Model decides LISTEN/SPEAK "
                             "once per second.")
        else:
            self.tts_label.setText(
                "voice out: Kokoro clause streaming · SpeexDSP AEC · barge-in")
            self.ticker_label.setText(
                "turns — grey listening · green reply · amber interruption:")
            self._set_status("idle — press Start. Audio goes directly into "
                             "Gemma; a pause ends the turn.")
        for widget in (self.voice_combo, self.speed_spin, self.speaker_spin,
                       self.threshold_spin, self.silence_spin,
                       self.cooldown_spin, self.context_spin,
                       self.front_end_combo):
            widget.setEnabled(not minicpm)
        self._refresh_counts()

    # -- left: INPUT --------------------------------------------------------
    def _build_input_panel(self) -> QWidget:
        box = QGroupBox("INPUT — what the model receives")
        lay = QVBoxLayout(box)

        self.cam_label = QLabel("camera off")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumHeight(280)
        self.cam_label.setStyleSheet("background:#101014;color:#666;")
        lay.addWidget(self.cam_label, stretch=3)

        self.video_check = QCheckBox("send video — 1 frame/s through the vision encoder (omni mode)")
        self.video_check.toggled.connect(self._toggle_camera)
        lay.addWidget(self.video_check)

        src = QHBoxLayout()
        self.live_btn = QPushButton("● Live")
        self.live_btn.setCheckable(True)
        self.live_btn.setChecked(True)
        self.live_btn.clicked.connect(self._pick_live)
        self.file_btn = QPushButton("File…")
        self.file_btn.setCheckable(True)
        self.file_btn.clicked.connect(self._pick_file)
        self.source_label = QLabel("microphone")
        self.source_label.setStyleSheet("color:#888;")
        src.addWidget(self.live_btn)
        src.addWidget(self.file_btn)
        src.addWidget(self.source_label)
        src.addStretch(1)
        lay.addLayout(src)

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        configured = os.environ.get("JAEGER_INPUT_DEVICE", "").strip()
        default_device = sd.default.device[0]
        selected = -1
        try:
            for index, info in enumerate(sd.query_devices()):
                if int(info.get("max_input_channels", 0)) <= 0:
                    continue
                name = str(info["name"])
                self.device_combo.addItem(f"{index}: {name}", index)
                matches_override = (configured == str(index) or
                                    configured.lower() in name.lower())
                if matches_override or (not configured and index == default_device):
                    selected = self.device_combo.count() - 1
        except Exception as exc:
            self.device_combo.addItem(f"device query failed: {exc}", None)
        if selected >= 0:
            self.device_combo.setCurrentIndex(selected)
        self.capture_health_label = QLabel("capture: idle · 0 callback errors")
        self.capture_health_label.setStyleSheet("color:#888;")
        device_row.addWidget(QLabel("microphone:"))
        device_row.addWidget(self.device_combo, stretch=1)
        lay.addLayout(device_row)
        lay.addWidget(self.capture_health_label)

        lay.addWidget(QLabel("you — microphone waveform (16 kHz, fed continuously):"))
        self.waveform = Waveform()
        lay.addWidget(self.waveform)

        lay.addWidget(QLabel(
            "optional session instruction — applied at Start, never an input transcript:"))
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("optional extra system prompt, applied at Start")
        lay.addWidget(self.prompt_edit)

        row = QHBoxLayout()
        self.front_end_combo = QComboBox()
        self.front_end_combo.addItem(
            "Gemma native ear · live rollback", "tower")
        self.front_end_combo.addItem(
            "Whisper medium + Jaeger 50M · production candidate", "stock")
        self.front_end_combo.addItem(
            "Whisper medium + clean-room 41M · 3.23% candidate", "cleanroom")
        row.addWidget(QLabel("audio input:"))
        row.addWidget(self.front_end_combo)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["en", "zh"])
        row.addWidget(QLabel("language:"))
        row.addWidget(self.lang_combo)
        self.mute_check = QCheckBox(
            "MiniCPM only: mute mic while speaking (reference path has no AEC)")
        self.mute_check.setChecked(True)
        row.addWidget(self.mute_check)
        row.addStretch(1)
        lay.addLayout(row)

        tuning = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.addItems([
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george"])
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.7, 1.5)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.0)
        self.speaker_spin = QSpinBox()
        self.speaker_spin.setRange(0, 4)
        self.speaker_spin.setValue(2)
        tuning.addWidget(QLabel("voice:"))
        tuning.addWidget(self.voice_combo)
        tuning.addWidget(QLabel("speed:"))
        tuning.addWidget(self.speed_spin)
        tuning.addWidget(QLabel("room speakers:"))
        tuning.addWidget(self.speaker_spin)
        tuning.addStretch(1)
        lay.addLayout(tuning)

        runtime = QHBoxLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(0.005, 0.500)
        self.threshold_spin.setSingleStep(0.010)
        self.threshold_spin.setValue(0.200)
        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(300, 2000)
        self.silence_spin.setSingleStep(50)
        self.silence_spin.setValue(700)
        self.silence_spin.setSuffix(" ms")
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.0, 60.0)
        self.cooldown_spin.setValue(4.0)
        self.cooldown_spin.setSuffix(" s")
        self.context_spin = QSpinBox()
        self.context_spin.setRange(1000, 12000)
        self.context_spin.setSingleStep(500)
        self.context_spin.setValue(6000)
        runtime.addWidget(QLabel("room floor:"))
        runtime.addWidget(self.threshold_spin)
        runtime.addWidget(QLabel("silence:"))
        runtime.addWidget(self.silence_spin)
        runtime.addWidget(QLabel("cooldown:"))
        runtime.addWidget(self.cooldown_spin)
        runtime.addWidget(QLabel("roll:"))
        runtime.addWidget(self.context_spin)
        runtime.addStretch(1)
        lay.addLayout(runtime)

        self.tts_label = QLabel()
        self.tts_label.setStyleSheet("color:#888;")
        lay.addWidget(self.tts_label)
        return box

    def _pick_live(self) -> None:
        self.source_wav = None
        self.file_btn.setChecked(False)
        self.live_btn.setChecked(True)
        self.source_label.setText("microphone")

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose audio to replay at wall-clock pace", str(HERE),
            "Audio (*.wav *.mp3 *.flac)")
        if not path:
            self.file_btn.setChecked(False)
            return
        self.source_wav = Path(path)
        self.live_btn.setChecked(False)
        self.file_btn.setChecked(True)
        self.source_label.setText(f"file: {self.source_wav.name} (paced replay)")

    # -- right: OUTPUT -------------------------------------------------------
    def _build_output_panel(self) -> QWidget:
        box = QGroupBox("OUTPUT — what the model decides and says")
        lay = QVBoxLayout(box)

        chips = QHBoxLayout()
        chips.addWidget(QLabel("Preset:"))
        for label, lang in (("English Call", "en"), ("中文通话", "zh")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, lg=lang: self._apply_preset(lg))
            chips.addWidget(b)
        meeting = QPushButton("Meeting third-person")
        meeting.setToolTip(
            "listen continuously; contribute only when useful and track decisions")
        meeting.clicked.connect(self._apply_meeting_preset)
        chips.addWidget(meeting)
        chips.addStretch(1)
        lay.addLayout(chips)

        self.ticker_label = QLabel()
        lay.addWidget(self.ticker_label)
        self.ticker = DecisionTicker()
        lay.addWidget(self.ticker)
        self.counts_label = QLabel("0 frames → 0 listen, 0 SPEAK")
        lay.addWidget(self.counts_label)

        lay.addWidget(QLabel("agent voice — real output waveform (flat when silent):"))
        self.agent_waveform = Waveform(color=(80, 160, 240))
        lay.addWidget(self.agent_waveform)

        lay.addWidget(QLabel("Conversation:"))
        self.convo = QTextEdit()
        self.convo.setReadOnly(True)
        lay.addWidget(self.convo, stretch=1)

        self.metrics_label = QLabel("—")
        lay.addWidget(self.metrics_label)
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color:#8aa;")
        lay.addWidget(self.status_label)
        return box

    # -- session control -----------------------------------------------------
    def _apply_preset(self, lang: str) -> None:
        self.lang_combo.setCurrentText(lang)
        self.prompt_edit.setText("")
        self._set_status(f"preset applied: language={lang}")

    def _apply_meeting_preset(self) -> None:
        self.engine_jaeger.setChecked(True)
        self.lang_combo.setCurrentText("en")
        self.prompt_edit.setText(
            "You are the third participant in a live meeting. Listen "
            "continuously. Speak only when directly asked or when you can "
            "fill a missing gap, resolve ambiguity, or briefly summarize a "
            "decision and its action items. Do not narrate or repeat the "
            "conversation.")
        self.cooldown_spin.setValue(20.0)
        self._set_status(
            "meeting preset applied · selective contributions · 20 s cooldown")

    def _start(self) -> None:
        self.counts = {"frames": 0, "listen": 0, "speak": 0,
                       "commits": 0, "turns": 0}
        self.convo.clear()
        self.ticker.decisions.clear()
        self.ticker.update()
        self._rec = []
        self._capture_errors = 0
        self._capture_error_text = ""
        self.device_combo.setEnabled(False)
        self.pause_btn.setChecked(False)
        if self.engine_jaeger.isChecked():
            if self.lang_combo.currentText() != "en":
                self._set_status("note: Jaeger path is English-tuned; zh untested")
            self.worker = JaegerEngineWorker(
                system_prompt=self.prompt_edit.text(),
                voice=self.voice_combo.currentText(),
                speed=self.speed_spin.value(),
                threshold=self.threshold_spin.value(),
                silence_ms=self.silence_spin.value(),
                cooldown=self.cooldown_spin.value(),
                max_context=self.context_spin.value(),
                speakers=self.speaker_spin.value(),
                front_end=self.front_end_combo.currentData())
            self.worker.reply.connect(self._on_j_reply)
            self.worker.speaker.connect(self._on_j_speaker)
            self.worker.metrics.connect(self.metrics_label.setText)
            self.worker.barged.connect(self._on_j_barge)
        else:
            self.worker = EngineWorker(
                language=self.lang_combo.currentText(),
                send_video=self.video_check.isChecked(),
                mute_while_speaking=self.mute_check.isChecked(),
                system_prompt=self.prompt_edit.text())
            self.worker.decision.connect(self._on_decision)
            self.worker.frame_pushed.connect(self._on_pushed)
            self.worker.backlog.connect(self._on_backlog)
        self.worker.status.connect(self._set_status)
        self.worker.ready.connect(self._on_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_ready(self) -> None:
        self.hush_btn.setEnabled(True)
        if self.source_wav is not None:
            self._start_file_feed()
            return
        try:
            device = self.device_combo.currentData()
            self.mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=SAMPLE_RATE // 10, device=device,
                callback=self._mic_cb)
            self.mic_stream.start()
            self.capture_health_label.setText(
                f"capture: active · {self.device_combo.currentText()} · "
                "0 callback errors")
        except Exception as exc:
            self._on_failed(f"microphone: {exc}")

    def _start_file_feed(self) -> None:
        """Replay a file at wall-clock pace — the pacing the models require;
        after EOF keep feeding silence so turn-ends and SPEAK can happen."""
        try:
            import soundfile as _sf
            pcm, sr = _sf.read(str(self.source_wav), dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm[:, 0]
            if sr != SAMPLE_RATE:
                idx = np.linspace(0, pcm.shape[0] - 1,
                                  int(pcm.shape[0] * SAMPLE_RATE / sr))
                pcm = pcm[idx.astype(int)].astype(np.float32)
        except Exception as exc:
            self._on_failed(f"file read: {exc}")
            return
        state = {"i": 0}
        block = SAMPLE_RATE // 10

        def feed() -> None:
            if self.worker is None or self.pause_btn.isChecked():
                return
            i = state["i"]
            piece = pcm[i:i + block]
            if piece.size < block:
                piece = np.zeros(block, np.float32)
            state["i"] = i + block
            if hasattr(self.worker, "push_audio"):
                self.worker.push_audio(piece)
            else:
                self.worker.audio_q.put(piece)
            self._mic_level = float(np.sqrt(np.mean(np.square(piece))))

        self._file_timer = QTimer(self)
        self._file_timer.timeout.connect(feed)
        self._file_timer.start(100)

    def _mic_cb(self, indata, frames, time_info, status) -> None:
        if status:
            self._capture_errors += 1
            self._capture_error_text = str(status)
        mono = indata[:, 0].copy()
        if self.pause_btn.isChecked():
            self._mic_level = 0.0
            return
        if self.rec_check.isChecked():
            self._rec.append(mono)
        if self.worker is not None:
            if hasattr(self.worker, "push_audio"):
                self.worker.push_audio(mono)
            else:
                self.worker.audio_q.put(mono)
        # audio thread: only record the level; a UI timer paints it
        self._mic_level = float(np.sqrt(np.mean(np.square(mono))))

    def _stop(self) -> None:
        if self._file_timer is not None:
            self._file_timer.stop()
            self._file_timer = None
        if self._rec:
            out = WORK_DIR / time.strftime("capture_%Y%m%d_%H%M%S.wav")
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, np.concatenate(self._rec), SAMPLE_RATE)
                self._set_status(f"capture saved: {out.name}")
            except Exception as exc:
                self._set_status(f"capture save failed: {exc}")
            self._rec = []
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
        if self.worker is not None:
            self.worker.stop()
        self.device_combo.setEnabled(True)
        self.capture_health_label.setText(
            f"capture: idle · {self._capture_errors} callback errors")
        self.stop_btn.setEnabled(False)
        self.hush_btn.setEnabled(False)

    def _hush(self) -> None:
        if self.worker is not None and hasattr(self.worker, "hush"):
            self.worker.hush()

    def _on_finished(self) -> None:
        self.device_combo.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.hush_btn.setEnabled(False)

    def _on_failed(self, msg: str) -> None:
        self._set_status(f"FAILED: {msg}")
        self._stop()

    # -- signals from the worker ---------------------------------------------
    def _on_pushed(self, seq: int) -> None:
        self.counts["frames"] = seq
        self._refresh_counts()

    def _on_backlog(self, seconds: float, dropped_s: int) -> None:
        if dropped_s:
            self.metrics_label.setText(
                f"BEHIND: engine slower than realtime — {seconds:.1f}s queued, "
                f"{dropped_s}s of oldest audio dropped (shown, never silent)")
        elif seconds > 2.0:
            self.metrics_label.setText(f"engine catching up: {seconds:.1f}s queued")

    def _on_decision(self, is_speak: bool, text: str, ms_total: float) -> None:
        self.ticker.push(is_speak)
        self.counts["speak" if is_speak else "listen"] += 1
        if is_speak:
            self._set_status("MODEL CHOSE TO SPEAK")
            if text:
                self.convo.append(f"<b>assistant:</b> {text}")
        self.metrics_label.setText(f"last frame: {ms_total:.0f} ms end-to-end")
        self._refresh_counts()

    def _on_j_speaker(self, label: str) -> None:
        self.convo.append(
            f'<span style="color:#888">[speaker {label}] · audio turn</span>')
        self.ticker.push(False)

    def _on_j_reply(self, reply: str) -> None:
        self.counts["turns"] += 1
        self.convo.append(f"<b>Jaeger:</b> {reply}")
        self.ticker.push(True)
        self._refresh_counts()

    def _on_j_barge(self) -> None:
        self.convo.append(
            '<span style="color:#c90">interrupted · listening continued</span>')
        self.ticker.interrupt()

    def _refresh_counts(self) -> None:
        c = self.counts
        if self.engine_jaeger.isChecked():
            self.counts_label.setText(
                f"{c['turns']} audio-native turns · input transcript disabled")
        else:
            self.counts_label.setText(
                f"{c['frames']} frames → {c['listen']} listen, {c['speak']} SPEAK")

    def _paint_tick(self) -> None:
        self.waveform.push(self._mic_level)
        w = self.worker
        player = getattr(w, "player", None) if w is not None else None
        self.agent_waveform.push(player.level if player is not None else 0.0)
        if self.mic_stream is not None:
            detail = (f" · last: {self._capture_error_text}"
                      if self._capture_error_text else "")
            self.capture_health_label.setText(
                f"capture: active · {self.device_combo.currentText()} · "
                f"{self._capture_errors} callback errors{detail}")
            self.capture_health_label.setStyleSheet(
                "color:#c33;" if self._capture_errors else "color:#397;")
        if (self.engine_minicpm.isChecked() and w is not None
                and getattr(w, "mute_while_speaking", False)
                and getattr(w, "player", None) is not None
                and w.player.playing.is_set()):
            self.counts_label.setStyleSheet("color:#c90;")
            if "muted" not in self.counts_label.text():
                self.counts_label.setText(
                    self.counts_label.text() + "   ·   mic→engine MUTED (assistant speaking)")
        else:
            self.counts_label.setStyleSheet("")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # -- camera --------------------------------------------------------------
    def _toggle_camera(self, on: bool) -> None:
        if not on:
            if self.camera is not None:
                self.camera.stop()
                self.camera = None
            self.cam_label.setText("camera off")
            self.cam_label.setPixmap(QPixmap())
            return
        cams = QMediaDevices.videoInputs()
        if not cams:
            self.cam_label.setText("no camera found")
            self.video_check.setChecked(False)
            return
        self.camera = QCamera(cams[0])
        self.capture = QMediaCaptureSession()
        self.sink = QVideoSink()
        self.capture.setCamera(self.camera)
        self.capture.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self._on_video_frame)
        self.camera.start()

    def _on_video_frame(self, frame) -> None:
        img: QImage = frame.toImage()
        if img.isNull():
            return
        self.cam_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.worker is not None and self.video_check.isChecked():
            # one JPEG per second is plenty — the engine embeds per 1s frame
            now = time.monotonic()
            if now - getattr(self, "_last_jpeg", 0.0) >= 1.0:
                self._last_jpeg = now
                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                img.scaledToWidth(448, Qt.SmoothTransformation).save(buf, "JPG", 85)
                self.worker.set_jpeg(bytes(buf.data()))

    def closeEvent(self, ev) -> None:
        self._stop()
        super().closeEvent(ev)


def check() -> int:
    """No-model sanity for the canonical Jaeger demo surface."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication([sys.argv[0]])
    win = OmniDuplexWindow()
    win.show()
    problems = []
    optional = []
    try:
        from jaeger_agent_omni.runtime.aec import available
        from jaeger_agent_omni.live import _ready_chunks
        chunks, _ = _ready_chunks(
            "A deliberately long opening clause, followed by the answer.", 0)
        if not chunks:
            problems.append("clause streaming check failed")
        if not available():
            problems.append("SpeexDSP missing (macOS: brew install speexdsp)")
    except Exception as exc:
        problems.append(f"Jaeger runtime import: {exc}")
    if not JAEGER_MODEL.exists():
        problems.append(f"Jaeger model missing: {JAEGER_MODEL}")
    if not JAEGER_STOCK_NODE.exists():
        problems.append(
            f"Whisper production-candidate node missing: {JAEGER_STOCK_NODE}")
    if not JAEGER_CLEANROOM_NODE.exists():
        problems.append(
            f"Clean-room candidate node missing: {JAEGER_CLEANROOM_NODE}")
    if sd.query_devices(kind="input") is None:
        problems.append("no input audio device")
    if SHIM is None or not SHIM.exists() or not LLM_GGUF.exists():
        optional.append("MiniCPM reference unavailable (does not block Jaeger)")
    for msg in problems:
        print(f"CHECK FAIL: {msg}")
    for msg in optional:
        print(f"CHECK NOTE: {msg}")
    if not problems:
        print("demo --check ok: window, Jaeger model, audio-native runtime, "
              "SpeexDSP AEC, clause streaming, and mic present")
        if not optional:
            print(f"MiniCPM reference detected: {SHIM}")
    win.close()
    app.quit()
    return 1 if problems else 0


def minicpm_selftest(seconds: int) -> int:
    """REAL end-to-end test, no window: loads the engine and replays the
    question.wav asset through the EXACT worker path the GUI uses — 0.1s
    blocks at wall-clock pace (the pacing the burst-fed tests lacked), then
    trailing silence, which is the moment the model should choose SPEAK.
    Prints every 1Hz decision; the summary is the experiment's readout."""
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication(sys.argv)
    wav = _find_reference_resource("assets/question.wav", "OMNI_QUESTION_WAV")
    if wav is None or not LLM_GGUF.exists():
        print("selftest needs question.wav + the model")
        return 1
    pcm, sr = sf.read(wav, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sr != SAMPLE_RATE:
        idx = np.linspace(0, pcm.shape[0] - 1,
                          int(pcm.shape[0] * SAMPLE_RATE / sr))
        pcm = pcm[idx.astype(int)]

    worker = EngineWorker(language="en", send_video=False,
                          mute_while_speaking=False, system_prompt="")
    tally = {"speak": 0, "listen": 0, "pushed": 0}
    # The worker redirects fd 1/2 to engine.log for the whole session; keep a
    # dup of the real console so THIS test's output still reaches the terminal.
    console = os.fdopen(os.dup(1), "w", buffering=1)

    def say(msg: str) -> None:
        print(msg, file=console, flush=True)

    def on_decision(is_speak: bool, text: str, ms: float) -> None:
        kind = "SPEAK " if is_speak else "listen"
        tally["speak" if is_speak else "listen"] += 1
        say(f"  [{kind}] {ms:6.0f}ms  {text[:70]}")

    worker.status.connect(lambda m: say(f"status: {m}"))
    worker.decision.connect(on_decision)
    worker.frame_pushed.connect(lambda n: tally.__setitem__("pushed", n))
    worker.failed.connect(lambda m: (say(f"FAILED: {m}"), app.exit(2)))

    block = SAMPLE_RATE // 10
    state = {"i": 0, "t0": None}

    feeder = QTimer(app)

    def feed() -> None:
        if state["t0"] is None:
            state["t0"] = time.monotonic()
        elapsed = time.monotonic() - state["t0"]
        if elapsed >= seconds:
            feeder.stop()
            say(f"\n{seconds}s elapsed — stopping session…")
            worker.stop()
            return
        i = state["i"]
        piece = pcm[i:i + block]
        if piece.size < block:               # wav done -> trailing silence,
            piece = np.zeros(block, np.float32)   # the natural turn boundary
        state["i"] = i + block
        worker.audio_q.put(piece)

    worker.ready.connect(lambda: feeder.start(100))
    feeder.timeout.connect(feed)

    def watch_voice() -> None:
        player = worker.player
        if player is not None:
            tally["peak_voice"] = max(tally.get("peak_voice", 0.0), player.level)
    voice_watch = QTimer()
    voice_watch.timeout.connect(watch_voice)
    worker.ready.connect(lambda: voice_watch.start(25))

    # Watchdog: a broken engine can load and then spin forever without ever
    # going live (seen with a mismatched backbone) — fail loudly instead.
    def not_live() -> None:
        if not feeder.isActive() and state["t0"] is None:
            say("FAILED: session not live after 240s — engine hung "
                "(see engine.log)")
            worker.stop()
            app.exit(3)
    QTimer.singleShot(240_000, not_live)

    def done() -> None:
        peak = tally.get("peak_voice", 0.0)
        say(f"agent voice peak level (real playback samples): {peak:.3f} "
            f"{'— voice output CONFIRMED' if peak > 0.01 else '— no voice played'}")
        speech_s = min(pcm.size // SAMPLE_RATE + 1, seconds)
        say(f"\nselftest summary: {tally['pushed']} frames pushed "
            f"({speech_s}s speech, then silence) -> "
            f"{tally['listen']} listen, {tally['speak']} SPEAK")
        say(f"engine log: {ENGINE_LOG}")
        if tally["speak"]:
            say("RESULT: the model chose to speak — native duplex CONFIRMED")
        else:
            say("RESULT: no SPEAK decision — a REGRESSION: duplex was "
                "confirmed 2026-08-20 (SPEAK at frame 9, 'Paris.'). Check "
                "pacing (must be wall-clock), language, and engine.log")
        app.exit(0 if tally["pushed"] and tally["speak"] and peak > 0.01 else 2)

    worker.finished.connect(done)
    worker.start()
    return app.exec()


def jaeger_selftest(wav_path: Path, seconds: int, threshold: float,
                    silence_ms: float, cooldown: float,
                    max_context: int, front_end: str = "tower") -> int:
    """Paced-file test of the exact Jaeger worker used by the window."""
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication([sys.argv[0]])
    pcm, sr = sf.read(wav_path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = max(1, round(len(pcm) * SAMPLE_RATE / sr))
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n),
                        np.arange(len(pcm)), pcm).astype(np.float32)

    worker = JaegerEngineWorker(
        threshold=threshold, silence_ms=silence_ms, cooldown=cooldown,
        max_context=max_context, speakers=2, front_end=front_end)
    tally = {"replies": 0, "failed": False, "speakers": []}
    state = {"i": 0, "started": 0.0}
    def say(msg: str) -> None:
        print(msg, flush=True)

    def feed() -> None:
        deadline = state["started"] + seconds
        next_at = state["started"]
        while time.monotonic() < deadline and worker.isRunning():
            i = state["i"]
            block = pcm[i:i + SAMPLE_RATE // 10]
            if block.size < SAMPLE_RATE // 10:
                block = np.zeros(SAMPLE_RATE // 10, np.float32)
            state["i"] += SAMPLE_RATE // 10
            worker.push_audio(block)
            next_at += 0.1
            time.sleep(max(0.0, next_at - time.monotonic()))
        say(f"{seconds}s paced input complete — stopping")
        worker.stop()

    def ready() -> None:
        state["started"] = time.monotonic()
        say("paced input started")
        threading.Thread(target=feed, daemon=True).start()

    def replied(text: str) -> None:
        tally["replies"] += 1
        say(f"JAEGER: {text}")

    def failed(msg: str) -> None:
        tally["failed"] = True
        say(f"FAILED: {msg}")
        worker.stop()

    def done() -> None:
        import statistics
        latency = (f" · median END→AUDIO "
                   f"{statistics.median(worker.end_audio_ms):.0f}ms across "
                   f"{len(worker.end_audio_ms)} starts"
                   if worker.end_audio_ms else "")
        say(f"selftest: {tally['replies']} replies · "
            f"{worker.fed_samples/SAMPLE_RATE:.1f}s fed · "
            f"{worker.dropped_samples/SAMPLE_RATE:.1f}s dropped · "
            f"{worker.barge_count} barge-ins · {worker.rolls} rolls · "
            f"speakers {','.join(tally['speakers']) or 'none'}{latency}")
        app.exit(0 if tally["replies"] and not tally["failed"] else 2)

    worker.status.connect(lambda msg: say(f"status: {msg}"))
    worker.metrics.connect(lambda msg: say(f"metrics: {msg}"))
    worker.ready.connect(ready, Qt.QueuedConnection)
    worker.reply.connect(replied)
    worker.speaker.connect(lambda label: tally["speakers"].append(label))
    worker.failed.connect(failed)
    worker.finished.connect(done, Qt.QueuedConnection)
    worker.start()
    def timed_out() -> None:
        if worker.isRunning():
            tally["failed"] = True
            say("FAILED: self-test exceeded duration plus 90s shutdown allowance")
            worker.stop()

    QTimer.singleShot(max(240, seconds + 90) * 1000, timed_out)
    return app.exec()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="build the window and verify local dependencies")
    ap.add_argument("--selftest-jaeger", type=Path, metavar="WAV",
                    help="pace WAV through the exact Jaeger GUI worker")
    ap.add_argument("--selftest-minicpm", type=int, nargs="?", const=25,
                    metavar="SECONDS", help="run the optional reference engine")
    ap.add_argument("--seconds", type=int, default=20,
                    help="Jaeger self-test duration including trailing silence")
    ap.add_argument("--threshold", type=float, default=0.012,
                    help="Jaeger self-test room RMS floor")
    ap.add_argument("--silence-ms", type=float, default=700.0,
                    help="Jaeger self-test endpoint silence")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="Jaeger self-test reply cooldown")
    ap.add_argument("--max-context", type=int, default=6000,
                    help="Jaeger self-test context-roll threshold")
    ap.add_argument("--front-end", choices=("tower", "stock", "cleanroom"),
                    default="tower",
                    help="Jaeger self-test audio node")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.selftest_jaeger:
        return jaeger_selftest(
            args.selftest_jaeger, args.seconds, args.threshold,
            args.silence_ms, args.cooldown, args.max_context, args.front_end)
    if args.selftest_minicpm is not None:
        return minicpm_selftest(args.selftest_minicpm)
    app = QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    win = OmniDuplexWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
