"""Acoustic echo cancellation against the waveform Jaeger is playing.

SpeexDSP is deliberately loaded with :mod:`ctypes` instead of a Python
wrapper.  The Python packages named ``speexdsp`` are stale on modern Python,
while the C ABI has been stable for years.  Install the system library (for
example ``brew install speexdsp`` on macOS) and this module finds it.

The canceller receives microphone blocks plus the exact output waveform and
removes the room echo with an adaptive filter.  Near-end speech is not used as
a gate: Speex's double-talk handling leaves it in the residual.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time
from pathlib import Path

SPEEX_ECHO_SET_SAMPLING_RATE = 24


def _library_path() -> str:
    override = os.environ.get("JAEGER_SPEEXDSP_LIB")
    candidates = [
        override,
        ctypes.util.find_library("speexdsp"),
        "/opt/homebrew/lib/libspeexdsp.dylib",
        "/usr/local/lib/libspeexdsp.dylib",
        "/usr/lib/libspeexdsp.so",
        "/usr/local/lib/libspeexdsp.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        # A soname returned by find_library need not be an absolute path.
        if "/" not in candidate or Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "SpeexDSP was not found; install it (macOS: brew install speexdsp) "
        "or pass --no-aec"
    )


def available() -> bool:
    try:
        _library_path()
        return True
    except RuntimeError:
        return False


class SpeexEchoCanceller:
    """Single-channel, fixed-rate Speex adaptive echo canceller.

    ``start_playback`` establishes a wall-clock mapping for the known far-end
    waveform.  ``process`` is then called from the microphone callback.  The
    adaptive filter absorbs normal sound-card and acoustic delay (up to
    ``filter_ms``); callers do not need to guess a fixed delay.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20,
                 filter_ms: int = 500):
        import numpy as np

        self.np = np
        self.sample_rate = int(sample_rate)
        self.frame_size = self.sample_rate * int(frame_ms) // 1000
        self.filter_length = self.sample_rate * int(filter_ms) // 1000
        self.library_path = _library_path()
        self._lib = ctypes.CDLL(self.library_path)
        ptr = ctypes.POINTER(ctypes.c_int16)
        self._sample_ptr = ptr
        self._lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.speex_echo_state_init.restype = ctypes.c_void_p
        self._lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        self._lib.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p, ptr, ptr, ptr]
        self._lib.speex_echo_ctl.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._state = self._lib.speex_echo_state_init(
            self.frame_size, self.filter_length)
        if not self._state:
            raise RuntimeError("SpeexDSP could not allocate echo state")
        rate = ctypes.c_int(self.sample_rate)
        if self._lib.speex_echo_ctl(
                self._state, SPEEX_ECHO_SET_SAMPLING_RATE,
                ctypes.byref(rate)) != 0:
            self.close()
            raise RuntimeError("SpeexDSP rejected the sampling rate")

        self._lock = threading.Lock()
        self._reference = None
        self._started_at = 0.0
        self._tail_until = 0.0
        self.raw_energy = 0.0
        self.clean_energy = 0.0
        self.samples_cancelled = 0

    def close(self):
        state = getattr(self, "_state", None)
        if state:
            self._lib.speex_echo_state_destroy(state)
            self._state = None

    def __del__(self):
        self.close()

    @staticmethod
    def _mono(samples):
        import numpy as np

        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return np.ascontiguousarray(x.reshape(-1))

    def start_playback(self, samples, sample_rate: int,
                       started_at: float | None = None):
        """Register the exact waveform immediately before output starts."""
        np = self.np
        wav = self._mono(samples)
        if int(sample_rate) != self.sample_rate and len(wav):
            # Linear interpolation is plenty for the far-end reference.  The
            # adaptive filter models the actual loudspeaker/room coloration.
            n = max(1, round(len(wav) * self.sample_rate / int(sample_rate)))
            old = np.arange(len(wav), dtype=np.float64)
            new = np.arange(n, dtype=np.float64) * int(sample_rate) / self.sample_rate
            wav = np.interp(new, old, wav).astype(np.float32)
        start = time.monotonic() if started_at is None else float(started_at)
        with self._lock:
            self._reference = wav
            self._started_at = start
            self._tail_until = start + len(wav) / self.sample_rate + \
                self.filter_length / self.sample_rate

    @property
    def playback_age(self) -> float:
        with self._lock:
            return max(0.0, time.monotonic() - self._started_at)

    def _reference_for(self, n: int, captured_at: float):
        np = self.np
        with self._lock:
            reference = self._reference
            started = self._started_at
            tail_until = self._tail_until
        if reference is None or captured_at > tail_until:
            return None
        # The callback runs after this input block was captured.  Align its
        # beginning with the corresponding far-end time.  Filter history then
        # learns the remaining device/acoustic delay.
        end = round((captured_at - started) * self.sample_rate)
        begin = end - n
        out = np.zeros(n, dtype=np.float32)
        src0, src1 = max(0, begin), min(len(reference), end)
        if src1 > src0:
            out[src0 - begin:src1 - begin] = reference[src0:src1]
        return out

    def process(self, microphone, captured_at: float | None = None):
        """Return echo-cancelled float32 audio with identical length."""
        np = self.np
        mic = self._mono(microphone)
        now = time.monotonic() if captured_at is None else float(captured_at)
        ref = self._reference_for(len(mic), now)
        if ref is None or not len(mic):
            return mic

        pieces = []
        for start in range(0, len(mic), self.frame_size):
            valid = min(self.frame_size, len(mic) - start)
            rec_f = np.zeros(self.frame_size, dtype=np.float32)
            play_f = np.zeros(self.frame_size, dtype=np.float32)
            rec_f[:valid] = mic[start:start + valid]
            play_f[:valid] = ref[start:start + valid]
            rec = np.ascontiguousarray(
                np.clip(rec_f * 32767.0, -32768, 32767).astype(np.int16))
            play = np.ascontiguousarray(
                np.clip(play_f * 32767.0, -32768, 32767).astype(np.int16))
            clean = np.empty(self.frame_size, dtype=np.int16)
            self._lib.speex_echo_cancellation(
                self._state,
                rec.ctypes.data_as(self._sample_ptr),
                play.ctypes.data_as(self._sample_ptr),
                clean.ctypes.data_as(self._sample_ptr))
            pieces.append(clean[:valid].astype(np.float32) / 32768.0)
        out = np.concatenate(pieces) if pieces else mic.copy()
        self.raw_energy += float(np.dot(mic, mic))
        self.clean_energy += float(np.dot(out, out))
        self.samples_cancelled += len(mic)
        return out

    @property
    def attenuation_db(self) -> float:
        import math

        if not self.samples_cancelled:
            return 0.0
        return 10.0 * math.log10(
            (self.raw_energy + 1e-12) / (self.clean_energy + 1e-12))


def demo() -> None:
    """Corpus-style synthetic echo + double-talk check, not one clip."""
    import numpy as np

    rate, frame = 16000, 320
    rng = np.random.default_rng(0)
    cancellations, near_ratios = [], []
    for seed in range(8):
        far = rng.normal(0, 0.12, rate * 8).astype(np.float32)
        far = np.convolve(far, np.ones(5) / 5, mode="same").astype(np.float32)
        delay = 800 + seed * 83
        mic = np.zeros_like(far)
        mic[delay:] = 0.55 * far[:-delay]
        mic[delay + 400:] += 0.18 * far[:-(delay + 400)]
        near = np.zeros_like(far)
        near[rate * 6:rate * 7] = np.convolve(
            rng.normal(0, 0.12, rate), np.ones(3) / 3,
            mode="same").astype(np.float32)
        mic += near

        aec = SpeexEchoCanceller(rate)
        # Offline timestamps reproduce the same waveform/time mapping used by
        # the live callbacks without sleeping for eight seconds per case.
        aec.start_playback(far, rate, started_at=0.0)
        cleaned = []
        for i in range(0, len(far), frame):
            cleaned.append(aec.process(
                mic[i:i + frame], captured_at=(i + frame) / rate))
        cleaned = np.concatenate(cleaned)
        echo_sl = slice(rate * 4, rate * 6)
        near_sl = slice(rate * 6, rate * 7)
        rms = lambda x, sl: float(np.sqrt(np.mean(x[sl] ** 2)))
        cancellations.append(20 * np.log10(
            (rms(mic, echo_sl) + 1e-9) /
            (rms(cleaned, echo_sl) + 1e-9)))
        near_ratios.append(rms(cleaned, near_sl) / (rms(near, near_sl) + 1e-9))
        aec.close()

    assert np.median(cancellations) > 12.0, cancellations
    assert np.median(near_ratios) > 0.70, near_ratios
    print(f"runtime.aec: 8-case median echo attenuation "
          f"{np.median(cancellations):.1f} dB; double-talk survival "
          f"{100*np.median(near_ratios):.0f}%")


if __name__ == "__main__":
    demo()
