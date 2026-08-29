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
CONTIGUOUS_MAX_S = 600      # reply-stream capacity; beyond counts as overflow
CONTIGUOUS_STALE_GAP_S = 10.0   # append this long after the tail = new reply
ECHO_LOCK_TARGET_S = 0.08       # fed-vs-echo depth the servo steers toward


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

    ``reference_mode`` selects how the far-end stream is reconstructed:

    ``wallclock``
        The measured baseline: every microphone block independently re-derives
        its reference slice from wall-clock time.  Callback jitter therefore
        appears to Speex as far-end discontinuities.
    ``contiguous`` (experimental, opt-in)
        One reply-level reference stream.  Clauses are appended at their
        playback times with real silence between them, and ``process``
        consumes the stream sequentially, advancing exactly one block per
        call.  Wall clock is only used to *audit* alignment: the measured
        error is recorded (``estimated_lag_ms`` / ``estimated_drift_ppm``)
        and corrected in discrete jumps (``alignment_corrections``) when it
        exceeds one Speex frame.  A contiguous stream does NOT by itself fix
        independent ADC/DAC clock drift; the drift telemetry exists to
        measure that hypothesis in the physical room.
    """

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20,
                 filter_ms: int = 500, library_path: str | Path | None = None,
                 reference_mode: str = "wallclock",
                 output_latency_ms: float = 0.0):
        import numpy as np

        if reference_mode not in ("wallclock", "contiguous"):
            raise ValueError(
                f"reference_mode must be wallclock or contiguous, "
                f"got {reference_mode!r}")
        self.reference_mode = reference_mode
        # Measured constant output-path latency (chirp probe 2026-08-29:
        # 614 ms on External Headphones, zero spread/drift).  Shifting the
        # reference timeline by it moves the echo inside the adaptive
        # filter's span instead of asking the filter to model a delay it
        # physically cannot reach.  0.0 preserves the measured baseline.
        self.output_latency_ms = float(output_latency_ms)
        self._output_latency_s = self.output_latency_ms / 1000.0

        self.np = np
        self.sample_rate = int(sample_rate)
        self.frame_size = self.sample_rate * int(frame_ms) // 1000
        self.filter_length = self.sample_rate * int(filter_ms) // 1000
        self.library_path = str(library_path) if library_path else _library_path()
        if "/" in self.library_path and not Path(self.library_path).is_file():
            raise RuntimeError(f"SpeexDSP library does not exist: {self.library_path}")
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
        # Contiguous-mode reply stream state and telemetry.  The fields exist
        # in both modes so report writers never need mode-specific guards.
        self._stream = None            # reply-level far-end sample stream
        self._stream_t0 = 0.0          # wall-clock time of stream sample 0
        self._read_pos = None          # next sample ``process`` consumes
        self._uncorrected_read = None  # read position had we never corrected
        self._new_reply = True
        self._offset_first = None      # (time, uncorrected offset) for drift
        self._offset_last = None
        self._align_tolerance = self.frame_size    # one Speex frame (20 ms)
        self.reference_underflow_samples = 0
        self.reference_overflow_samples = 0
        self.alignment_corrections = 0
        self.estimated_lag_ms = 0.0
        self.estimated_drift_ppm = 0.0
        # Echo lock (contiguous mode): wall clock only provides the INITIAL
        # alignment; afterwards the read position is servoed onto the echo
        # itself by periodic mic-vs-stream cross-correlation.  Measured need:
        # the capture clock drifts ~1,250 ppm against wall time and offline
        # replay shows cancellation dies when the reference runs >20 ms late,
        # so a fixed offset cannot hold for a 30-second reply.
        self._mic_recent = np.zeros(0, dtype=np.float32)
        self._samples_since_lock = 0
        self._lock_offset = 0              # echo-measured refinement, samples
        self.echo_lock_corrections = 0
        self.echo_lock_lag_ms = None       # last measured fed-vs-echo depth
        self.echo_lock_confidence = 0.0
        self.playback_generation = 0
        self.raw_energy = 0.0
        self.clean_energy = 0.0
        self.samples_cancelled = 0
        self.reference_active = False
        self.block_playback_age = 0.0
        self.reference_rms = 0.0
        self.block_attenuation_db = 0.0
        self.raw_reference_correlation = 0.0
        self.residual_reference_correlation = 0.0

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

    def _resampled(self, samples, sample_rate: int):
        np = self.np
        wav = self._mono(samples)
        if int(sample_rate) != self.sample_rate and len(wav):
            # Linear interpolation is plenty for the far-end reference.  The
            # adaptive filter models the actual loudspeaker/room coloration.
            n = max(1, round(len(wav) * self.sample_rate / int(sample_rate)))
            old = np.arange(len(wav), dtype=np.float64)
            new = np.arange(n, dtype=np.float64) * int(sample_rate) / self.sample_rate
            wav = np.interp(new, old, wav).astype(np.float32)
        return wav

    def start_playback(self, samples, sample_rate: int,
                       started_at: float | None = None):
        """Register the exact waveform immediately before output starts."""
        if self.reference_mode == "contiguous":
            # Mode-agnostic callers keep working: a bare start is a one-clause
            # reply in the contiguous stream.
            self.begin_reply_playback()
            return self.append_playback(samples, sample_rate, started_at)
        wav = self._resampled(samples, sample_rate)
        start = (time.monotonic() if started_at is None
                 else float(started_at)) + self._output_latency_s
        with self._lock:
            self._reference = wav
            self._started_at = start
            self._tail_until = start + len(wav) / self.sample_rate + \
                self.filter_length / self.sample_rate
            self.playback_generation += 1

    def begin_reply_playback(self) -> None:
        """Mark the next appended clause as the start of a new reply.

        One assistant reply keeps ONE playback generation, so downstream
        barge/echo-floor state is no longer reset for every spoken clause.
        A no-op in wallclock mode.
        """
        if self.reference_mode != "contiguous":
            return
        with self._lock:
            self._new_reply = True

    def append_playback(self, samples, sample_rate: int,
                        started_at: float | None = None):
        """Append one clause to the reply-level far-end stream.

        Gaps between clause start times become literal zeros — silence the
        loudspeaker really produced.  A clause that arrives after ``process``
        already consumed its position counts the late samples as reference
        underflow; samples beyond the stream capacity count as overflow.
        """
        if self.reference_mode != "contiguous":
            raise RuntimeError("append_playback requires contiguous mode")
        np = self.np
        wav = self._resampled(samples, sample_rate)
        now = (time.monotonic() if started_at is None
               else float(started_at)) + self._output_latency_s
        with self._lock:
            # ``begin_reply_playback`` is the intended reply boundary; the
            # stale-gap fallback only nets a caller that forgot it.  Synth
            # pauses between clauses may legitimately exceed the filter tail.
            stale = (self._stream is not None and
                     now > self._tail_until + CONTIGUOUS_STALE_GAP_S)
            if self._new_reply or self._stream is None or stale:
                self._stream = np.zeros(0, dtype=np.float32)
                self._stream_t0 = now
                self._started_at = now
                self._read_pos = None
                self._uncorrected_read = None
                self._offset_first = None
                self._offset_last = None
                self._new_reply = False
                self._samples_since_lock = 0
                self._lock_offset = 0
                self.playback_generation += 1
            write = round((now - self._stream_t0) * self.sample_rate)
            if write < 0:
                write = 0
            cap = CONTIGUOUS_MAX_S * self.sample_rate
            end = write + len(wav)
            if end > cap:
                self.reference_overflow_samples += end - cap
                wav = wav[:max(0, cap - write)]
                end = write + len(wav)
            if self._read_pos is not None and write < self._read_pos:
                # Those positions were already fed to Speex as zeros.
                self.reference_underflow_samples += min(
                    len(wav), self._read_pos - write)
            if end > len(self._stream):
                self._stream = np.concatenate(
                    (self._stream,
                     np.zeros(end - len(self._stream), dtype=np.float32)))
            if len(wav):
                self._stream[write:end] = wav
            self._tail_until = (self._stream_t0 +
                                len(self._stream) / self.sample_rate +
                                self.filter_length / self.sample_rate)

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
            return None, None
        # The callback runs after this input block was captured.  Align its
        # beginning with the corresponding far-end time.  Filter history then
        # learns the remaining device/acoustic delay.
        end = round((captured_at - started) * self.sample_rate)
        begin = end - n
        out = np.zeros(n, dtype=np.float32)
        src0, src1 = max(0, begin), min(len(reference), end)
        if src1 > src0:
            out[src0 - begin:src1 - begin] = reference[src0:src1]
        # Correlation searches the entire adaptive-filter delay range. The
        # microphone receives a delayed/coloured copy, not the same-time
        # digital block, so direct sample equality is the wrong test.
        history_begin = max(0, begin - self.filter_length)
        history_end = min(len(reference), max(0, end))
        history = reference[history_begin:history_end]
        return out, history

    def _try_echo_lock(self, stream) -> None:
        """Servo the read position onto the echo actually in the microphone.

        Called with the lock held.  Cross-correlates the last ~0.8 s of raw
        mic against the reference stream around the read position, measures
        the fed-vs-echo depth, and nudges the read position so the echo sits
        ECHO_LOCK_TARGET_S behind the fed reference — inside the filter span
        with an early-side margin, since offline replay shows cancellation
        survives an early reference but dies ~20 ms late.
        """
        np = self.np
        dec = 4
        window = int(0.8 * self.sample_rate)
        mic = self._mic_recent
        if len(mic) < window or self._read_pos is None:
            return
        mic = mic[-window:]
        lo = max(0, self._read_pos - window - self.filter_length)
        hi = min(len(stream),
                 self._read_pos + int(0.25 * self.sample_rate))
        segment = stream[lo:hi]
        a = mic[:(window // dec) * dec].reshape(-1, dec).mean(1)
        b = segment[:(len(segment) // dec) * dec].reshape(-1, dec).mean(1)
        if len(b) <= len(a) + 8:
            return
        a = a - a.mean()
        b = b - b.mean()
        corr = np.correlate(b, a, mode="valid")
        cumulative = np.concatenate((
            np.zeros(1), np.cumsum(np.square(b), dtype=np.float64)))
        energy = cumulative[len(a):] - cumulative[:-len(a)]
        normalized = np.abs(corr) / np.sqrt(
            np.maximum(energy * float(np.dot(a, a)), 1e-12))
        peak = int(np.argmax(normalized))
        confidence = float(normalized[peak])
        self.echo_lock_confidence = confidence
        if confidence < 0.15:
            return                       # silence or no audible echo: hold
        echo_end = lo + peak * dec + window   # stream idx of mic[-1]'s echo
        depth = self._read_pos - echo_end     # fed-vs-echo delay, samples
        self.echo_lock_lag_ms = 1000.0 * depth / self.sample_rate
        # Deadband, not a servo-to-setpoint: any depth Speex can model is
        # left alone (constant in-band delay is the filter's job, and
        # correcting sub-frame measurement noise every cycle thrashes the
        # filter into Speex's divergence reset).  Only a depth outside the
        # usable span — echo ahead of the fed reference, or beyond the
        # filter — jumps back to the early-side target.
        low = self.frame_size // 2
        high = self.filter_length - self.sample_rate // 10
        if depth < low or depth > high:
            adjust = int(ECHO_LOCK_TARGET_S * self.sample_rate) - depth
            self._read_pos += adjust
            self._lock_offset += adjust
            self.echo_lock_corrections += 1

    def _contiguous_reference_for(self, n: int, captured_at: float, mic):
        """Consume the reply stream sequentially, echo-locked.

        The stream advances exactly ``n`` samples per call, so Speex sees a
        continuous far-end signal.  Wall clock provides only the initial and
        between-gap alignment; the periodic echo lock then refines a
        ``_lock_offset`` that absorbs output latency AND the measured
        ~1,250 ppm capture-clock drift a fixed offset cannot hold.
        """
        np = self.np
        with self._lock:
            stream = self._stream
            t0 = self._stream_t0
            tail_until = self._tail_until
            if stream is None:
                return None, None
            # ideal_raw audits the pure wall-clock mapping (drift telemetry
            # must not see the servo's own moves); ideal_eff adds the
            # echo-locked refinement and drives alignment.
            ideal_raw = round((captured_at - t0) * self.sample_rate) - n
            ideal_begin = ideal_raw + self._lock_offset
            if captured_at > tail_until:
                # Between-clause or post-reply quiet: resync the audit
                # positions instead of freezing them, so the resume after a
                # legitimate gap is neither a fake alignment correction nor a
                # fake drift trend.  The echo-locked offset is retained.
                if self._read_pos is not None:
                    self._read_pos = ideal_begin + n
                    self._uncorrected_read = ideal_raw + n
                    self._offset_first = None
                    self._offset_last = None
                return None, None
            if self._read_pos is None:
                self._read_pos = ideal_begin
                self._uncorrected_read = ideal_raw
            if mic is not None and len(mic):
                # Lock BEFORE appending this block: the retained window then
                # ends exactly at the current read position, so the measured
                # fed-vs-echo depth carries no one-block bias.
                self._samples_since_lock += len(mic)
                if self._samples_since_lock >= 2 * self.sample_rate:
                    self._samples_since_lock = 0
                    self._try_echo_lock(stream)
                    ideal_begin = ideal_raw + self._lock_offset
                keep = int(1.0 * self.sample_rate)
                self._mic_recent = np.concatenate(
                    (self._mic_recent, mic))[-keep:]
            raw_offset = ideal_raw - self._uncorrected_read
            error = ideal_begin - self._read_pos
            self.estimated_lag_ms = 1000.0 * error / self.sample_rate
            if self._offset_first is None:
                self._offset_first = (captured_at, raw_offset)
            self._offset_last = (captured_at, raw_offset)
            span = self._offset_last[0] - self._offset_first[0]
            if span > 1.0:
                self.estimated_drift_ppm = 1e6 * (
                    self._offset_last[1] - self._offset_first[1]) / (
                        span * self.sample_rate)
            if abs(error) > self._align_tolerance:
                self._read_pos += error
                self.alignment_corrections += 1
            begin = self._read_pos
            end = begin + n
            out = np.zeros(n, dtype=np.float32)
            src0, src1 = max(0, begin), min(len(stream), end)
            if src1 > src0:
                out[src0 - begin:src1 - begin] = stream[src0:src1]
            history_begin = max(0, begin - self.filter_length)
            history_end = min(len(stream), max(0, end))
            history = stream[history_begin:history_end]
            self._read_pos = end
            self._uncorrected_read += n
        return out, history

    @staticmethod
    def _max_reference_correlation(signal, history, decimation: int = 4):
        """Maximum normalized correlation over plausible acoustic delays.

        Four-sample boxcar decimation keeps this cheap enough for a callback
        while retaining the speech band needed to distinguish playback from
        independent near-end speech.
        """
        import numpy as np

        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        history = np.asarray(history, dtype=np.float32).reshape(-1)
        signal_n = (len(signal) // decimation) * decimation
        history_n = (len(history) // decimation) * decimation
        if signal_n < decimation * 8 or history_n < signal_n:
            return 0.0
        probe = signal[:signal_n].reshape(-1, decimation).mean(axis=1)
        far = history[:history_n].reshape(-1, decimation).mean(axis=1)
        probe = probe - np.mean(probe)
        far = far - np.mean(far)
        probe_energy = float(np.dot(probe, probe))
        if probe_energy < 1e-10:
            return 0.0
        numerator = np.correlate(far, probe, mode="valid")
        cumulative = np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.square(far), dtype=np.float64),
        ))
        window = len(probe)
        far_energy = cumulative[window:] - cumulative[:-window]
        denominator = np.sqrt(
            np.maximum(far_energy * probe_energy, 1e-20))
        return float(np.max(np.abs(numerator) / denominator))

    def process(self, microphone, captured_at: float | None = None):
        """Return echo-cancelled float32 audio with identical length."""
        np = self.np
        mic = self._mono(microphone)
        now = time.monotonic() if captured_at is None else float(captured_at)
        if self.reference_mode == "contiguous":
            ref, history = self._contiguous_reference_for(len(mic), now, mic)
        else:
            ref, history = self._reference_for(len(mic), now)
        if ref is None or not len(mic):
            self.reference_active = False
            self.block_playback_age = max(0.0, now - self._started_at)
            self.reference_rms = 0.0
            self.block_attenuation_db = 0.0
            self.raw_reference_correlation = 0.0
            self.residual_reference_correlation = 0.0
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
        raw_rms = float(np.sqrt(np.mean(np.square(mic))))
        clean_rms = float(np.sqrt(np.mean(np.square(out))))
        self.reference_active = True
        self.block_playback_age = max(0.0, now - self._started_at)
        self.reference_rms = float(np.sqrt(np.mean(np.square(ref))))
        self.block_attenuation_db = 20.0 * float(np.log10(
            (raw_rms + 1e-9) / (clean_rms + 1e-9)))
        self.raw_reference_correlation = self._max_reference_correlation(
            mic, history)
        self.residual_reference_correlation = \
            self._max_reference_correlation(out, history)
        return out

    @property
    def attenuation_db(self) -> float:
        import math

        if not self.samples_cancelled:
            return 0.0
        return 10.0 * math.log10(
            (self.raw_energy + 1e-12) / (self.clean_energy + 1e-12))


class EchoAwareBargeIn:
    """Classify residual speech using Jaeger's known playback waveform.

    Energy alone cannot distinguish a listener from uncancelled reverberation.
    This detector requires an energetic residual to be decorrelated from the
    far-end reference for a short confirmation interval. Correlated residuals
    are Jaeger's own voice and remain model input only after cancellation.
    """

    def __init__(self, sample_rate: int, speech_rms: float,
                 warmup_ms: float = 500.0, confirm_ms: float = 80.0,
                 echo_correlation: float = 0.45,
                 residual_echo_correlation: float = 0.35,
                 residual_margin: float = 1.1,
                 strong_margin: float = 1.5,
                 strong_attenuation_db: float = 2.5,
                 strong_confirm_ms: float = 200.0):
        self.sample_rate = int(sample_rate)
        self.speech_rms = float(speech_rms)
        self.warmup_s = float(warmup_ms) / 1000.0
        self.confirm_samples = max(
            1, round(self.sample_rate * float(confirm_ms) / 1000.0))
        self.echo_correlation = float(echo_correlation)
        self.residual_echo_correlation = float(residual_echo_correlation)
        self.residual_margin = float(residual_margin)
        self.strong_margin = float(strong_margin)
        self.strong_attenuation_db = float(strong_attenuation_db)
        self.strong_confirm_samples = max(
            1, round(self.sample_rate * float(strong_confirm_ms) / 1000.0))
        self._strong_samples = 0
        self._filter_proven = False
        self._generation = -1
        self._candidate_samples = 0
        self._echo_floor = self.speech_rms * 0.5

    @property
    def echo_floor(self) -> float:
        return self._echo_floor

    def reset(self, generation: int = -1) -> None:
        self._generation = int(generation)
        self._candidate_samples = 0
        self._strong_samples = 0
        self._filter_proven = False
        self._echo_floor = self.speech_rms * 0.5

    def update(self, aec: SpeexEchoCanceller, clean_rms: float,
               samples: int) -> bool:
        generation = int(aec.playback_generation)
        if generation != self._generation:
            self.reset(generation)

        clean_rms = float(clean_rms)
        # Classify the mixture, not the post-AEC residual.  An independent
        # near-end voice dilutes raw/reference coherence.  Searching the
        # residual itself can spuriously match some unrelated speech elsewhere
        # in the half-second reference history.
        # Echo remains coherent with the known playback both before and after
        # cancellation.  Independent near speech often leaves one accidental
        # correlation high, but not both; treating either one alone as echo
        # delays real interruptions well into an utterance.
        correlated = (
            aec.raw_reference_correlation >= self.echo_correlation and
            aec.residual_reference_correlation >=
            self.residual_echo_correlation)
        # During filter convergence, learn the room's residual echo envelope
        # but never mistake that startup transient for an interruption.
        if aec.block_playback_age < self.warmup_s:
            self._echo_floor = max(
                self.speech_rms * 0.5,
                0.85 * self._echo_floor + 0.15 * clean_rms)
            self._candidate_samples = 0
            return False

        # Correlation proves reference content is PRESENT; it cannot prove
        # the block is ONLY echo.  Genuine double-talk keeps the mixture
        # correlated for seconds (measured p95 843 ms).  The separating
        # feature is per-block attenuation: a converged filter SUPPRESSES
        # loud echo-only blocks (measured p5 3.05 dB) but cannot suppress
        # near speech (measured median 0.88 dB), so a loud block the filter
        # failed to attenuate is double-talk.  Guards against echo-only
        # bursts: mature floor, 250 ms sustain, and a strong candidate never
        # teaches the floor its own energy.
        # "Filter failed to suppress this block" is only evidence of
        # double-talk once the filter has PROVEN it suppresses echo in this
        # room: in a misaligned physical path nothing is ever attenuated and
        # the override would fire on Jaeger's own voice.  Unproven rooms
        # fall back to the correlation-only rule.
        if aec.block_attenuation_db >= 6.0:
            self._filter_proven = True
        mature = aec.block_playback_age >= self.warmup_s + 1.0
        strong = (mature and self._filter_proven and
                  aec.block_attenuation_db < self.strong_attenuation_db and
                  clean_rms > max(self.speech_rms * 1.5,
                                  self._echo_floor * self.strong_margin))
        if (correlated or clean_rms <= self.speech_rms) and not strong:
            self._echo_floor = max(
                self.speech_rms * 0.25,
                0.92 * self._echo_floor + 0.08 * clean_rms)

        minimum = max(
            self.speech_rms * 1.1,
            self._echo_floor * self.residual_margin)
        external = clean_rms > minimum and not correlated
        self._candidate_samples = (
            self._candidate_samples + int(samples) if external else 0)
        # Leaky, not hard-reset: double-talk blocks are intermittently
        # attenuated when near and far pause out of phase, and one such
        # block must not restart the whole confirmation.
        self._strong_samples = (
            self._strong_samples + int(samples) if strong
            else self._strong_samples // 2)
        return (self._candidate_samples >= self.confirm_samples or
                self._strong_samples >= self.strong_confirm_samples)


def demo() -> None:
    """Corpus-style synthetic echo + double-talk check, not one clip."""
    import numpy as np

    rate, frame = 16000, 1600
    rng = np.random.default_rng(0)
    cancellations, near_ratios, barge_latencies = [], [], []
    false_barges = 0
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
        detector = EchoAwareBargeIn(rate, speech_rms=0.012)
        # Offline timestamps reproduce the same waveform/time mapping used by
        # the live callbacks without sleeping for eight seconds per case.
        aec.start_playback(far, rate, started_at=0.0)
        cleaned = []
        first_barge = None
        for i in range(0, len(far), frame):
            block = aec.process(
                mic[i:i + frame], captured_at=(i + frame) / rate)
            cleaned.append(block)
            clean_rms = float(np.sqrt(np.mean(np.square(block))))
            if first_barge is None and detector.update(
                    aec, clean_rms, len(block)):
                first_barge = (i + len(block)) / rate
        cleaned = np.concatenate(cleaned)
        echo_sl = slice(rate * 4, rate * 6)
        near_sl = slice(rate * 6, rate * 7)
        rms = lambda x, sl: float(np.sqrt(np.mean(x[sl] ** 2)))
        cancellations.append(20 * np.log10(
            (rms(mic, echo_sl) + 1e-9) /
            (rms(cleaned, echo_sl) + 1e-9)))
        near_ratios.append(rms(cleaned, near_sl) / (rms(near, near_sl) + 1e-9))
        if first_barge is None or first_barge >= 7.0:
            barge_latencies.append(float("inf"))
        elif first_barge < 6.0:
            false_barges += 1
        else:
            barge_latencies.append(first_barge - 6.0)
        aec.close()

    assert np.median(cancellations) > 12.0, cancellations
    assert np.median(near_ratios) > 0.70, near_ratios
    assert false_barges == 0, false_barges
    assert len(barge_latencies) == 8 and np.median(barge_latencies) <= 0.1, \
        barge_latencies

    # Experimental contiguous mode: one reply-level generation across
    # clauses, silence between them, and a safe do-nothing fallback once the
    # reference stream is exhausted.  The full six-case matrix lives in
    # jaeger_agent_omni/tests/test_aec_contiguous.py.
    aec = SpeexEchoCanceller(rate, reference_mode="contiguous")
    clause = rng.normal(0, 0.1, rate).astype(np.float32)
    aec.begin_reply_playback()
    aec.append_playback(clause, rate, started_at=0.0)
    aec.append_playback(clause, rate, started_at=1.5)     # 0.5 s real gap
    assert aec.playback_generation == 1, "clauses must share one generation"
    mic = np.zeros(int(rate * 2.5), dtype=np.float32)
    mic[:rate] = 0.4 * clause
    mic[int(1.5 * rate):] = 0.4 * clause
    for i in range(0, len(mic), frame):
        out = aec.process(mic[i:i + frame], captured_at=(i + frame) / rate)
        assert len(out) == frame
    assert aec.alignment_corrections == 0
    late = aec.process(np.zeros(frame, np.float32), captured_at=60.0)
    assert not aec.reference_active and len(late) == frame, \
        "exhausted reference must fall back to a pass-through"
    aec.close()

    print(f"runtime.aec: 8-case median echo attenuation "
          f"{np.median(cancellations):.1f} dB; double-talk survival "
          f"{100*np.median(near_ratios):.0f}%; reference-aware barge "
          f"{1000*np.median(barge_latencies):.0f}ms, "
          f"{false_barges} self-echo false triggers; contiguous reply "
          f"stream OK")


if __name__ == "__main__":
    demo()
