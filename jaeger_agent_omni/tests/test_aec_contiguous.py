"""Deterministic tests for the experimental contiguous AEC reference mode.

Everything runs offline with explicit ``started_at``/``captured_at``
timestamps, so no test sleeps and no test depends on real devices.  Wallclock
mode is the locked baseline and is deliberately not re-tested here beyond the
existing ``runtime.aec`` demo.

    python -m unittest jaeger_agent_omni.tests.test_aec_contiguous
"""

from __future__ import annotations

import unittest

import numpy as np

from jaeger_agent_omni.core.runtime.aec import SpeexEchoCanceller, available

RATE = 16_000
BLOCK = 1_600           # live callback cadence: 100 ms


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wav = rng.normal(0, 0.12, round(seconds * RATE)).astype(np.float32)
    return np.convolve(wav, np.ones(5) / 5, mode="same").astype(np.float32)


def _echo(reference: np.ndarray, total: int, delay: int,
          gain: float = 0.5) -> np.ndarray:
    mic = np.zeros(total, dtype=np.float32)
    end = min(total, delay + len(reference))
    if end > delay:
        mic[delay:end] = gain * reference[:end - delay]
    return mic


def _run(aec: SpeexEchoCanceller, mic: np.ndarray,
         clock_scale: float = 1.0) -> np.ndarray:
    cleaned = []
    for start in range(0, len(mic) - BLOCK + 1, BLOCK):
        captured_at = ((start + BLOCK) / RATE) * clock_scale
        cleaned.append(aec.process(mic[start:start + BLOCK],
                                   captured_at=captured_at))
    return np.concatenate(cleaned) if cleaned else np.zeros(0, np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


@unittest.skipUnless(available(), "SpeexDSP library not installed")
class ContiguousReferenceTests(unittest.TestCase):

    def test_continuous_far_end_consumption(self):
        """One appended reference is consumed sequentially and cancelled."""
        far = _speechlike(6.0, seed=1)
        aec = SpeexEchoCanceller(RATE, reference_mode="contiguous")
        aec.begin_reply_playback()
        aec.append_playback(far, RATE, started_at=0.0)
        mic = _echo(far, len(far), delay=800)
        cleaned = _run(aec, mic)
        tail = slice(4 * RATE, 6 * RATE)     # after convergence
        attenuation = 20 * np.log10(
            (_rms(mic[tail]) + 1e-9) / (_rms(cleaned[tail]) + 1e-9))
        self.assertGreater(attenuation, 12.0)
        self.assertEqual(aec.alignment_corrections, 0)
        self.assertEqual(aec.reference_underflow_samples, 0)
        self.assertEqual(aec.playback_generation, 1)
        aec.close()

    def test_clause_boundaries_share_one_generation(self):
        """Back-to-back clauses form one stream and one generation."""
        clause_a = _speechlike(2.0, seed=2)
        clause_b = _speechlike(2.0, seed=3)
        aec = SpeexEchoCanceller(RATE, reference_mode="contiguous")
        aec.begin_reply_playback()
        aec.append_playback(clause_a, RATE, started_at=0.0)
        aec.append_playback(clause_b, RATE, started_at=2.0)
        self.assertEqual(aec.playback_generation, 1)
        far = np.concatenate((clause_a, clause_b))
        mic = _echo(far, len(far), delay=800)
        cleaned = _run(aec, mic)
        # The boundary must not reset convergence: the second clause is
        # cancelled at least as well as the mid-first-clause warmup region.
        second = slice(3 * RATE, 4 * RATE)
        attenuation = 20 * np.log10(
            (_rms(mic[second]) + 1e-9) / (_rms(cleaned[second]) + 1e-9))
        self.assertGreater(attenuation, 12.0)
        self.assertEqual(aec.alignment_corrections, 0)
        aec.close()

    def test_silence_insertion_between_clauses(self):
        """A clause gap becomes literal zeros, not a reference reset."""
        clause = _speechlike(1.5, seed=4)
        aec = SpeexEchoCanceller(RATE, reference_mode="contiguous")
        aec.begin_reply_playback()
        aec.append_playback(clause, RATE, started_at=0.0)
        aec.append_playback(clause, RATE, started_at=2.5)   # 1.0 s gap
        self.assertEqual(aec.playback_generation, 1)
        gap = aec._stream[int(1.6 * RATE):int(2.4 * RATE)]
        self.assertEqual(float(np.max(np.abs(gap))), 0.0)
        total = int(4.0 * RATE)
        mic = _echo(clause, total, delay=800)
        mic += _echo(clause, total, delay=int(2.5 * RATE) + 800)
        cleaned = _run(aec, mic)
        second = slice(int(3.0 * RATE), int(3.9 * RATE))
        attenuation = 20 * np.log10(
            (_rms(mic[second]) + 1e-9) / (_rms(cleaned[second]) + 1e-9))
        self.assertGreater(attenuation, 12.0)
        aec.close()

    def test_fixed_output_latency_absorbed_by_filter(self):
        """A constant device delay needs no alignment corrections."""
        far = _speechlike(6.0, seed=5)
        aec = SpeexEchoCanceller(RATE, reference_mode="contiguous")
        aec.begin_reply_playback()
        aec.append_playback(far, RATE, started_at=0.0)
        delay = int(0.24 * RATE)             # 240 ms < 500 ms filter span
        mic = _echo(far, len(far), delay=delay)
        cleaned = _run(aec, mic)
        tail = slice(4 * RATE, 6 * RATE)
        attenuation = 20 * np.log10(
            (_rms(mic[tail]) + 1e-9) / (_rms(cleaned[tail]) + 1e-9))
        self.assertGreater(attenuation, 10.0)
        self.assertEqual(aec.alignment_corrections, 0)
        self.assertLess(abs(aec.estimated_drift_ppm), 100.0)
        aec.close()

    def test_simulated_clock_drift_is_measured(self):
        """±1% ADC clock skew is detected, counted, and never crashes."""
        for scale, sign in ((1.01, 1), (0.99, -1)):
            far = _speechlike(10.0, seed=6)
            aec = SpeexEchoCanceller(RATE, reference_mode="contiguous")
            aec.begin_reply_playback()
            aec.append_playback(far, RATE, started_at=0.0)
            mic = _echo(far, len(far), delay=800)
            cleaned = _run(aec, mic, clock_scale=scale)
            self.assertEqual(len(cleaned), (len(mic) // BLOCK) * BLOCK)
            self.assertGreaterEqual(aec.alignment_corrections, 1,
                                    f"scale {scale} produced no corrections")
            self.assertGreater(sign * aec.estimated_drift_ppm, 5_000.0,
                               f"scale {scale}: {aec.estimated_drift_ppm}")
            aec.close()

    def test_reference_underflow_falls_back_safely(self):
        """Late appends are counted; an exhausted stream passes audio through."""
        clause = _speechlike(1.0, seed=7)
        aec = SpeexEchoCanceller(RATE, reference_mode="contiguous")
        aec.begin_reply_playback()
        aec.append_playback(clause, RATE, started_at=0.0)
        _run(aec, _echo(clause, 2 * RATE, delay=800))   # consume 2.0 s
        consumed = aec._read_pos
        # A clause claiming to have STARTED at 1.5 s arrives only now — the
        # first 0.5 s of it was already consumed as zeros.
        aec.append_playback(clause, RATE, started_at=1.5)
        self.assertEqual(aec.reference_underflow_samples,
                         consumed - int(1.5 * RATE))
        self.assertEqual(aec.playback_generation, 1)
        # Beyond the stream and its filter tail: exact pass-through, no state.
        probe = _speechlike(0.1, seed=8)[:BLOCK]
        out = aec.process(probe, captured_at=120.0)
        self.assertFalse(aec.reference_active)
        np.testing.assert_array_equal(out, probe)
        aec.close()


if __name__ == "__main__":
    unittest.main()
