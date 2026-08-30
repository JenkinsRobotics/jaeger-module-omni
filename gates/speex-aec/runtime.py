"""Gate-local entrypoint for the packaged SpeexDSP library."""

from pathlib import Path

from jaeger_agent_omni.core.runtime.aec import SpeexEchoCanceller

ROOT = Path(__file__).resolve().parent


def build(*, sample_rate=16000, frame_ms=20, filter_ms=500):
    return SpeexEchoCanceller(
        sample_rate=sample_rate, frame_ms=frame_ms, filter_ms=filter_ms,
        library_path=ROOT / "weights/libspeexdsp.dylib")
