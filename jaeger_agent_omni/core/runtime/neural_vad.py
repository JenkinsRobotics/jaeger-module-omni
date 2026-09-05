"""Load the repository-owned neural speech gate for live turn detection."""

from __future__ import annotations

from pathlib import Path


# runtime/neural_vad.py lives three directories below the repository root.
# Keep the packaged gate outside the import package: it is an independently
# verified runtime asset, not Python package data.
REPO = Path(__file__).resolve().parents[3]
GATE_ROOT = REPO / "gates" / "silero-vad"


class SpeechPresence:
    """Independent raw-waveform speech evidence while the main loop generates.

    This detects speech, not overlap or addressed intent. Each instance owns
    its probability engine; the endpoint's recurrent state is never shared
    across the microphone callback and the inference loop.
    """

    def __init__(self, probability, *, frame_samples=480, threshold=0.30,
                 confirm_frames=3):
        import numpy as np
        self.probability = probability
        self.frame_samples = int(frame_samples)
        self.threshold = float(threshold)
        self.confirm_frames = int(confirm_frames)
        self.pending = np.empty(0, dtype=np.float32)
        self.consecutive = 0
        self.frames = 0
        self.last_probability = 0.0
        self.update_ms = []

    def update(self, audio):
        import time
        import numpy as np
        started = time.perf_counter()
        self.pending = np.concatenate((self.pending, np.asarray(audio, dtype=np.float32)))
        while len(self.pending) >= self.frame_samples:
            frame, self.pending = self.pending[:self.frame_samples], self.pending[self.frame_samples:]
            self.last_probability = float(self.probability(frame))
            self.consecutive = self.consecutive + 1 if self.last_probability >= self.threshold else 0
            self.frames += 1
        self.update_ms.append((time.perf_counter() - started) * 1000)
        return self.consecutive >= self.confirm_frames

    def summary(self):
        import numpy as np
        return {"node": "silero-vad", "input": "raw_pcm", "frames": self.frames,
                "threshold": self.threshold, "confirm_frames": self.confirm_frames,
                "last_probability": self.last_probability,
                "update_ms_p95": float(np.percentile(self.update_ms, 95)) if self.update_ms else None,
                "update_ms_max": max(self.update_ms, default=0.0)}


def load_silero():
    """Verify and construct the immutable local Silero ONNX engine.

    Loading is deliberately explicit and fail-closed.  ``--vad neural`` must
    never silently become an RMS gate because that would make a physical run's
    configuration and results disagree.
    """
    from jaeger_agent_omni.node_registry import (
        SUPPORTED_GATE_KINDS, load_factory, load_package, verify_weights,
    )

    package = load_package(
        GATE_ROOT, manifest_name="gate.json",
        supported_kinds=SUPPORTED_GATE_KINDS)
    verify_weights(package)
    engine = load_factory(package)()
    return engine, package


def configure_endpoint(endpoint, *, mode: str, energy_threshold: float,
                       open_threshold: float = 0.30,
                       close_threshold: float = 0.125) -> str:
    """Attach neural VAD or the explicit legacy energy fallback."""
    endpoint.speech_rms = float(energy_threshold)
    if mode == "energy":
        endpoint.frame_ms = 20.0
        endpoint.speech_probability = None
        endpoint.__post_init__()
        return "energy RMS"
    if mode != "neural":
        raise ValueError(f"unknown VAD mode: {mode}")
    engine, package = load_silero()
    endpoint.frame_ms = (
        int(package.manifest["frame_samples"]) * 1000.0 /
        int(package.manifest["sample_rate"]))
    endpoint.speech_probability = engine.probability
    endpoint.speech_open_threshold = float(open_threshold)
    endpoint.speech_close_threshold = float(close_threshold)
    endpoint.__post_init__()
    return f"{package.node_id} ONNX"


def demo() -> None:
    import numpy as np
    from .endpoint import Endpoint

    endpoint = Endpoint()
    label = configure_endpoint(
        endpoint, mode="neural", energy_threshold=0.012)
    frame = np.zeros(480, dtype=np.float32)
    probability = endpoint.speech_probability(frame)
    assert endpoint.frame_ms == 30.0
    assert endpoint.speech_mode == "neural"
    assert 0.0 <= probability <= 1.0
    assert "silero-vad" in label
    print(f"neural_vad: verified local gate · silence p={probability:.4f}")


if __name__ == "__main__":
    demo()
