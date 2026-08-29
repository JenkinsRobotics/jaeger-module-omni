"""Node-local CampPlus speaker embedding and online identity runtime."""

from pathlib import Path

from jaeger_agent_omni.runtime.speaker import OnlineSpeakerTracker

ROOT = Path(__file__).resolve().parent


def build(*, k=3, new_speaker_cosine=0.65):
    return OnlineSpeakerTracker(
        k=k, new_speaker_cosine=new_speaker_cosine,
        model_path=ROOT / "weights/campplus.onnx")
