"""Offline, node-local Kokoro synthesis runtime."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Engine:
    def __init__(self, lang="a", device=None):
        from kokoro import KModel, KPipeline

        model_root = ROOT / "weights/model"
        model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(model_root / "config.json"),
            model=str(model_root / "kokoro-v1_0.pth"))
        if device:
            model = model.to(device)
        model.eval()
        self.pipeline = KPipeline(
            lang_code=lang, repo_id="hexgrad/Kokoro-82M",
            model=model, device=device)
        self.model_root = model_root

    def synthesize(self, text, voice="af_heart", speed=1.0):
        import numpy as np

        voice_path = self.model_root / "voices" / f"{voice}.pt"
        if not voice_path.is_file():
            raise ValueError(f"voice is not packaged in this node: {voice}")
        chunks = [np.asarray(row.audio, dtype=np.float32).reshape(-1)
                  for row in self.pipeline(
                      text, voice=str(voice_path), speed=float(speed))]
        return np.concatenate(chunks) if chunks else np.zeros(1, np.float32)


def build(*, lang="a", device=None):
    return Engine(lang=lang, device=device)
