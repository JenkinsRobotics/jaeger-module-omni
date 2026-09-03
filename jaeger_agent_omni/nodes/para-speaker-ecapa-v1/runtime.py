"""TRAINED speaker + wake-embedding node: ECAPA-TDNN (speechbrain,
VoxCeleb) — the NN that replaces the hand-coded fingerprint for identity
and gives the QbE wake a LEARNED embedding space. Wraps the forge core's
SpeakerEmbedder/SpeakerRegistry (calibrated on real two-speaker audio:
within-cluster 0.51, between 0.17, threshold 0.30)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parent.parent / "core"


def _core():
    if "mlx_audio_streaming" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "mlx_audio_streaming", CORE / "mlx_audio_streaming.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["mlx_audio_streaming"] = m
        spec.loader.exec_module(m)
    return sys.modules["mlx_audio_streaming"]


class SpeakerEcapa:
    """Turn-level identity + one-shot wake matching in ECAPA space."""

    WAKE_SIM = 0.60      # cosine floor for a wake-template hit

    def __init__(self) -> None:
        self._emb = None
        self._reg = None
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def load(self) -> None:
        core = _core()
        self._emb = core.SpeakerEmbedder()
        self._reg = core.SpeakerRegistry()

    def _tfile(self) -> Path:
        return HERE / "weights" / "wake_templates_ecapa.npz"

    def _load_templates(self) -> None:
        if self._tfile().exists():
            z = np.load(self._tfile())
            self.templates = {k: z[k] for k in z.files}

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        pcm = np.asarray(audio, np.float32).reshape(-1)
        t = self._emb.torch.tensor(pcm[None, :])
        v = self._emb.encoder.encode_batch(t).squeeze().detach().numpy()
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def label_turn(self, audio: np.ndarray) -> dict:
        sid, purity = self._emb.turn_speaker(
            np.asarray(audio, np.float32).reshape(-1), self._reg)
        return {"speaker": int(sid), "conf": round(float(purity), 3)}

    def enroll(self, name: str, audio: np.ndarray) -> None:
        """ONE example -> one template, in the LEARNED space."""
        v = self.embed(audio)
        if v is None:
            raise ValueError("no voiced audio to enroll")
        self.templates[name] = v
        np.savez(self._tfile(), **self.templates)

    def wake_match(self, audio: np.ndarray):
        if not self.templates:
            return None
        v = self.embed(audio)
        if v is None:
            return None
        best, sim = None, -1.0
        for name, tpl in self.templates.items():
            s = float(np.dot(v, tpl))
            if s > sim:
                best, sim = name, s
        return (best, round(sim, 3)) if sim >= self.WAKE_SIM else None


def build(cfg=None):
    return SpeakerEcapa()
