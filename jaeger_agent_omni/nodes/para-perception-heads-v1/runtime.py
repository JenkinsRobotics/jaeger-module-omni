"""TRAINED perception heads node: the five per-frame heads riding the
streaming adapter's trunk (speaker_change, overlap, turn_end, speaker_id,
directed) — the NN that replaces hand-coded turn/addressing perception.
Wraps an ALREADY-BUILT transcriber (same checkpoint, same front end): one
extra front_end+adapter pass per analyzed utterance, no second model."""
from __future__ import annotations

import numpy as np


def summarize_heads(heads: dict) -> dict:
    """Pure summary of per-frame logits -> turn-level probabilities.
    Selftested; the trained meaning of each number comes from the forge's
    training data, the MATH here is just calibrated pooling."""
    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
    out = {}
    if "directed" in heads and len(heads["directed"]):
        out["directed_p"] = round(float(np.mean(sig(heads["directed"]))), 3)
    if "turn_end" in heads and len(heads["turn_end"]):
        tail = sig(heads["turn_end"])[-max(1, len(heads["turn_end"]) // 4):]
        out["turn_end_p"] = round(float(np.mean(tail)), 3)
    if "overlap" in heads and len(heads["overlap"]):
        out["overlap_p"] = round(float(np.max(sig(heads["overlap"]))), 3)
    if "speaker_change" in heads and len(heads["speaker_change"]):
        p = sig(heads["speaker_change"])
        out["speaker_change_frames"] = [int(i) for i in np.where(p > 0.5)[0]]
    return out


class PerceptionHeads:
    def __init__(self, transcriber) -> None:
        if not hasattr(transcriber.adapter, "perceive"):
            raise RuntimeError(
                "adapter checkpoint carries no perception heads — "
                "streaming_causal architecture required")
        self._t = transcriber

    def analyze(self, audio: np.ndarray, self_speech=None) -> dict:
        import mlx.core as mx
        cached = self._t.front_end(
            np.asarray(audio, np.float32).reshape(-1))
        _emb, heads = self._t.adapter.perceive(mx.array(cached), self_speech)
        return summarize_heads(
            {k: np.asarray(v) for k, v in heads.items()})


def build(transcriber=None, cfg=None):
    if transcriber is None:
        raise RuntimeError(
            "pass the built semantic transcriber — the heads ride its trunk")
    return PerceptionHeads(transcriber)
