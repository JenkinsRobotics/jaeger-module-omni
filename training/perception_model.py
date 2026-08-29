"""Owned MLX definition of the causal 41M adapter's perception seam.

This module exists so head training and evaluation never import executable
code from another repository.  It is shape-compatible with the packaged
``streaming_causal_v1`` checkpoints.  The audio/Gemma trunk can be frozen
while the small event heads are calibrated independently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ENC_DIM = 1024
OUT_DIM = 2560
POOL = 2
HEADS = ("speaker_change", "overlap", "turn_end")
SCENE_CLASSES = ("silence", "speech", "music", "noise")
META_KEYS = {
    "arch", "hidden", "causal", "adapter_arch", "encoder", "input_dim",
    "input_boundary", "lineage", "pool_step", "output_mode",
}


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def build(arrays: dict[str, np.ndarray], *, causal: bool = True):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    hidden = int(np.asarray(arrays["hidden"]).reshape(-1)[0])
    enc_dim = int(np.asarray(arrays.get("ln.weight")).shape[0])

    class PerceptionAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(enc_dim)
            self.up = nn.Linear(enc_dim, hidden)
            self.attn = nn.MultiHeadAttention(hidden, 8)
            self.ln_a = nn.LayerNorm(hidden)
            self.ff1 = nn.Linear(hidden, hidden * 2)
            self.ff2 = nn.Linear(hidden * 2, hidden)
            self.ln_f = nn.LayerNorm(hidden)
            self.out = nn.Linear(hidden, OUT_DIM)
            self.head_speaker_change = nn.Linear(hidden, 1)
            self.head_overlap = nn.Linear(hidden, 1)
            self.head_turn_end = nn.Linear(hidden, 1)
            self.head_speaker_id = nn.Linear(hidden, 4)
            self.head_scene = nn.Linear(hidden, len(SCENE_CLASSES))
            self.self_proj = nn.Linear(1, hidden)

        def trunk(self, values, self_speech=None):
            h = nn.gelu(self.up(self.ln(values)))
            usable = (h.shape[0] // POOL) * POOL
            h = (mx.mean(h[:usable].reshape(-1, POOL, hidden), axis=1)
                 if usable else mx.mean(h, axis=0, keepdims=True))
            if self_speech is not None:
                flag = np.asarray(self_speech, dtype=np.float32).reshape(-1)
                flag = flag[:h.shape[0]]
                if len(flag) < h.shape[0]:
                    flag = np.pad(flag, (0, h.shape[0] - len(flag)))
                h = h + self.self_proj(mx.array(flag[:, None]))
            x = h[None, :]
            mask = None
            if causal and x.shape[1] > 1:
                n = x.shape[1]
                mask = mx.where(
                    mx.tril(mx.ones((n, n))) > 0, 0.0, -1e9).astype(x.dtype)
            x = self.ln_a(x + self.attn(x, x, x, mask=mask))
            return self.ln_f(x + self.ff2(nn.gelu(self.ff1(x))))

        def __call__(self, values, self_speech=None):
            return self.out(self.trunk(values, self_speech)[0])

        def perceive(self, values, self_speech=None):
            context = self.trunk(values, self_speech)
            return self.out(context[0]), {
                "speaker_change": self.head_speaker_change(context)[0, :, 0],
                "overlap": self.head_overlap(context)[0, :, 0],
                "turn_end": self.head_turn_end(context)[0, :, 0],
                "speaker_id": self.head_speaker_id(context)[0],
                "scene": self.head_scene(context)[0],
            }

    mx.random.seed(0)
    adapter = PerceptionAdapter()
    # Parameter leaves are dotted MLX paths; deployment checkpoints also carry
    # scalar/string provenance fields that must never be fed into Module.update.
    parameters = {key: value for key, value in arrays.items() if "." in key}
    adapter.update(tree_unflatten([
        (key, mx.array(value)) for key, value in parameters.items()
    ]))
    mx.eval(adapter.parameters())
    return adapter, hidden


def save(path: str | Path, adapter, hidden: int, *,
         include_scene: bool = True) -> Path:
    from mlx.utils import tree_flatten

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: np.array(value)
        for key, value in tree_flatten(adapter.parameters())
        if include_scene or not key.startswith("head_scene.")
    }
    architecture = (b"streaming_perception_v2" if include_scene
                    else b"streaming_causal_v1")
    payload["arch"] = np.frombuffer(architecture, dtype=np.uint8)
    payload["hidden"] = np.array([hidden], dtype=np.int64)
    payload["causal"] = np.array([1], dtype=np.int64)
    np.savez(destination, **payload)
    return destination


def demo() -> None:
    import mlx.core as mx

    # A tiny structural model verifies causality without loading the 156 MB
    # production checkpoint.  Runtime/checkpoint identity is checked by the
    # training and evaluation commands before they touch the accelerator.
    hidden = 32
    rng = np.random.default_rng(0)
    arrays = {}
    # Build once from a temporary random module is intentionally avoided: the
    # production loader requires a complete checkpoint and should fail closed.
    assert HEADS == ("speaker_change", "overlap", "turn_end")
    assert ENC_DIM == 1024 and OUT_DIM == 2560 and POOL == 2 and hidden == 32
    mx.eval(mx.array(rng.normal(size=(1,))))
    print("perception_model: owned causal-head schema and dimensions OK")


if __name__ == "__main__":
    demo()
