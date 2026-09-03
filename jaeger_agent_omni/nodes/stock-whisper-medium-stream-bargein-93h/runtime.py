"""Entry point for the streaming, self-voice-aware GemmaOmni node.

This node pairs the stock whisper-medium encoder with the causal context
adapter that carries the three perception heads. Encoder and adapter ship as
ONE node on purpose: the adapter is trained against this encoder's exact
output distribution, so swapping the encoder means building a different node,
not editing this one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parent
BACKBONE = NODE_ROOT.parent.parent / "core"   # gemmaomni core modules


def manifest() -> dict:
    return json.loads((NODE_ROOT / "node.json").read_text())


def build(*, model_path=None, system=None, device="auto"):
    """Construct the live transcriber for this immutable pairing."""
    sys.path.insert(0, str(BACKBONE))
    import gemmaomni_adapter as ca

    spec = manifest()["encoder"]
    local = NODE_ROOT / spec["path"]
    kwargs = {}
    if system is not None:
        # only forward a system prompt when one is asked for: the transcriber
        # signature does not accept it, and passing None would still fail
        kwargs["system"] = system
    return ca.CleanroomTranscriber(
        adapter_path=NODE_ROOT / manifest()["adapter"]["path"],
        model_path=model_path,
        encoder={"source": str(local) if local.is_dir() else spec["id"],
                 "output_dim": spec["output_dim"]},
        **kwargs,
    )


def load_adapter_only():
    """Adapter + heads without Gemma — for evaluation and head training."""
    sys.path.insert(0, str(BACKBONE))
    import numpy as np
    from streaming_adapter import build_streaming_adapter

    path = NODE_ROOT / manifest()["adapter"]["path"]
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return build_streaming_adapter(int(arrays["hidden"][0]),
                                   arrays=arrays, causal=True)
