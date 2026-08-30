"""Load MiniCPM-o's ENTIRE trained audio front end into our modules.

"Copy their baseline, then delta." Their encoder is NOT stock whisper-medium —
measured 17-31% relative weight changes from OpenBMB's alignment training — so
the trained pairing is (their encoder + their projector), and both get lifted:

    their whisper encoder (fine-tuned)   FROZEN   371 tensors from the GGUF
    their audio projector                FROZEN   21M params
    DeltaAdapter -> our backbone         TRAINED  the only new weights

    python research/load_minicpm.py <corpus>    # diversity check, full stack
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_gguf import read_tensors                               # noqa: E402
from load_minicpm_projector import load as load_projector           # noqa: E402

AUDIO_GGUF = Path.home() / "models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf"

# GGUF name -> HF WhisperEncoder name (inverse of their convert_apm.py map).
_FIXED = {
    "encoder.conv1.weight": "conv1.weight", "encoder.conv1.bias": "conv1.bias",
    "encoder.conv2.weight": "conv2.weight", "encoder.conv2.bias": "conv2.bias",
    "encoder.positional_embedding": "embed_positions.weight",
    "encoder.ln_post.weight": "layer_norm.weight",
    "encoder.ln_post.bias": "layer_norm.bias",
}
_BLOCK = {
    "attn.query": "self_attn.q_proj", "attn.key": "self_attn.k_proj",
    "attn.value": "self_attn.v_proj", "attn.out": "self_attn.out_proj",
    "attn_ln": "self_attn_layer_norm",
    "mlp.0": "fc1", "mlp.2": "fc2", "mlp_ln": "final_layer_norm",
}


def _hf_name(gguf_name: str) -> str | None:
    if gguf_name in _FIXED:
        return _FIXED[gguf_name]
    if gguf_name.startswith("encoder.blocks."):
        rest = gguf_name[len("encoder.blocks."):]
        n, sub = rest.split(".", 1)
        for theirs, ours in _BLOCK.items():
            if sub.startswith(theirs + "."):
                return f"layers.{n}.{ours}.{sub[len(theirs) + 1:]}"
    return None    # audio_projector.*, decoder.* — not the encoder's


def load_encoder(gguf_path=AUDIO_GGUF):
    """Their fine-tuned whisper-medium encoder in a stock HF module."""
    from transformers import WhisperModel

    encoder = WhisperModel.from_pretrained("openai/whisper-medium").encoder
    tensors = read_tensors(gguf_path)
    state, skipped = {}, []
    for gname, arr in tensors.items():
        hname = _hf_name(gname)
        if hname is None:
            skipped.append(gname)
            continue
        t = torch.from_numpy(arr.copy()).squeeze()
        # conv biases were stored [1, C]; HF wants [C]. Weights keep shape.
        state[hname] = t

    ref = encoder.state_dict()
    for k, v in state.items():
        if k not in ref:
            raise KeyError(f"mapped to unknown HF key {k}")
        state[k] = v.reshape(ref[k].shape)
    missing = [k for k in ref if k not in state]
    if missing:
        raise ValueError(f"encoder tensors missing from GGUF: {missing[:5]}")
    encoder.load_state_dict(state)
    return encoder.eval().requires_grad_(False)


def load_front_end(gguf_path=AUDIO_GGUF):
    """(their encoder, their projector) — the frozen pairing, verified below."""
    return load_encoder(gguf_path), load_projector(gguf_path)


def load_remapped_front_end(map_path: str | Path | None = None,
                            adapter_path: str | Path | None = None,
                            gguf_path=AUDIO_GGUF):
    """Load tuned Whisper plus ONE MiniCPM-shaped Gemma projector.

    ``adapter_path`` is a trained ``.npz`` containing a 2560x4096 weight and
    2560 bias. Without it, ``map_path`` must contain the offline Qwen->Gemma
    4096x2560 map used to compose MiniCPM's original final affine. There is no
    separate post-projector delta in either case.
    """
    import numpy as np

    from jaeger_agent_omni.core.neural import MiniCPMRemappedProjector

    encoder = load_encoder(gguf_path)
    original = load_projector(gguf_path)
    projector = MiniCPMRemappedProjector()
    with torch.no_grad():
        projector.linear1.weight.copy_(original.linear1.weight)
        projector.linear1.bias.copy_(original.linear1.bias)
        if adapter_path is not None:
            saved = np.load(adapter_path)
            weight = torch.from_numpy(saved["weight"].astype(np.float32))
            bias = torch.from_numpy(saved["bias"].astype(np.float32))
        else:
            if map_path is None:
                raise ValueError("need adapter_path or map_path")
            mapping = torch.from_numpy(np.load(map_path).astype(np.float32))
            if mapping.shape != (4096, 2560):
                raise ValueError(f"Qwen->Gemma map {tuple(mapping.shape)} != "
                                 "(4096, 2560)")
            weight = mapping.T @ original.linear2.weight
            bias = mapping.T @ original.linear2.bias
        if weight.shape != projector.linear2.weight.shape:
            raise ValueError(f"adapter weight {tuple(weight.shape)} != "
                             f"{tuple(projector.linear2.weight.shape)}")
        if bias.shape != projector.linear2.bias.shape:
            raise ValueError(f"adapter bias {tuple(bias.shape)} != "
                             f"{tuple(projector.linear2.bias.shape)}")
        projector.linear2.weight.copy_(weight)
        projector.linear2.bias.copy_(bias)
    projector.freeze_minicpm_base()
    return encoder, projector


def main() -> int:
    import torch.nn.functional as F

    from jaeger_agent_omni.core.capture import SAMPLE_RATE, CaptureReader
    from transformers import WhisperProcessor

    encoder, projector = load_front_end()
    print(f"their encoder loaded: {sum(p.numel() for p in encoder.parameters())/1e6:.0f}M params")
    print(f"their projector loaded: {projector.n_params()/1e6:.1f}M params")

    corpus = sys.argv[1] if len(sys.argv) > 1 else None
    if not corpus:
        return 0
    proc = WhisperProcessor.from_pretrained("openai/whisper-medium")
    reader = CaptureReader(corpus)
    pooled = []
    for t in list(reader)[:8]:
        f = proc(reader.audio(t), sampling_rate=SAMPLE_RATE,
                 return_tensors="pt").input_features
        with torch.no_grad():
            h = encoder(f).last_hidden_state.float()
            out = projector(h[:, : max(1, int(t.duration_s * 50))])
        pooled.append(F.normalize(out[0].mean(0), dim=-1))
    V = torch.stack(pooled)
    off = (V @ V.T)[~torch.eye(len(V), dtype=bool)]
    print(f"\nTHEIR encoder + THEIR projector, our code:")
    print(f"  cross-clip cosine {off.mean():.4f}   "
          f"(stock encoder + their projector was 0.9052)")
    return 0


def demo():
    # Name mapping must cover every encoder tensor family and nothing else.
    assert _hf_name("encoder.conv1.weight") == "conv1.weight"
    assert _hf_name("encoder.blocks.3.attn.query.weight") == \
        "layers.3.self_attn.q_proj.weight"
    assert _hf_name("encoder.blocks.23.mlp.2.bias") == "layers.23.fc2.bias"
    assert _hf_name("encoder.blocks.0.attn_ln.weight") == \
        "layers.0.self_attn_layer_norm.weight"
    assert _hf_name("encoder.positional_embedding") == "embed_positions.weight"
    assert _hf_name("audio_projector.linear1.weight") is None
    assert _hf_name("decoder.token_embedding.weight") is None
    print("load_minicpm: name mapping OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
