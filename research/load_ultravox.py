"""Lift Ultravox's trained projector into our own modules.

Their published model will not load on transformers v5 — the v4-era loader
dies in `tie_weights` on meta tensors, after their config unconditionally
fetches Meta's GATED Llama repo. Four blockers cleared (gated id, peft,
relative imports, config bypass) before hitting a genuine version wall.

None of that stops us reading the weights, which is what we actually wanted:
a second expert's trained answer to "encoder output -> LLM embedding space",
to compare against MiniCPM's.

Measured from `model.safetensors` (v0.5-llama-3.2-1b):

    ln_pre    (10240,)        RMSNorm over STACKED input: 1280 x 8 frames
    linear_1  (4096, 10240)   4096 = 2048 x 2, SwiGLU halves it
    ln_mid    (2048,)         RMSNorm mid-stack (projector_ln_mid=True)
    linear_2  (2048, 2048)    -> Llama 3.2 1B width
                              ~48M params, no biases anywhere

They also ship `audio_tower` weights — a FINE-TUNED Whisper, exactly as
MiniCPM does. Two independent teams, neither using stock Whisper.

Where they disagree with MiniCPM, which is the whole reason to look:

    compression   stack 8 frames (keep everything)  vs  mean-pool 5 (average)
    activation    SwiGLU (gated)                    vs  ReLU
    normalisation RMSNorm pre AND mid               vs  none at all
    bias          none                              vs  yes

MiniCPM's no-normalisation choice is the one our own from-scratch run
punished: it collapsed to a constant (cross-clip cosine 0.990) until we pinned
the output scale. Ultravox solved the same instability structurally.

    python research/load_ultravox.py            # inspect + load into ours
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

# Their stack factor and dims, read from the shipped tensors rather than docs.
STACK_FACTOR = 8
WHISPER_DIM = 1280          # whisper-large family: 10240 / 8
LLAMA_1B_DIM = 2048


def weights(repo: str = REPO) -> dict:
    """The trained projector tensors, keyed without the module prefix."""
    import safetensors.torch as st
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(repo))
    sd = st.load_file(d / "model.safetensors")
    return {k.replace("multi_modal_projector.", ""): v
            for k, v in sd.items() if "multi_modal_projector" in k}


def into_ab_projector(cognition_dim: int | None = None):
    """Their weights inside our `ab_projectors.ABProjector('ultravox')`.

    Only possible when the target width matches theirs (2048); otherwise the
    second linear is a different shape and the variant must be trained. That
    is the honest constraint — an adapter is a bridge between two specific
    models, and theirs bridges to Llama 3.2 1B.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    from ab_projectors import VARIANTS, ABProjector

    w = weights()
    out_dim = w["linear_2.weight"].shape[0]
    if cognition_dim not in (None, out_dim):
        raise ValueError(
            f"their projector emits {out_dim} (Llama 3.2 1B); it cannot be "
            f"loaded into a {cognition_dim}-wide backbone without retraining")

    p = ABProjector(VARIANTS["ultravox"], input_dim=WHISPER_DIM,
                    cognition_dim=out_dim)
    p.ln_pre.weight.data.copy_(w["ln_pre.weight"])
    p.linear_1.weight.data.copy_(w["linear_1.weight"])
    p.ln_mid.weight.data.copy_(w["ln_mid.weight"])
    p.linear_2.weight.data.copy_(w["linear_2.weight"])
    return p


def main() -> int:
    import torch
    import torch.nn.functional as F

    w = weights()
    print(f"ultravox projector — {sum(v.numel() for v in w.values())/1e6:.1f}M params")
    for k, v in sorted(w.items()):
        print(f"  {k:22s} {str(tuple(v.shape)):>16s}  std {v.float().std():.4f}")

    p = into_ab_projector()
    print(f"\nloaded into our ABProjector: {WHISPER_DIM} -> {p.cognition_dim}, "
          f"{p.tokens_per_second:.2f} tok/s (stack {STACK_FACTOR})")

    # Does their trained bridge preserve structure? Same probe we ran on
    # MiniCPM's — 8 sources, per-take variation, measured after projection.
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(8, WHISPER_DIM), dim=-1)
    X = torch.cat([(anchors[i % 8].repeat(240, 1)
                    + 0.35 * torch.randn(240, WHISPER_DIM)).unsqueeze(0)
                   for i in range(16)])
    with torch.no_grad():
        out = F.normalize(p(X).mean(1), dim=-1)
    S = out @ out.T
    labels = torch.tensor([i % 8 for i in range(16)])
    same = torch.tensor([[labels[i] == labels[j] for j in range(16)]
                         for i in range(16)])
    eye = torch.eye(16, dtype=bool)
    print(f"  cross-clip cosine  {S[~eye].mean():.4f}")
    print(f"  class separation   {S[same & ~eye].mean() - S[~same].mean():.4f}")
    print("  (MiniCPM's trained projector measured 0.9052 cross-clip; theirs "
          "normalises hard, so the scales are not directly comparable)")
    return 0


def demo():
    # Shape facts asserted from the published checkpoint, so a silent change
    # upstream is caught rather than absorbed.
    try:
        w = weights()
    except Exception as exc:                           # noqa: BLE001
        print(f"load_ultravox: weights unavailable ({type(exc).__name__}), "
              "shape contract unchecked")
        return
    assert w["linear_1.weight"].shape == (4096, 10240), w["linear_1.weight"].shape
    assert w["linear_2.weight"].shape == (2048, 2048)
    assert w["ln_pre.weight"].shape == (10240,), "RMSNorm sits on STACKED input"
    assert w["ln_mid.weight"].shape == (2048,), "projector_ln_mid=True"
    # SwiGLU halves: linear_1 emits 2x what linear_2 consumes.
    assert w["linear_1.weight"].shape[0] == 2 * w["linear_2.weight"].shape[1]
    # Stacking: input width is whisper-dim x stack_factor.
    assert w["linear_1.weight"].shape[1] == WHISPER_DIM * STACK_FACTOR
    assert not any("bias" in k for k in w), "Ultravox uses no biases"
    print(f"load_ultravox: {sum(v.numel() for v in w.values())/1e6:.1f}M params, "
          "stack/SwiGLU/RMSNorm shape contract OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
