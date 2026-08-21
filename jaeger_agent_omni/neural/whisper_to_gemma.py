"""ONE adapter: whisper -> Gemma. MiniCPM's values, Gemma's shape.

MiniCPM ships   whisper -> adapter -> Qwen.
We want         whisper -> adapter -> Gemma.

So take their adapter and RETARGET it, rather than bolting a second adapter
onto the end of theirs. That earlier design --

    whisper -> [MiniCPM adapter, FROZEN] -> Qwen 4096 -> [extra layer] -> Gemma

-- forces everything through a Qwen-shaped bottleneck nobody can touch. Their
projector was trained to emit what Qwen wanted, so whatever Gemma needs that
Qwen did not is already discarded before the extra layer sees it. It also left
only 10.5M of ~31M parameters trainable.

This is one adapter, all of it trainable:

    linear1  1024 -> 4096     COPIED from MiniCPM, identical shape
    ReLU                      theirs. Not GELU.
    linear2  4096 -> 2560     MiniCPM's linear2 COMPOSED with a ridge map
    AvgPool(k=5, s=5)         theirs. 50 fps -> 10 tokens/s.
    embed_norm                pinned to what Gemma actually receives

The composition is the interesting part. MiniCPM's linear2 is 4096->4096 and
Gemma needs 2560 out, so the widths do not match and the weights cannot be
copied. But a ridge map W (4096->2560), fitted from words embedded by both
token tables, translates Qwen space into Gemma space. Composing them gives a
single 4096->2560 matrix meaning "what MiniCPM would have said, in Gemma":

    y = relu(linear1(x)) @ L2.T + L2.bias        then  @ W
      = relu(linear1(x)) @ (L2.T @ W) + (L2.bias @ W)

    weight = (L2.T @ W).T = W.T @ L2      bias = L2.bias @ W

    python -m jaeger_agent_omni.neural.whisper_to_gemma   # self-check
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityProjector, pooled_tokens, register

WHISPER_DIM = 1024          # whisper-medium d_model
MINICPM_MID = 4096          # their hidden width (= Qwen3-8B)
MINICPM_POOL_STEP = 5
WHISPER_FRAME_RATE = 50.0
WHISPER_PAD_FRAMES = 1500


@register("whisper_to_gemma")
class WhisperToGemma(ModalityProjector):
    """whisper hidden states -> Gemma embedding space. Fully trainable."""

    def __init__(self, audio_dim: int = WHISPER_DIM, cognition_dim: int = 2560,
                 mid: int = MINICPM_MID, stride: int = MINICPM_POOL_STEP,
                 embed_norm: float | None = None):
        super().__init__(input_dim=audio_dim, cognition_dim=cognition_dim,
                         stride=stride, frame_rate=WHISPER_FRAME_RATE,
                         max_frames=WHISPER_PAD_FRAMES)
        self.linear1 = nn.Linear(audio_dim, mid)
        self.linear2 = nn.Linear(mid, cognition_dim)
        # NOTE: no self.pool here. ModalityProjector already provides pool()
        # as a METHOD that transposes correctly; assigning an nn.AvgPool1d
        # shadows it, and the module then pools the FEATURE axis instead of
        # time -- 2560 became 512 and the shape check caught it.
        self.register_buffer("embed_norm",
                             torch.tensor(float(embed_norm)) if embed_norm
                             else torch.tensor(0.0))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU and pool-AFTER-the-linears, both measured from their graph.
        x = self.linear2(F.relu(self.linear1(x)))
        x = self.pool(x)                      # base method: pools over TIME
        if self.embed_norm > 0:
            # Pinned to what the backbone RECEIVES, which for Gemma is the
            # embedding table times sqrt(hidden). Pinning to the raw table
            # makes the audio 51x quieter than the text beside it.
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.embed_norm
        return x

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_minicpm(cls, minicpm_projector, W_fwd, cognition_dim: int = 2560,
                     embed_norm: float | None = None):
        """Seed from MiniCPM's trained weights, retargeted to Gemma.

        `W_fwd` is the ridge map Qwen(4096) -> Gemma(2560) from
        training/fit_alignment.py. linear1 copies exactly; linear2 is composed
        with W_fwd because its output width changes.
        """
        src1, src2 = minicpm_projector.linear1, minicpm_projector.linear2
        W = torch.as_tensor(W_fwd, dtype=torch.float32)
        if W.shape != (src2.out_features, cognition_dim):
            raise ValueError(
                f"ridge map is {tuple(W.shape)}, expected "
                f"({src2.out_features}, {cognition_dim}) — a map fitted for a "
                "different backbone cannot retarget this adapter")

        m = cls(audio_dim=src1.in_features, cognition_dim=cognition_dim,
                mid=src2.out_features, embed_norm=embed_norm)
        with torch.no_grad():
            m.linear1.weight.copy_(src1.weight)
            m.linear1.bias.copy_(src1.bias)
            # y = h @ L2.T + b  then @ W   ->   weight = W.T @ L2, bias = b @ W
            m.linear2.weight.copy_(W.T @ src2.weight)
            m.linear2.bias.copy_(src2.bias @ W)
        return m


def demo() -> None:
    import numpy as np

    # a stand-in for MiniCPM's projector, with their real shapes
    class Src(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(1024, 4096)
            self.linear2 = nn.Linear(4096, 4096)

    src = Src()
    W = torch.randn(4096, 2560) * 0.02

    m = WhisperToGemma.from_minicpm(src, W, cognition_dim=2560, embed_norm=61.75)
    assert m.linear1.weight.shape == (4096, 1024)
    assert m.linear2.weight.shape == (2560, 4096)
    # linear1 must be COPIED, not re-initialised
    assert torch.equal(m.linear1.weight, src.linear1.weight)
    assert torch.equal(m.linear1.bias, src.linear1.bias)

    # the composition must equal running theirs then applying the ridge map
    h = torch.randn(2, 7, 1024)
    theirs = src.linear2(F.relu(src.linear1(h))) @ W
    ours = m.linear2(F.relu(m.linear1(h)))
    assert torch.allclose(theirs, ours, atol=1e-3), (theirs - ours).abs().max()

    # end to end: pooling and the norm pin
    out = m(torch.randn(1, 1500, 1024), duration_s=5.0)
    assert out.shape[-1] == 2560, out.shape
    n = out.norm(dim=-1)
    assert torch.allclose(n, torch.full_like(n, 61.75), atol=1e-2), n.mean()
    # trimming: 5 s of a 30 s window, pooled by 5
    assert out.shape[1] == 50, out.shape

    # EVERYTHING trainable — the whole point of retargeting instead of stacking
    total = sum(p.numel() for p in m.parameters())
    trainable = m.n_params()
    assert trainable == total, f"{trainable} of {total} trainable"

    # a ridge map for the wrong backbone must be refused, not silently used
    try:
        WhisperToGemma.from_minicpm(src, torch.randn(4096, 576), cognition_dim=2560)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-width ridge map must raise")

    print(f"whisper_to_gemma: linear1 copied, linear2 composed, "
          f"{trainable/1e6:.1f}M params ALL trainable, norm pinned, "
          f"wrong-width map refused")


if __name__ == "__main__":
    demo()
