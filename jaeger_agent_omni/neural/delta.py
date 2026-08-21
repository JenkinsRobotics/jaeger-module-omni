"""DeltaAdapter — map MiniCPM's audio space onto a different backbone.

The "copy their baseline, train a delta" architecture:

    audio -> THEIR encoder (frozen) -> THEIR projector (frozen) -> Qwen space
                                                                       |
                                                              DeltaAdapter  <- trained
                                                                       |
                                                               our backbone

Their frozen stack already solved the hard problem — audio to a semantic
embedding space, discriminative (cross-clip cosine 0.861, measured). The delta
only has to learn a map between two LLM embedding spaces, which is a far
smaller ask than audio-to-language from scratch. Our from-scratch 554K
projector collapsed to a constant; this cannot collapse the front end because
the front end is not trainable.

    python -m jaeger_agent_omni.neural.delta     # self-check, no downloads
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QWEN3_8B_HIDDEN = 4096      # what MiniCPM's projector emits into


class DeltaAdapter(nn.Module):
    """source LLM space -> target LLM space.

    Default is a single Linear: embedding-space translation is close to linear
    in practice, and fewer trainable weights means less room to find degenerate
    solutions. `hidden` adds a bottleneck MLP as the A/B candidate.

    `embed_norm` pins output scale to the target backbone's own embedding norm
    — the frozen-backbone lesson from the scratch runs, kept.
    """

    def __init__(self, source_dim: int = QWEN3_8B_HIDDEN, cognition_dim: int = 2048,
                 hidden: int | None = None, embed_norm: float | None = None):
        super().__init__()
        if source_dim < 1 or cognition_dim < 1:
            raise ValueError("dims must be positive")
        self.source_dim = source_dim
        self.cognition_dim = cognition_dim
        if hidden:
            self.net = nn.Sequential(nn.Linear(source_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, cognition_dim))
        else:
            self.net = nn.Linear(source_dim, cognition_dim)
        self.register_buffer("embed_norm",
                             torch.tensor(float(embed_norm)) if embed_norm
                             else torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.source_dim:
            raise ValueError(f"expected source_dim {self.source_dim}, got {x.shape[-1]}")
        x = self.net(x)
        if self.embed_norm > 0:
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.embed_norm
        return x

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MiniCPMDeltaPath(nn.Module):
    """Their frozen front end + our trained delta, presenting the same
    interface as `JaegerAudioProjector` so Trainer and OmniApp need no changes.
    """

    modality = "audio"

    def __init__(self, minicpm_projector: nn.Module, delta: DeltaAdapter):
        super().__init__()
        if getattr(minicpm_projector, "cognition_dim", None) != delta.source_dim:
            raise ValueError(
                f"front end emits {getattr(minicpm_projector, 'cognition_dim', '?')}, "
                f"delta expects {delta.source_dim}")
        self.minicpm = minicpm_projector.eval().requires_grad_(False)
        self.delta = delta

    def forward(self, x: torch.Tensor, duration_s: float | None = None) -> torch.Tensor:
        with torch.no_grad():
            z = self.minicpm(x, duration_s=duration_s)
        return self.delta(z)

    # -- Trainer interface ---------------------------------------------------
    @property
    def cognition_dim(self) -> int:
        return self.delta.cognition_dim

    @property
    def embed_norm(self):
        return self.delta.embed_norm

    @property
    def tokens_per_second(self) -> float:
        return self.minicpm.tokens_per_second

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def demo():
    from .audio import JaegerAudioProjector

    torch.manual_seed(0)

    # Small stand-in for their projector: same class, toy dims.
    front = JaegerAudioProjector(audio_dim=32, cognition_dim=64)
    delta = DeltaAdapter(source_dim=64, cognition_dim=48, embed_norm=3.0)
    path = MiniCPMDeltaPath(front, delta)

    # Only the delta trains. This is the property that makes collapse
    # impossible for the front end and the whole point of the architecture.
    assert path.n_trainable() == delta.n_params()
    assert path.n_trainable() < path.n_params()

    out = path(torch.randn(2, 50, 32))
    assert out.shape == (2, 10, 48), out.shape
    assert path.cognition_dim == 48

    # embed_norm pinning holds on the composite output.
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 3.0), atol=1e-4)

    # Gradients reach the delta and never the front end.
    path(torch.randn(1, 50, 32)).sum().backward()
    assert delta.net.weight.grad is not None
    assert all(p.grad is None for p in front.parameters()), "front end must stay frozen"

    # A random linear map roughly preserves the input's similarity structure —
    # which is why this cannot silently collapse the way scratch training did.
    # The input needs REAL structure to test that: random gaussians in 64-dim
    # are all near-orthogonal, and correlating noise with noise proves nothing.
    anchors = F.normalize(torch.randn(3, 64), dim=-1)
    base = F.normalize(anchors.repeat_interleave(2, 0)
                       + 0.1 * torch.randn(6, 64), dim=-1)   # 3 tight pairs
    d2 = DeltaAdapter(64, 48)
    mapped = F.normalize(d2(base), dim=-1)
    for a in range(0, 6, 2):
        sims = (mapped[a] @ mapped.T).clone()
        sims[a] = -2
        assert int(sims.argmax()) == a + 1, \
            "each vector's nearest neighbour must survive the map"

    # Trimming still happens in the frozen front end.
    assert path(torch.randn(1, 1500, 32), duration_s=5.0).shape[1] == 50

    # Interface mismatches fail loudly at construction, not mid-run.
    try:
        MiniCPMDeltaPath(front, DeltaAdapter(source_dim=999, cognition_dim=48))
    except ValueError:
        pass
    else:
        raise AssertionError("front/delta dim mismatch must be rejected")

    try:
        delta(torch.randn(1, 10, 32))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong source_dim must be rejected")

    # state_dict round-trips the whole composite — checkpoint self-contained.
    p2 = MiniCPMDeltaPath(JaegerAudioProjector(audio_dim=32, cognition_dim=64),
                          DeltaAdapter(64, 48, embed_norm=3.0))
    p2.load_state_dict(path.state_dict())
    x = torch.randn(1, 50, 32)
    assert torch.allclose(path(x), p2(x), atol=1e-6)

    mlp = DeltaAdapter(64, 48, hidden=128)
    assert mlp.n_params() > DeltaAdapter(64, 48).n_params()

    print(f"neural.delta: {delta.n_params()/1e3:.1f}K trainable of "
          f"{path.n_params()/1e3:.0f}K total ({path.n_trainable()/path.n_params()*100:.0f}%)")
    print("neural.delta: freeze split, norm pin, structure preservation OK")


if __name__ == "__main__":
    demo()
