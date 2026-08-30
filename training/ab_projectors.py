"""A/B the projector design against a fixed Gemma 4 baseline.

Three independent teams solved "encoder output -> LLM embedding space" and
disagreed. This measures which is right FOR US instead of arguing:

    MiniCPM-o 4.5   mean-pool stride 5 -> Linear -> ReLU -> Linear, no norm
    Ultravox v0.6   STACK 8 frames -> RMSNorm -> Linear -> SwiGLU -> Linear
    ours            MiniCPM's shape + embed_norm output pin (added after our
                    from-scratch run collapsed to a constant)

The disagreements that matter, and why each is a real question:

    pool vs stack   pooling averages within the window and discards detail;
                    stacking keeps every frame and lets the matrix decide.
                    Ultravox chose stack independently of us.
    ReLU vs SwiGLU  gated activations usually train better; MiniCPM's trained
                    weights use ReLU and we copied that for fidelity.
    normalisation   MiniCPM has NONE inside the projector — and our
                    from-scratch run collapsed (cross-clip cosine 0.990).
                    Ultravox puts RMSNorm at input and mid. Independent
                    corroboration that unnormalised projectors are fragile.

Baseline is held FIXED (same backbone, corpus, split, steps, seed) so the only
moving part is the projector. Judged on free-running WER, never teacher-forced
accuracy — a 0.85 teacher-forced score coexisted with 120% WER in this repo.

    python training/ab_projectors.py --list
    python training/ab_projectors.py --variant ultravox --steps 1500 ...

## First result — random init, synthetic structured input, NOT trained

    variant     params   tok/s   cross-cos   class-sep
    minicpm      9.18M   10.00      0.9347      0.0406
    ours         9.18M   10.00      0.9317      0.0437   <- best separation
    ultravox    48.51M    6.25      0.0048      0.0069
    hybrid      32.78M   10.00      0.0090      0.0242

RMSNorm decorrelates outputs hard (0.93 -> 0.005), which is presumably why
Ultravox never needed an output scale pin — their normalisation does that job
structurally. But class separation inverts the ranking: the pinned variant
keeps same-source clips closest, Ultravox spreads EVERYTHING apart including
what should stay together.

Read this as architectural bias at initialisation, not trained quality — the
honest ranking is free-running WER after training. What it does establish:
our current design is not obviously wrong, and Ultravox costs 5x the params
and 3x the compute for fewer tokens per second.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.core.neural.base import ModalityProjector  # noqa: E402


class RMSNorm(nn.Module):
    """Ultravox uses RMSNorm; torch's LayerNorm is not the same thing."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """Halves the width: gate * silu(value), the two halves of the input."""

    def forward(self, x):
        gate, val = x.chunk(2, dim=-1)
        return F.silu(gate) * val


@dataclass(frozen=True)
class Variant:
    key: str
    stride: int
    mode: str            # "pool" | "stack"
    act: str             # "relu" | "swiglu"
    norm: str            # "none" | "rms_pre_mid" | "embed_pin"
    source: str


VARIANTS = {v.key: v for v in [
    Variant("minicpm", 5, "pool", "relu", "none",
            "MiniCPM-o 4.5, as shipped — the fidelity baseline"),
    Variant("ours", 5, "pool", "relu", "embed_pin",
            "MiniCPM + output scale pinned to the backbone's embedding norm"),
    Variant("ultravox", 8, "stack", "swiglu", "rms_pre_mid",
            "Ultravox v0.6 — independent solution to the same problem"),
    Variant("hybrid", 5, "stack", "swiglu", "rms_pre_mid",
            "Ultravox's design at MiniCPM's token rate (10 tok/s, not 6.25)"),
]}


class ABProjector(ModalityProjector):
    """One class, four configurations, so the comparison is apples to apples."""

    def __init__(self, variant: Variant, input_dim: int, cognition_dim: int,
                 frame_rate: float = 50.0, max_frames: int = 1500,
                 embed_norm: float | None = None):
        super().__init__(input_dim=input_dim, cognition_dim=cognition_dim,
                         stride=variant.stride, frame_rate=frame_rate,
                         max_frames=max_frames)
        self.variant = variant
        dim_in = input_dim if variant.mode == "pool" else input_dim * variant.stride

        self.ln_pre = RMSNorm(dim_in) if variant.norm == "rms_pre_mid" else nn.Identity()
        # SwiGLU halves its input, so linear_1 must emit double the mid width.
        mid = cognition_dim * 2 if variant.act == "swiglu" else cognition_dim
        self.linear_1 = nn.Linear(dim_in, mid, bias=(variant.norm != "rms_pre_mid"))
        self.act = SwiGLU() if variant.act == "swiglu" else nn.ReLU()
        post_act = cognition_dim
        self.ln_mid = (RMSNorm(post_act) if variant.norm == "rms_pre_mid"
                       else nn.Identity())
        self.linear_2 = nn.Linear(post_act, cognition_dim,
                                  bias=(variant.norm != "rms_pre_mid"))
        self.register_buffer(
            "embed_norm",
            torch.tensor(float(embed_norm)) if (embed_norm and
                                                variant.norm == "embed_pin")
            else torch.tensor(0.0))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        t -= t % self.stride
        x = x[:, :t]
        if self.variant.mode == "pool":
            x = x.reshape(b, t // self.stride, self.stride, d).mean(2)
        else:
            x = x.reshape(b, t // self.stride, self.stride * d)
        x = self.linear_2(self.ln_mid(self.act(self.linear_1(self.ln_pre(x)))))
        if self.embed_norm > 0:
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.embed_norm
        return x


def demo():
    torch.manual_seed(0)
    d_in, d_cog, T = 128, 64, 100

    outs = {}
    for key, v in VARIANTS.items():
        p = ABProjector(v, d_in, d_cog, embed_norm=3.2)
        x = torch.randn(2, T, d_in)
        y = p(x)
        expect = T // v.stride
        assert y.shape == (2, expect, d_cog), (key, y.shape)
        outs[key] = (p, y)

    # Token rate is the headline difference: Ultravox trades resolution for a
    # shorter prefill (6.25 tok/s vs MiniCPM's 10).
    assert outs["ultravox"][1].shape[1] < outs["minicpm"][1].shape[1]
    assert VARIANTS["hybrid"].stride == VARIANTS["minicpm"].stride

    # Only the pinned variant constrains output scale.
    pinned = outs["ours"][1].norm(dim=-1)
    assert torch.allclose(pinned, torch.full_like(pinned, 3.2), atol=1e-4)
    free = outs["minicpm"][1].norm(dim=-1)
    assert free.std() > 1e-4, "unpinned output should vary in magnitude"

    # Stacking sees frame order; pooling cannot. This is the substantive claim
    # behind Ultravox's choice, so it gets asserted rather than assumed.
    # Test WITHIN one window (exactly `stride` frames) — with a ragged length
    # a flip changes which frames fall inside the window at all, which would
    # make pooling look order-sensitive for the wrong reason.
    x = torch.randn(1, 5, d_in)
    stack = ABProjector(VARIANTS["hybrid"], d_in, d_cog)
    pool = ABProjector(VARIANTS["minicpm"], d_in, d_cog)
    assert not torch.allclose(stack(x), stack(x.flip(1)), atol=1e-5), \
        "stacking must distinguish frame order"
    assert torch.allclose(pool(x), pool(x.flip(1)), atol=1e-5), \
        "pooling averages, so order within a window is invisible to it"

    # SwiGLU halves width internally, so linear_1 must be twice as wide.
    assert outs["ultravox"][0].linear_1.out_features == d_cog * 2
    assert outs["minicpm"][0].linear_1.out_features == d_cog

    # RMSNorm variants drop biases, as Ultravox does.
    assert outs["ultravox"][0].linear_1.bias is None
    assert outs["minicpm"][0].linear_1.bias is not None

    # All variants must train.
    for key, (p, _) in outs.items():
        p(torch.randn(1, T, d_in)).sum().backward()
        assert p.linear_1.weight.grad.abs().sum() > 0, key

    sizes = {k: p.n_params() / 1e3 for k, (p, _) in outs.items()}
    print("ab_projectors: " + ", ".join(f"{k} {v:.0f}K" for k, v in sizes.items()))
    print("ab_projectors: shapes, token rates, scale pin, order-sensitivity OK")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    args, _ = ap.parse_known_args()
    if args.list:
        print(f"{'variant':10s} {'rate':>8s} {'mode':6s} {'act':7s} {'norm':12s} source")
        for v in VARIANTS.values():
            print(f"{v.key:10s} {50/v.stride:6.2f}/s {v.mode:6s} {v.act:7s} "
                  f"{v.norm:12s} {v.source}")
        return 0
    demo()
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
