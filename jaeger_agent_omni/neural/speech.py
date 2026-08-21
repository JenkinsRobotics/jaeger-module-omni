"""Direct Gemma-to-Talker conditioning; no decoded text at the boundary.

MiniCPM builds one 768-wide Talker condition per language token from Qwen's
token embeddings and hidden states. Jaeger learns the equivalent tensor from
Gemma's own token embeddings and hidden states. Text/tokenizer alignment is
permitted offline when building teacher pairs, but this runtime API accepts
tensors only.

    python -m jaeger_agent_omni.neural.speech
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpeechConditionerConfig:
    cognition_dim: int = 2560
    token_dim: int = 2560
    talker_dim: int = 768
    hidden_dim: int = 1536

    def __post_init__(self):
        if min(asdict(self).values()) < 1:
            raise ValueError("all conditioner dimensions must be positive")


class GemmaTalkerConditioner(nn.Module):
    """Gemma token/cognition tensors -> merged Talker condition tensors."""

    def __init__(self, config: SpeechConditionerConfig | None = None):
        super().__init__()
        self.config = config or SpeechConditionerConfig()
        c = self.config
        self.token_norm = nn.RMSNorm(c.token_dim)
        self.hidden_norm = nn.RMSNorm(c.cognition_dim)
        self.token_path = nn.Linear(c.token_dim, c.talker_dim)
        self.hidden_path = nn.Sequential(
            nn.Linear(c.cognition_dim, c.hidden_dim), nn.SiLU(),
            nn.Linear(c.hidden_dim, c.talker_dim))
        self.gate = nn.Sequential(
            nn.Linear(c.cognition_dim + c.token_dim, c.talker_dim),
            nn.Sigmoid())
        self.out_norm = nn.RMSNorm(c.talker_dim)
        self.out_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, token_embeddings: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        c = self.config
        if token_embeddings.ndim != 3 or hidden_states.ndim != 3:
            raise ValueError("conditioner inputs must be [batch, tokens, width]")
        if token_embeddings.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("token embeddings and hidden states must align")
        if token_embeddings.shape[-1] != c.token_dim:
            raise ValueError(f"expected token width {c.token_dim}, got "
                             f"{token_embeddings.shape[-1]}")
        if hidden_states.shape[-1] != c.cognition_dim:
            raise ValueError(f"expected hidden width {c.cognition_dim}, got "
                             f"{hidden_states.shape[-1]}")
        te = self.token_norm(token_embeddings)
        hs = self.hidden_norm(hidden_states)
        lexical = self.token_path(te)
        semantic = self.hidden_path(hs)
        mix = self.gate(torch.cat((te, hs), dim=-1))
        return self.out_norm(lexical + mix * semantic) * self.out_scale

    @staticmethod
    def loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Regress exact teacher magnitude and direction."""
        if predicted.shape != target.shape:
            raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
        mse = F.mse_loss(predicted, target)
        cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
        return mse + 0.1 * cosine

    def checkpoint(self) -> dict:
        return {"format": "jaeger-gemma-talker-conditioner-v1",
                "config": asdict(self.config), "state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "GemmaTalkerConditioner":
        if checkpoint.get("format") != "jaeger-gemma-talker-conditioner-v1":
            raise ValueError("not a Jaeger Gemma-to-Talker conditioner checkpoint")
        model = cls(SpeechConditionerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state"])
        return model.eval()


def demo() -> None:
    torch.manual_seed(0)
    cfg = SpeechConditionerConfig(cognition_dim=32, token_dim=32,
                                  talker_dim=12, hidden_dim=24)
    model = GemmaTalkerConditioner(cfg)
    token, hidden = torch.randn(2, 7, 32), torch.randn(2, 7, 32)
    out = model(token, hidden)
    assert out.shape == (2, 7, 12)
    assert model.loss(out, torch.randn_like(out)).isfinite()
    restored = GemmaTalkerConditioner.from_checkpoint(model.checkpoint())
    assert torch.allclose(out, restored(token, hidden))
    try:
        model(token[:, :-1], hidden)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned token/hidden sequences must fail")
    print("neural.speech: tensor-only boundary, guards, checkpoint OK")


if __name__ == "__main__":
    demo()
