"""Direct Gemma-to-Talker conditioning; no decoded text at the boundary.

MiniCPM builds one 768-wide Talker condition per language token from Qwen's
token embeddings and hidden states. Jaeger learns the equivalent tensor from
Gemma's own token embeddings and hidden states. Text/tokenizer alignment is
permitted offline when building teacher pairs, but this runtime API accepts
tensors only.

    python -m jaeger_agent_omni.core.neural.speech
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        # Begin as a lexical bridge. A random contextual branch overwhelms a
        # useful pretrained token map before it has learned anything.
        nn.init.zeros_(self.hidden_path[-1].weight)
        nn.init.zeros_(self.hidden_path[-1].bias)

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
        return (lexical + mix * semantic) * self.out_scale

    @staticmethod
    def loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Regress exact teacher magnitude and direction."""
        if predicted.shape != target.shape:
            raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
        mse = F.mse_loss(predicted, target)
        cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
        return mse + 0.1 * cosine

    def checkpoint(self) -> dict:
        return {"format": "jaeger-gemma-talker-conditioner-v2",
                "config": asdict(self.config), "state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "GemmaTalkerConditioner":
        if checkpoint.get("format") != "jaeger-gemma-talker-conditioner-v2":
            raise ValueError("not a Jaeger Gemma-to-Talker conditioner checkpoint")
        model = cls(SpeechConditionerConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state"])
        return model.eval()


class TokenIdBridge:
    """Deterministic Gemma-ID -> Qwen-ID expansion without runtime text.

    The table is compiled offline from tokenizer vocabularies. Production
    receives integer IDs and aligned hidden tensors only. When one Gemma token
    expands to several Qwen tokens, its contextual state is repeated so the
    Talker's lexical and semantic streams remain exactly aligned.
    """

    def __init__(self, offsets: np.ndarray, values: np.ndarray):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.int64)
        if self.offsets.ndim != 1 or self.values.ndim != 1:
            raise ValueError("token bridge arrays must be flat")
        if len(self.offsets) < 2 or self.offsets[0] != 0:
            raise ValueError("invalid token bridge offsets")
        if self.offsets[-1] != len(self.values):
            raise ValueError("token bridge payload length mismatch")

    @classmethod
    def load(cls, path) -> "TokenIdBridge":
        data = np.load(path)
        return cls(data["offsets"], data["values"])

    def expand(self, source_ids, hidden_states: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [int(x) for x in source_ids]
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("hidden_states must be [1, tokens, width]")
        if hidden_states.shape[1] != len(ids):
            raise ValueError("one hidden state is required per source token")
        target, rows = [], []
        for i, token_id in enumerate(ids):
            if token_id < 0 or token_id + 1 >= len(self.offsets):
                raise ValueError(f"source token {token_id} outside bridge")
            lo, hi = self.offsets[token_id:token_id + 2]
            mapped = self.values[lo:hi].tolist()
            target.extend(mapped)
            rows.extend([hidden_states[:, i:i + 1]] * len(mapped))
        if not target:
            raise ValueError("source sequence mapped to no Talker tokens")
        qwen_ids = torch.tensor([target], dtype=torch.long,
                                device=hidden_states.device)
        return qwen_ids, torch.cat(rows, dim=1)


@dataclass(frozen=True)
class HiddenBridgeConfig:
    source_dim: int = 2560
    target_dim: int = 4096
    rank: int = 128
    target_norm: float = 72.6281


class GemmaQwenHiddenBridge(nn.Module):
    """Learn Gemma cognition -> Qwen cognition for Talker semantics."""

    def __init__(self, config: HiddenBridgeConfig | None = None):
        super().__init__()
        self.config = config or HiddenBridgeConfig()
        c = self.config
        self.norm = nn.RMSNorm(c.source_dim)
        self.net = nn.Sequential(nn.Linear(c.source_dim, c.rank, bias=False),
                                 nn.GELU(),
                                 nn.Linear(c.rank, c.target_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(self.norm(x))
        return F.normalize(y, dim=-1) * self.config.target_norm

    @staticmethod
    def loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
        norm_target = F.normalize(target, dim=-1) * predicted.norm(
            dim=-1, keepdim=True).detach()
        return cosine + 0.02 * F.mse_loss(predicted, norm_target)

    def checkpoint(self) -> dict:
        return {"format": "jaeger-gemma-qwen-hidden-v1",
                "config": asdict(self.config), "state": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "GemmaQwenHiddenBridge":
        if checkpoint.get("format") != "jaeger-gemma-qwen-hidden-v1":
            raise ValueError("not a Gemma-Qwen hidden bridge checkpoint")
        model = cls(HiddenBridgeConfig(**checkpoint["config"]))
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
    bridge = TokenIdBridge(np.array([0, 1, 3]), np.array([9, 10, 11]))
    qids, repeated = bridge.expand([0, 1], torch.randn(1, 2, 4))
    assert qids.tolist() == [[9, 10, 11]] and repeated.shape == (1, 3, 4)
    assert torch.equal(repeated[:, 1], repeated[:, 2])
    hb = GemmaQwenHiddenBridge(HiddenBridgeConfig(32, 48, 8, 7.0))
    hy = hb(torch.randn(2, 5, 32))
    assert hy.shape == (2, 5, 48)
    assert torch.allclose(hy.norm(dim=-1), torch.full((2, 5), 7.0), atol=1e-4)
    print("neural.speech: tensor-only boundary, ID expansion, guards, checkpoint OK")


if __name__ == "__main__":
    demo()
