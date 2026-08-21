"""JaegerAudioProjector — the first Jaeger neural network.

A direct port of MiniCPM-o 4.5's audio projector, read from its weights and
from `llama.cpp-omni/tools/omni/audition.cpp` rather than guessed.

Ground truth (audition.cpp, "Step 6: audio projector"):

    encoder -> ln_post -> linear1 -> ReLU -> linear2 -> AvgPool(k=5, s=5)

    audio_pool_step                  5              50 fps -> 10 tokens/s
    audio_projector.linear1.weight   [1024, 4096]   whisper d_model -> LLM hidden
    audio_projector.linear2.weight   [4096, 4096]
                                     21.0M params

Three details that are easy to get wrong and were wrong here first:

  1. ReLU, not GELU.
  2. No LayerNorm inside the projector. The norm ahead of it is Whisper's own
     final `ln_post`, part of the encoder.
  3. Pooling happens AFTER both linears, not before. The linears therefore run
     at the full 50 fps and averaging happens in projected space. Costs 5x the
     projector FLOPs; it is what the reference does and what its weights were
     trained for.

What is NOT there: no perceiver resampler, no learned queries, no attention
pooling. The projector cannot select or reject anything. Whatever filtering of
irrelevant audio happens, happens in the LLM's attention downstream. Do not add
a resampler without a measurement saying this is the bottleneck.

## One deliberate deviation: `embed_norm`

MiniCPM has no output normalisation and neither did this, until a real run
showed why it needs one. Measured over 150 steps of Stage 3 warmup against a
frozen SmolLM2-135M:

    fresh projector      out-norm/token  3.97   =  1.24x the text-embedding norm
    after 150 steps      out-norm/token 14.23   =  4.44x
    weight norms         13.85 -> 14.08         (barely moved)

The weights did not grow — the projector rotated into a configuration emitting
embeddings four times larger than any token the frozen backbone was trained on.
It drove its own input out of distribution: loss climbed from 1.77 back to 3.2
while gradient norms went 47 -> 648.

MiniCPM never meets this because it trains the backbone *jointly*, so the
backbone adapts to whatever the projector emits. Training against a frozen
backbone is a different regime and imposes a constraint their architecture does
not have to satisfy. `embed_norm` re-scales each output token to the backbone's
own mean embedding norm, leaving direction free and the scale pinned.

Set `embed_norm=None` to reproduce the reference exactly — correct if the
backbone is being trained too, wrong if it is frozen.

    python -m jaeger_agent_omni.neural.audio     # self-check, no download
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModalityProjector, pooled_tokens, register

WHISPER_FRAME_RATE = 50      # frames per second out of the encoder
WHISPER_PAD_FRAMES = 1500    # 30 s — also the streaming KV-cache depth
WHISPER_MEDIUM_DIM = 1024    # d_model

MINICPM_POOL_STEP = 5        # audio_pool_step, from the GGUF header
MINICPM_PROJECTOR_PARAMS = 21_000_000


@register("audio")
class JaegerAudioProjector(ModalityProjector):
    """MiniCPM-o's audio projector with a swappable cognition dimension.

    `pool_last=True` reproduces the reference exactly. `pool_last=False` pools
    before the linears instead — 5x cheaper, averages in encoder space rather
    than projected space. Kept as the A/B candidate, not the default: the
    default should be the thing known to work.
    """

    def __init__(
        self,
        audio_dim: int = WHISPER_MEDIUM_DIM,
        cognition_dim: int = 2048,      # gemma-4-E4B hidden_size — read at load
        stride: int = MINICPM_POOL_STEP,
        pool_last: bool = True,
        embed_norm: float | None = None,
    ):
        super().__init__(input_dim=audio_dim, cognition_dim=cognition_dim,
                         stride=stride, frame_rate=WHISPER_FRAME_RATE,
                         max_frames=WHISPER_PAD_FRAMES)
        self.pool_last = pool_last
        # MiniCPM: linear1 1024->4096, linear2 4096->4096. Hidden == cognition.
        self.linear1 = nn.Linear(audio_dim, cognition_dim)
        self.linear2 = nn.Linear(cognition_dim, cognition_dim)
        # Buffer, not a plain attribute: it must ride along in the state dict,
        # or a reloaded projector silently emits a different scale.
        self.register_buffer("embed_norm",
                             torch.tensor(float(embed_norm)) if embed_norm
                             else torch.tensor(0.0))

    @classmethod
    def for_backbone(cls, backbone, audio_dim: int, **kw) -> "JaegerAudioProjector":
        """Build one matched to a frozen backbone.

        Reads cognition_dim and the mean text-embedding norm from the model
        itself — the two numbers that must never be hardcoded.
        """
        emb = backbone.get_input_embeddings().weight
        return cls(audio_dim=audio_dim,
                   cognition_dim=backbone.config.hidden_size,
                   embed_norm=emb.norm(dim=-1).mean().item(), **kw)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """x is expected post-`ln_post` — i.e. straight off the encoder."""
        if not self.pool_last:
            x = self.pool(x)
        x = self.linear2(F.relu(self.linear1(x)))       # ReLU. Not GELU.
        if self.pool_last:
            x = self.pool(x)                             # pool AFTER, as reference
        if self.embed_norm > 0:
            # Pin the scale, leave the direction free. Without this the
            # projector drifts out of the frozen backbone's input distribution.
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.embed_norm
        return x


def demo():
    torch.manual_seed(0)
    cognition_dim = 2048
    proj = JaegerAudioProjector(cognition_dim=cognition_dim)

    assert proj.modality == "audio"
    assert proj.stride == MINICPM_POOL_STEP
    assert proj.tokens_per_second == 10.0

    assert pooled_tokens(50, 5) == 10 and pooled_tokens(1500, 5) == 300

    # Full 30 s pad.
    assert proj(torch.randn(2, WHISPER_PAD_FRAMES, 1024)).shape == (2, 300, cognition_dim)

    # 5 s of real audio inside the same padded tensor: 50 tokens, not 300.
    out = proj(torch.randn(1, WHISPER_PAD_FRAMES, 1024), duration_s=5.0)
    assert out.shape == (1, 50, cognition_dim), out.shape
    assert 300 / out.shape[1] == 6.0, "trimming is the whole point"

    # One 1 s streaming chunk -> 10 tokens. This is the Stage 5 unit.
    assert proj(torch.randn(1, 50, 1024)).shape == (1, 10, cognition_dim)

    # Ragged tail dropped, not padded.
    assert proj(torch.randn(1, 1502, 1024)).shape[1] == 300

    # Same order of magnitude as MiniCPM's 21M; narrower only because E4B is.
    assert proj.n_params() < MINICPM_PROJECTOR_PARAMS * 2, proj.n_params()

    proj(torch.randn(1, 250, 1024)).sum().backward()
    assert proj.linear1.weight.grad.abs().sum() > 0

    # ReLU, not GELU: strongly negative pre-activations must be killed dead,
    # not leaked through. Regression guard on the activation choice.
    probe = JaegerAudioProjector(audio_dim=4, cognition_dim=8, stride=1)
    with torch.no_grad():
        probe.linear1.weight.fill_(-1.0)
        probe.linear1.bias.zero_()
        probe.linear2.weight.fill_(1.0)
        probe.linear2.bias.zero_()
        got = probe(torch.ones(1, 1, 4))
    assert got.abs().max() == 0.0, f"expected ReLU to zero this, got {got.abs().max()}"

    # pool-before is the cheap A/B variant and must differ from the reference.
    early = JaegerAudioProjector(cognition_dim=cognition_dim, pool_last=False)
    early.load_state_dict(proj.state_dict())
    x = torch.randn(1, 50, 1024)
    assert not torch.allclose(proj(x), early(x), atol=1e-4), \
        "pool-before and pool-after must not coincide — ReLU is nonlinear"
    assert proj(x).shape == early(x).shape

    # embed_norm pins output scale to the backbone's own embedding norm.
    scaled = JaegerAudioProjector(audio_dim=64, cognition_dim=32, embed_norm=3.2)
    out = scaled(torch.randn(2, 50, 64))
    got = out.norm(dim=-1)
    assert torch.allclose(got, torch.full_like(got, 3.2), atol=1e-4), got

    # Direction stays free — only magnitude is constrained.
    a = scaled(torch.randn(1, 50, 64))
    b = scaled(torch.randn(1, 50, 64))
    assert not torch.allclose(a, b), "normalisation must not collapse outputs"

    # Off by default, so the plain constructor is still the faithful port.
    assert float(proj.embed_norm) == 0.0
    assert proj(torch.randn(1, 50, 1024)).norm(dim=-1).std() > 0, \
        "unnormalised output should vary in magnitude"

    # The scale must survive a save/load round trip, or a reloaded projector
    # quietly emits different embeddings than the one that was trained.
    reloaded = JaegerAudioProjector(audio_dim=64, cognition_dim=32)
    reloaded.load_state_dict(scaled.state_dict())
    assert abs(float(reloaded.embed_norm) - 3.2) < 1e-5, float(reloaded.embed_norm)

    print(f"neural.audio  {proj.n_params()/1e6:5.1f}M params  "
          f"{proj.tokens_per_second:.0f} tok/s  1024 -> {cognition_dim}")
    print(f"minicpm ref   {MINICPM_PROJECTOR_PARAMS/1e6:5.1f}M params  "
          f"{WHISPER_FRAME_RATE/MINICPM_POOL_STEP:.0f} tok/s  1024 -> 4096")
    print("neural.audio: shapes, token formula, trimming, ReLU, pool-order, "
          "embed_norm OK")


if __name__ == "__main__":
    demo()
