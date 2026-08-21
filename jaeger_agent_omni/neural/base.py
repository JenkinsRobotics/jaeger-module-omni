"""The shared contract every Jaeger modality projector implements.

A specialist encoder produces frames at some rate in its own dimensionality.
Cognition wants tokens in *its* dimensionality, at a rate that does not blow up
prefill. The projector is what bridges the two, and every modality needs the
same three things: trim padding, compress time, project into cognition space.

    audio    → whisper encoder → JaegerAudioProjector  ─┐
    vision   → SigLIP          → JaegerVisionProjector ─┼→ shared cognition
    memory   → retrieval net   → JaegerMemoryProjector ─┘

This base exists because that arithmetic is identical across modalities and is
easy to get subtly wrong — the padding trim especially, which is the difference
between a 5 s utterance costing 50 tokens or 300.

A neural tool is just another projector. See the plan doc, "Tools: two paths".

    python -m jaeger_agent_omni.neural.base     # self-check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Populated by @register. Lets a training config name a modality as a string
# rather than importing a class, which is what makes stages configurable.
REGISTRY = {}


def register(name: str):
    """Claim a modality name.

    Two *different* classes claiming one name is a real bug and raises. The
    same class re-registering is not: `python -m pkg.neural.audio` imports the
    module twice — once through the package __init__, once as __main__ — and
    those are distinct class objects for identical source. Comparing qualnames
    keeps the guard meaningful without breaking module-as-script.
    """
    def wrap(cls):
        existing = REGISTRY.get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(
                f"modality {name!r} already registered to {existing.__qualname__}")
        REGISTRY[name] = cls
        cls.modality = name
        return cls
    return wrap


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown modality {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]


def pooled_tokens(n_frames: int, stride: int) -> int:
    """Output token count after average pooling.

    MiniCPM's formula, from audition.cpp:1385 — (in - k)/s + 1, floor. For
    stride-aligned inputs this equals n//stride, but the ragged case differs
    and the reference is what the trained weights expect.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if n_frames < stride:
        return 0
    return (n_frames - stride) // stride + 1


class ModalityProjector(nn.Module):
    """Base for encoder-output → cognition-embedding projectors.

    Subclasses implement `project()`. Trimming, pooling and the token
    arithmetic live here so every modality gets them identically.
    """

    modality: str = "base"

    def __init__(self, input_dim: int, cognition_dim: int, stride: int,
                 frame_rate: float, max_frames: int):
        super().__init__()
        if input_dim < 1 or cognition_dim < 1:
            raise ValueError("dims must be positive")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.input_dim = input_dim
        self.cognition_dim = cognition_dim
        self.stride = stride
        self.frame_rate = frame_rate
        self.max_frames = max_frames

    # --- subclass responsibility ------------------------------------------
    def project(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # --- shared ------------------------------------------------------------
    def valid_frames(self, duration_s: float) -> int:
        """Real (non-padding) frames for `duration_s`.

        Encoders that pad to a fixed window (Whisper: always 1500 frames) make
        every short input cost a full window unless it is trimmed. This is the
        single highest-leverage line in the class.
        """
        if duration_s <= 0:
            raise ValueError(f"duration must be positive, got {duration_s}")
        return min(self.max_frames, max(1, int(round(duration_s * self.frame_rate))))

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """Average-pool over time. avg_pool1d works on the last dim."""
        return F.avg_pool1d(x.transpose(1, 2), self.stride, self.stride).transpose(1, 2)

    def trim(self, x: torch.Tensor, duration_s: float | None) -> torch.Tensor:
        if duration_s is None:
            return x
        return x[:, : self.valid_frames(duration_s), :]

    def forward(self, x: torch.Tensor, duration_s: float | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim {self.input_dim}, got {x.shape[-1]}")
        x = self.trim(x, duration_s)
        if x.shape[1] < self.stride:
            raise ValueError(f"{x.shape[1]} frames < stride {self.stride}")
        return self.project(x)

    @property
    def tokens_per_second(self) -> float:
        return self.frame_rate / self.stride

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def token_cost(self, duration_s: float) -> int:
        """Cognition tokens this modality will spend on `duration_s` of input.

        The number to watch: audio tokens are denser than text, so prefill grows
        with the compression ratio. 10 tok/s against text's ~3 is the trade.
        """
        return pooled_tokens(self.valid_frames(duration_s), self.stride)

    def extra_repr(self) -> str:
        return (f"modality={self.modality}, {self.input_dim}->{self.cognition_dim}, "
                f"stride={self.stride}, {self.tokens_per_second:.1f} tok/s")


def demo():
    @register("_demo")
    class Demo(ModalityProjector):
        def __init__(self, **kw):
            super().__init__(input_dim=8, cognition_dim=4, stride=5,
                             frame_rate=50.0, max_frames=1500, **kw)
            self.fc = nn.Linear(8, 4)

        def project(self, x):
            return self.fc(self.pool(x))

    p = Demo()
    assert p.modality == "_demo"
    assert get("_demo") is Demo
    assert p.tokens_per_second == 10.0

    # Trimming: the same padded tensor costs 6x fewer tokens when the real
    # duration is known.
    assert p(torch.randn(1, 1500, 8)).shape == (1, 300, 4)
    assert p(torch.randn(1, 1500, 8), duration_s=5.0).shape == (1, 50, 4)
    assert p.token_cost(5.0) == 50 and p.token_cost(30.0) == 300

    # Duration beyond the encoder window clamps rather than over-reading.
    assert p.token_cost(120.0) == 300

    # Token formula matches the reference, including the ragged case.
    assert pooled_tokens(50, 5) == 10
    assert pooled_tokens(52, 5) == 10, "ragged tail is dropped, not padded"
    assert pooled_tokens(4, 5) == 0

    for bad, exc in ((torch.randn(10, 8), ValueError),
                     (torch.randn(1, 10, 99), ValueError),
                     (torch.randn(1, 2, 8), ValueError)):
        try:
            p(bad)
        except exc:
            pass
        else:
            raise AssertionError(f"should have rejected {tuple(bad.shape)}")

    try:
        p.valid_frames(0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive duration must raise")

    # Re-registering the same class is fine (module-as-script re-import)...
    register("_demo")(Demo)
    assert get("_demo") is Demo

    # ...but a different class claiming the name is a genuine collision.
    class Impostor(ModalityProjector):
        def project(self, x):
            return x

    try:
        register("_demo")(Impostor)
    except ValueError:
        pass
    else:
        raise AssertionError("colliding registration must raise")

    try:
        get("_nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown modality must raise")

    # Base is abstract in practice.
    try:
        ModalityProjector(8, 4, 5, 50.0, 1500).project(torch.randn(1, 5, 8))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("base project() must not be usable")

    REGISTRY.pop("_demo")
    print(f"neural.base: registry contract, trim/pool/token math OK")


if __name__ == "__main__":
    demo()

def hidden_size_of(backbone) -> int:
    """A backbone's cognition width, wherever its config keeps it.

    MULTIMODAL configs (Gemma4Config) put hidden_size on the TEXT config;
    plain causal LMs keep it at the top level. Reading the wrong one raises
    AttributeError at best -- and at worst, on a config that has both, builds
    a projector of the wrong width that fails silently much later.
    """
    cfg = getattr(backbone, "config", backbone)
    if hasattr(cfg, "get_text_config"):
        return cfg.get_text_config().hidden_size
    return cfg.hidden_size


def embedding_norm_of(backbone) -> float:
    """Mean norm of the embeddings the model ACTUALLY RECEIVES.

    Not `.weight.norm()`. Gemma4's embedding module is scaled --

        return super().forward(input_ids) * self.embed_scale   # sqrt(2560)

    -- so the raw table reads 1.20 while the model is handed 61.75, a 51x gap.
    A projector pinned to 1.20 emits an audio prefix 51x quieter than the text
    tokens beside it in the same sequence, and attention across positions that
    differ 51x in magnitude goes non-finite. Measured: 31 NaN losses in 600
    steps, unmoved by two learning-rate changes, because it was never an
    optimisation problem.

    Calling the module picks up whatever scaling it applies, for any backbone,
    with no per-model special-casing.
    """
    import torch

    emb = backbone.get_input_embeddings()
    n = min(2048, emb.weight.shape[0])
    with torch.no_grad():
        probe = torch.arange(n, device=emb.weight.device)
        return float(emb(probe).norm(dim=-1).mean())
