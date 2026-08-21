"""Training run configuration.

One dataclass, validated. Presets per stage rather than a config framework —
there are four runs, not four hundred.

    python training/config.py     # self-check
"""

from dataclasses import dataclass, asdict, field


@dataclass
class TrainConfig:
    stage: str = "stage3_warmup"

    # Models. cognition_dim is NOT set here — it is read from the backbone's
    # config at load time. Hardcoding it is how a silent dim mismatch gets
    # broadcast into garbage.
    whisper_id: str = "openai/whisper-medium"
    backbone_id: str = ""              # gemma-4-E4B; must be set explicitly

    # Architecture.
    #   scratch        our projector, trained from random init (collapsed at
    #                  this scale — kept as the baseline)
    #   minicpm_delta  THEIR fine-tuned encoder + THEIR projector, frozen;
    #                  train only a DeltaAdapter into our backbone's space
    arch: str = "minicpm_delta"

    # Projector (scratch arch). Defaults are MiniCPM's measured values.
    pool_stride: int = 5               # 50 fps -> 10 tokens/s
    pool_last: bool = True             # pool AFTER the linears, as reference

    # Optimisation
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 20_000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Precision. bf16 on CUDA; fp16 on MPS, where bf16 falls back silently.
    dtype: str = "bf16"

    # Freezing. Stage 3/4 train the projector only; gradients still flow
    # THROUGH the frozen backbone, which is why VRAM stays high.
    train_backbone_lora: bool = False
    lora_rank: int = 16

    # Data
    total_samples: int = 100_000
    min_confidence: float = 0.8        # ignored for ground-truth tiers
    val_fraction: float = 0.1          # held out — target_acc is meaningless
                                       # without it on a small corpus

    # Concurrent full duplex is the default operating mode, so it is what the
    # model is trained for. Prefixes put mid-utterance audio in-distribution
    # and supply turn-end labels for free. Set 0.0 only to reproduce a
    # whole-utterance-only baseline.
    prefix_fraction: float = 0.3
    max_audio_s: float = 30.0          # whisper's context ceiling

    eval_every: int = 500
    save_every: int = 2000        # clamped below — see checkpoint_every
    seed: int = 0
    notes: str = ""

    def __post_init__(self):
        from data import STAGE_MIX          # local import keeps this testable

        if self.stage not in STAGE_MIX:
            raise ValueError(f"unknown stage {self.stage!r}")
        if not self.backbone_id:
            raise ValueError(
                "backbone_id must be set explicitly — no default, because the "
                "projector is only valid for the backbone it was trained against")
        if self.arch not in ("scratch", "minicpm_delta"):
            raise ValueError(f"arch {self.arch!r} unknown")
        if self.arch == "minicpm_delta" and "medium" not in self.whisper_id:
            raise ValueError(
                "minicpm_delta requires whisper-medium: their projector was "
                "trained on 1024-dim states from their fine-tuned medium encoder")
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError(f"dtype {self.dtype!r} not supported")
        if self.pool_stride < 1:
            raise ValueError("pool_stride must be >= 1")
        if not 0 < self.lr < 1:
            raise ValueError(f"lr {self.lr} is not plausible")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        if self.max_audio_s > 30.0:
            raise ValueError("whisper's context is 30 s; longer must be chunked")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError(f"val_fraction {self.val_fraction} outside (0, 0.5)")
        if not 0.0 <= self.prefix_fraction < 1.0:
            raise ValueError(
                f"prefix_fraction {self.prefix_fraction} outside [0, 1) — at 1.0 "
                "no complete utterance is ever seen")

    @property
    def checkpoint_every(self) -> int:
        """Steps between checkpoints, clamped so any run saves several times.

        A 300-step run with save_every=2000 never checkpoints, so an
        interruption loses everything — which is exactly what happened the
        first time this was exercised for real. Atomic writes protect the
        previous checkpoint; they cannot conjure one that was never taken.
        """
        return max(1, min(self.save_every, self.max_steps // 4))

    @property
    def tokens_per_second(self) -> float:
        return 50.0 / self.pool_stride

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def to_dict(self) -> dict:
        return asdict(self)


def warmup(backbone_id: str, **kw) -> TrainConfig:
    """Stage 3: audio->transcript. Cheap, stable, gets the projector in the door.

    Preset values are defaults, not overrides — an explicit kwarg wins, so a
    short smoke run can borrow the preset without fighting it.
    """
    kw.setdefault("lr", 3e-4)
    kw.setdefault("max_steps", 20_000)
    return TrainConfig(stage="stage3_warmup", backbone_id=backbone_id, **kw)


def distill(backbone_id: str, **kw) -> TrainConfig:
    """Stage 4: hidden-state alignment. Lower lr — the projector already works."""
    kw.setdefault("lr", 1e-4)
    kw.setdefault("max_steps", 40_000)
    return TrainConfig(stage="stage4_distill", backbone_id=backbone_id, **kw)


def demo():
    cfg = warmup("google/gemma-4-E4B-it")
    assert cfg.tokens_per_second == 10.0
    assert cfg.effective_batch == 32
    assert cfg.pool_last is True, "default must match MiniCPM"
    assert TrainConfig(**cfg.to_dict()) == cfg, "must round-trip"

    d = distill("google/gemma-4-E4B-it")
    assert d.lr < cfg.lr, "distill should be gentler than warmup"
    assert d.max_steps > cfg.max_steps

    # An explicit kwarg must beat the preset — smoke runs depend on it.
    short = warmup("x", max_steps=120, warmup_steps=10)
    assert short.max_steps == 120 and short.lr == 3e-4

    # backbone_id has no default on purpose: a projector is only valid for the
    # backbone it was trained against, so silently picking one is a trap.
    try:
        TrainConfig(stage="stage3_warmup")
    except ValueError as exc:
        assert "backbone_id" in str(exc)
    else:
        raise AssertionError("missing backbone_id must raise")

    for bad in (
        dict(stage="stage9"),
        dict(dtype="int4"),
        dict(pool_stride=0),
        dict(lr=5.0),
        dict(warmup_steps=99_000, max_steps=1_000),
        dict(max_audio_s=60.0),          # past whisper's context
        dict(val_fraction=0.0),          # no held-out set at all
        dict(val_fraction=0.9),          # more val than train
        dict(prefix_fraction=1.0),       # never sees a complete utterance
        dict(prefix_fraction=-0.1),
        dict(arch="quantum"),
        dict(arch="minicpm_delta", whisper_id="openai/whisper-tiny"),
    ):
        try:
            TrainConfig(backbone_id="x", **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected {bad}")

    # Any run checkpoints at least 4 times, however short.
    for steps in (40, 300, 20_000, 100_000):
        c = TrainConfig(backbone_id="x", max_steps=steps,
                        warmup_steps=max(1, steps // 10))
        assert c.checkpoint_every >= 1
        assert steps // c.checkpoint_every >= 4, (steps, c.checkpoint_every)

    print(f"config: warmup lr={cfg.lr} steps={cfg.max_steps} "
          f"eff_batch={cfg.effective_batch} {cfg.tokens_per_second:.0f} tok/s")
    print("config: presets, round-trip, guards OK")


if __name__ == "__main__":
    demo()
