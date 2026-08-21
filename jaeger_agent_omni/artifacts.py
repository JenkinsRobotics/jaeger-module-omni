"""Model artifacts — the industry-standard packaging of what we train.

A training checkpoint is not a release artifact. The checkpoint carries
optimizer moments, scheduler state and every frozen parameter it rode alongside
(112 MB for a 5 MB adapter, measured). The artifact is the opposite: only the
weights we trained, in safetensors, with a config that names exactly what they
pair with and a model card a human can read.

    artifacts/<name>/
        adapter_config.json         architecture + the pairing + provenance
        adapter_model.safetensors   ONLY the trained weights
        README.md                   model card: what, how trained, metrics

Naming follows the HF/PEFT adapter convention because that is what our trained
pieces are — adapters onto frozen giants. The frozen parts are *referenced* by
provenance, never redistributed: MiniCPM's encoder/projector reload from the
local GGUF, the backbone from its own repo.

    python -m jaeger_agent_omni.artifacts export <ckpt.pt> <out_dir>
    python -m jaeger_agent_omni.artifacts list
    python -m jaeger_agent_omni.artifacts     # self-check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"

# What a checkpoint's projector state splits into, per architecture:
#   minicpm_delta   trained keys live under "delta.", frozen under "minicpm."
#   scratch         every key is trained
_TRAINED_PREFIX = {"minicpm_delta": "delta.", "scratch": ""}


def export(checkpoint_path: str | Path, out_dir: str | Path) -> Path:
    """Training checkpoint -> release artifact. Strips everything untrained."""
    from safetensors.torch import save_file

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    arch = cfg.get("arch", "scratch")
    if arch not in _TRAINED_PREFIX:
        raise ValueError(f"unknown arch {arch!r} in checkpoint")

    prefix = _TRAINED_PREFIX[arch]
    state = ck["projector"]
    trained = {k[len(prefix):]: v.contiguous().clone()
               for k, v in state.items() if k.startswith(prefix)} if prefix else \
              {k: v.contiguous().clone() for k, v in state.items()}
    if not trained:
        raise ValueError(f"no trained weights under prefix {prefix!r} — "
                         f"keys: {sorted(state)[:4]}")

    n_params = sum(v.numel() for v in trained.values())
    history = ck.get("history", [])
    last = history[-1] if history else {}

    config = {
        "artifact_type": "delta_adapter" if arch == "minicpm_delta" else "projector",
        "modality": ck.get("modality", "audio"),
        "architecture": arch,
        "cognition_dim": ck["cognition_dim"],
        "embed_norm": round(float(ck.get("embed_norm", 0.0)), 6),
        "n_params": n_params,
        "pairing": {
            # A trained adapter is ONLY valid for this exact pairing. Loading
            # it against anything else is the silent failure that looks like a
            # bad model instead of a bad load.
            "backbone_id": cfg.get("backbone_id", "?"),
            "whisper_id": cfg.get("whisper_id", "?"),
            "front_end": ("openbmb/MiniCPM-o-4_5 audio GGUF (fine-tuned "
                          "whisper-medium encoder + 21M projector, frozen; "
                          "reloaded locally, not redistributed)")
                         if arch == "minicpm_delta" else "stock whisper encoder",
        },
        "training": {
            "stage": cfg.get("stage"),
            "steps": ck.get("step"),
            "chat_wrapped": ck.get("chat_wrapped", False),
            # None on legacy checkpoints = the old assistant default.
            "system_prompt": ck.get("system_prompt"),
            "prefix_fraction": cfg.get("prefix_fraction"),
            "final_loss": round(last.get("loss", float("nan")), 4) if last else None,
            "final_target_acc": round(last.get("target_acc", 0.0), 4) if last else None,
        },
        "format_version": 1,
    }
    if arch == "minicpm_delta":
        w = trained.get("net.weight")
        config["source_dim"] = int(w.shape[1]) if w is not None else None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(trained, out / WEIGHTS_NAME)
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    if not (out / "README.md").exists():
        (out / "README.md").write_text(_model_card(out.name, config))
    return out


def _model_card(name: str, c: dict) -> str:
    t = c["training"]
    return f"""# {name}

**{c['artifact_type']}** · {c['n_params']/1e6:.2f}M trained parameters ·
{c['modality']} → cognition

Maps MiniCPM-o 4.5's frozen audio space into
`{c['pairing']['backbone_id']}`'s embedding space. Valid **only** for that
exact pairing — an adapter is a bridge between two specific models, not a
free-standing artifact.

## Pairing (all required at load time)

```text
front end   {c['pairing']['front_end']}
backbone    {c['pairing']['backbone_id']}
whisper     {c['pairing']['whisper_id']}
embed_norm  {c['embed_norm']}   (output scale pinned to the backbone's
                                 mean text-embedding norm)
```

## Training

```text
stage            {t['stage']}
steps            {t['steps']}
chat template    {'spliced (audio stands where user text would)' if t['chat_wrapped'] else 'NOT used'}
prefix aug       {t['prefix_fraction']}
final loss       {t['final_loss']}
held-out acc     {t['final_target_acc']}   (teacher-forced; free-running WER is
                                            the honest metric — see repo STATUS)
```

## Load

```python
from jaeger_agent_omni.artifacts import load_adapter
adapter, config = load_adapter("artifacts/{name}")
```

Generated by `jaeger_agent_omni.artifacts`; edit freely — regeneration will not
overwrite an existing card.
"""


def load_adapter(artifact_dir: str | Path):
    """Artifact -> (module, config). The caller assembles the pairing."""
    from safetensors.torch import load_file

    d = Path(artifact_dir)
    config = json.loads((d / CONFIG_NAME).read_text())
    state = load_file(d / WEIGHTS_NAME)

    if config["artifact_type"] == "delta_adapter":
        from .neural import DeltaAdapter
        module = DeltaAdapter(source_dim=config["source_dim"],
                              cognition_dim=config["cognition_dim"],
                              embed_norm=config["embed_norm"] or None)
    else:
        from .neural import JaegerAudioProjector
        w1 = state["linear1.weight"]
        module = JaegerAudioProjector(audio_dim=w1.shape[1],
                                      cognition_dim=config["cognition_dim"],
                                      embed_norm=config["embed_norm"] or None)
    # The embed_norm buffer travels in the weights file too — the config copy
    # is for humans and for constructing; the tensor is the source of truth.
    module.load_state_dict(dict(state))
    return module.eval(), config


def list_artifacts(root: Path = ARTIFACTS_DIR) -> list:
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if (d / CONFIG_NAME).exists():
                c = json.loads((d / CONFIG_NAME).read_text())
                out.append((d.name, c))
    return out


def demo():
    import tempfile

    from .neural import DeltaAdapter, MiniCPMDeltaPath
    from .neural.audio import JaegerAudioProjector

    torch.manual_seed(0)
    front = JaegerAudioProjector(audio_dim=32, cognition_dim=64)
    delta = DeltaAdapter(source_dim=64, cognition_dim=48, embed_norm=3.0)
    path = MiniCPMDeltaPath(front, delta)

    fake_ck = {"projector": path.state_dict(), "cognition_dim": 48,
               "embed_norm": 3.0, "modality": "audio", "step": 500,
               "chat_wrapped": True,
               "config": {"arch": "minicpm_delta", "backbone_id": "test/backbone",
                          "whisper_id": "openai/whisper-medium",
                          "stage": "stage3_warmup", "prefix_fraction": 0.3},
               "history": [{"loss": 2.5, "target_acc": 0.41, "step": 500}]}

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ck.pt"
        torch.save(fake_ck, ckpt)
        out = export(ckpt, Path(tmp) / "my-adapter")

        # The artifact contains ONLY the trained weights — the 33% that was
        # trainable, not the frozen front end it rode with.
        from safetensors.torch import load_file
        stored = load_file(out / WEIGHTS_NAME)
        assert all(not k.startswith(("minicpm.", "delta.")) for k in stored), \
            "artifact keys must be unprefixed adapter keys"
        n_stored = sum(v.numel() for v in stored.values())
        # params + the 1-element embed_norm buffer (buffers are not parameters)
        assert n_stored == delta.n_params() + 1, (n_stored, delta.n_params())

        cfg = json.loads((out / CONFIG_NAME).read_text())
        assert cfg["pairing"]["backbone_id"] == "test/backbone"
        assert cfg["source_dim"] == 64
        assert cfg["training"]["final_target_acc"] == 0.41
        assert (out / "README.md").exists()

        # Round trip: the loaded adapter must be numerically identical.
        loaded, lcfg = load_adapter(out)
        x = torch.randn(2, 7, 64)
        assert torch.allclose(delta(x), loaded(x), atol=1e-6)
        assert lcfg["cognition_dim"] == 48

        # An existing model card is never clobbered by re-export.
        (out / "README.md").write_text("hand-edited")
        export(ckpt, out)
        assert (out / "README.md").read_text() == "hand-edited"

        # list sees it.
        found = list_artifacts(Path(tmp))
        assert [n for n, _ in found] == ["my-adapter"]

        # A checkpoint with nothing trained under the expected prefix fails
        # loudly — exporting an empty artifact would look like success.
        bad = dict(fake_ck)
        bad["projector"] = {"minicpm.linear1.weight": torch.zeros(2, 2)}
        torch.save(bad, ckpt)
        try:
            export(ckpt, Path(tmp) / "empty")
        except ValueError:
            pass
        else:
            raise AssertionError("empty artifact must be rejected")

    size = sum(v.numel() for v in delta.state_dict().values()) * 4 / 1e3
    print(f"artifacts: adapter {size:.0f}KB vs full checkpoint "
          f"{sum(v.numel() for v in fake_ck['projector'].values()) * 4 / 1e3:.0f}KB")
    print("artifacts: export strips frozen, round-trip, card preserved OK")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        demo()
        return 0
    if args[0] == "list":
        for name, c in list_artifacts():
            t = c["training"]
            print(f"{name:44s} {c['n_params']/1e6:5.2f}M  "
                  f"{c['pairing']['backbone_id']}  "
                  f"acc={t.get('final_target_acc')}")
        return 0
    if args[0] == "export" and len(args) == 3:
        out = export(args[1], args[2])
        c = json.loads((out / CONFIG_NAME).read_text())
        print(f"exported {c['n_params']/1e6:.2f}M params -> {out}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())


def load_audio_path(artifact_dir: str | Path):
    """Artifact -> (encoder_or_None, ready projector) with the pairing assembled.

    For a delta artifact this reloads MiniCPM's frozen front end from the local
    GGUF and returns THEIR encoder — the caller must use it in place of stock
    whisper, or the pairing is silently broken.

    ponytail: reaches into research/ for the GGUF loaders. Promote those into
    the package when a second caller needs them from outside a script.
    """
    module, config = load_adapter(artifact_dir)
    if config["artifact_type"] != "delta_adapter":
        return None, module

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
    from load_minicpm import load_front_end
    from .neural import MiniCPMDeltaPath

    encoder, front = load_front_end()
    return encoder, MiniCPMDeltaPath(front, module)
