"""Where does the NaN enter? Instrument every boundary, do not guess.

Stage 3 against E4B produced 31 non-finite losses. Tuning the learning rate
twice did not move it, so it is not an optimisation problem. This walks one
real batch through the chain and reports the first stage whose output stops
being finite:

    waveform -> whisper encoder -> projector -> delta -> backbone -> loss

Also runs the backbone in fp32 for the same batch, because the suspect is
bf16: ~3 decimal digits against fp32's ~7, and 42 layers to accumulate in.

    python research/nan_hunt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))


def check(name, t) -> bool:
    import torch
    if t is None:
        print(f"  {name:24s} None"); return True
    bad = (~torch.isfinite(t)).sum().item()
    a = t.detach().float()
    print(f"  {name:24s} shape {tuple(t.shape)}  |x| {a.abs().max():.3e}  "
          f"nonfinite {bad}")
    return bad == 0


def main() -> int:
    import torch
    from config import warmup
    from dataset import CaptureDataset

    from jaeger_agent_omni.core.neural import hidden_size_of

    dev = "mps"
    cfg = warmup("google/gemma-4-E4B-it")
    cfg.whisper_id = "openai/whisper-medium.en"
    cfg.arch = "minicpm_delta"
    cfg.__post_init__()

    from train import real_setup
    enc, backbone, proj, dl, val_dl, ds, val_ds, wrapper = real_setup(
        cfg, "training/corpus_mixed", dev)
    print(f"backbone hidden {hidden_size_of(backbone)}  "
          f"dtype {next(backbone.parameters()).dtype}")

    from train import Trainer
    tr = Trainer(cfg, enc, backbone, proj, device=dev, wrapper=wrapper)

    it = iter(dl)
    for trial in range(12):
        batch = next(it)
        # EXACTLY what the training loop feeds (train.py:557): features are
        # cast to the trainer dtype -- bf16 -- before the encoder sees them.
        feats = batch["features"].to(dev, tr.dtype)
        labels = batch["labels"].to(dev)
        dur = batch.get("durations")
        print(f"\n--- batch {trial}  features {tuple(feats.shape)}  "
              f"dur {[round(d,1) for d in (dur or [])][:3]} ---")
        ok = check("mel features (bf16)", feats)
        import torch
        f32 = batch["features"].to(dev, torch.float32)
        print(f"  {'mel range':24s} fp32 [{f32.min():.2f}, {f32.max():.2f}]  "
              f"bf16 [{feats.float().min():.2f}, {feats.float().max():.2f}]")
        with torch.no_grad():
            e32 = tr.encoder.float()(f32).last_hidden_state
            tr.encoder.to(tr.dtype)
        check("encoder(fp32 mel)", e32)
        with torch.no_grad():
            e16 = tr.encoder(feats).last_hidden_state
        ok &= check("encoder(bf16 mel)", e16)

        import torch
        with torch.no_grad():
            prefix = tr.audio_prefix(feats, dur)
        ok &= check("projector+delta out", prefix)
        if prefix is not None:
            n = prefix.norm(dim=-1)
            print(f"  {'prefix token norms':24s} mean {n.mean():.2f}  "
                  f"min {n.min():.2f}  max {n.max():.2f}")

        with torch.no_grad():
            out, full = tr.student(feats, labels=labels, durations=dur)
        ok &= check("backbone logits", out.logits)
        if not ok:
            print("  ^^ FIRST NON-FINITE STAGE")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
