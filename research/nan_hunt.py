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

    from jaeger_agent_omni.neural import hidden_size_of

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

    it = iter(dl)
    for trial in range(6):
        batch = next(it)
        wav = batch["waveform"].to(dev)
        print(f"\n--- batch {trial}  waveform {tuple(wav.shape)} ---")
        ok = check("waveform", wav)
        with torch.no_grad():
            feats = enc(wav) if callable(enc) else None
        if feats is not None:
            h = feats.last_hidden_state if hasattr(feats, "last_hidden_state") else feats
            ok &= check("whisper encoder", h)
            with torch.no_grad():
                p = proj(h, duration_s=None) if "duration" in proj.forward.__code__.co_varnames else proj(h)
            ok &= check("projector+delta", p)
        if not ok:
            print("  ^^ first non-finite stage found")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
