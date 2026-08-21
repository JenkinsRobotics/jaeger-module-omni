"""Torch delta checkpoint -> MLX weights, so the runtime stays one framework.

Training is PyTorch; StreamingSession is MLX. Crossing that boundary per audio
chunk would mean both frameworks resident, which is the 9.8 GB two-stack cost
we measured for the sequential pipeline and are trying to escape.

The delta is one Linear plus a scalar, so it ports exactly:

    delta.net.weight   (cognition_dim, 4096)
    delta.net.bias     (cognition_dim,)
    delta.embed_norm   scalar -- the output-scale pin

STILL TORCH after this: whisper's encoder and MiniCPM's 21M projector, both
frozen. Porting those needs mlx-whisper (not installed) or a hand conversion.
Until then the delta front end runs torch up to the projector and hands MLX a
finished embedding — correct, but it pays the two-framework memory cost.

    python research/delta_to_mlx.py --ckpt training/runs/delta_e4b.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(ckpt_path: Path, out_dir: Path) -> dict:
    import numpy as np
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    proj = ck["projector"]
    cog = ck.get("cognition_dim")
    if cog is None:
        raise SystemExit("checkpoint has no cognition_dim — refusing to guess")
    # A projector is valid ONLY for the backbone it trained against. E4B is
    # 2560; a 576 here means the SmolLM2 stand-in and must not load.
    if cog != 2560:
        raise SystemExit(
            f"cognition_dim={cog}, not 2560. This adapter was trained against "
            f"a different backbone and cannot drive Gemma E4B.")

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    for k, v in proj.items():
        if not hasattr(v, "detach"):
            continue
        a = v.detach().float().numpy()
        arrays[k] = a
        meta[k] = list(a.shape)
    np.savez(out_dir / "delta.npz", **arrays)

    card = {
        "cognition_dim": cog,
        "embed_norm": float(ck.get("embed_norm", 0.0)),
        "modality": ck.get("modality", "audio"),
        "step": ck.get("step"),
        "backbone_id": (ck.get("config") or {}).get("backbone_id")
                       if isinstance(ck.get("config"), dict)
                       else getattr(ck.get("config"), "backbone_id", None),
        "shapes": meta,
        "note": "whisper encoder + MiniCPM projector are still torch; "
                "this converts the trained delta only",
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(card, indent=1))
    return card


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="training/runs/delta_e4b.pt")
    ap.add_argument("--out", default="artifacts/audio-delta__whisper-medium__gemma-4-e4b")
    a = ap.parse_args()
    p = Path(a.ckpt)
    if not p.exists():
        raise SystemExit(f"no checkpoint at {p} — training has not saved yet")
    card = convert(p, Path(a.out))
    print(json.dumps(card, indent=1))
    print(f"\nwrote {a.out}/delta.npz")
    return 0


def demo() -> None:
    """Round-trip a synthetic checkpoint. No models, no GPU."""
    import tempfile

    import numpy as np
    import torch

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ck = {"projector": {"delta.net.weight": torch.randn(2560, 4096),
                            "delta.net.bias": torch.zeros(2560),
                            "delta.embed_norm": torch.tensor(1.21)},
              "cognition_dim": 2560, "embed_norm": 1.21, "step": 600,
              "modality": "audio", "config": {"backbone_id": "google/gemma-4-E4B-it"}}
        torch.save(ck, td / "c.pt")
        card = convert(td / "c.pt", td / "out")
        assert card["cognition_dim"] == 2560
        z = np.load(td / "out" / "delta.npz")
        assert z["delta.net.weight"].shape == (2560, 4096)
        assert abs(card["embed_norm"] - 1.21) < 1e-6

        # a wrong-backbone adapter must be REFUSED, not silently loaded --
        # 576 is the SmolLM2 stand-in and would produce confident nonsense
        ck["cognition_dim"] = 576
        torch.save(ck, td / "bad.pt")
        try:
            convert(td / "bad.pt", td / "bad")
        except SystemExit:
            pass
        else:
            raise AssertionError("a 576-wide adapter must not load for E4B")
    print("delta_to_mlx: round-trip OK, wrong-backbone adapter refused")


if __name__ == "__main__":
    import sys
    raise SystemExit(demo() if "--demo" in sys.argv else main())
