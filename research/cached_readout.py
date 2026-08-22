"""Read the trained adapter's output as words, using only cached features.

No whisper, no backbone, no GPU. The cache already holds the exact input to
the trained layer -- AvgPool5(ReLU(linear1(APM(audio)))) -- so the whole check
is one matmul plus a nearest-neighbour lookup in Gemma's embedding table:

    cached feature [T,4096] @ W.T + b  ->  [T,2560]  ->  nearest token rows

That makes it safe to run BESIDE a training job, which matters on a machine
where concurrent GPU work has produced four wrong measurements.

    python research/cached_readout.py --ckpt <adapter.npz> -n 4
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

CACHE = Path.home() / "models/gemma-audio-adapter/cache"
GEMMA = "google/gemma-4-E4B-it"


def main() -> int:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from fit_alignment import embedding_table

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=str(Path.home() /
                    "models/gemma-audio-adapter/adapter_jaeger.npz"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("-n", type=int, default=4)
    a = ap.parse_args()

    z = np.load(a.ckpt)
    W = torch.tensor(z["weight"])        # [2560, 4096]
    b = torch.tensor(z["bias"])
    # Mid-round saves carry fewer fields than end-of-round ones.
    meta = " ".join(f"{k} {float(z[k]):.4f}" if z[k].dtype.kind == "f"
                    else f"{k} {z[k]}"
                    for k in ("step", "validation_ce", "train_loss")
                    if k in z.files)
    print(f"adapter: {meta}")

    tok = AutoTokenizer.from_pretrained(GEMMA)
    tab = embedding_table(GEMMA)
    tab = tab * (tab.shape[1] ** 0.5)     # the scale the model receives
    tn = tab / tab.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    root = CACHE / a.split.replace(".", "_")
    rows = [json.loads(l) for l in
            (root / "manifest.jsonl").read_text().splitlines() if l.strip()][:a.n]

    for r in rows:
        # cached features are float16; the adapter is float32
        feat = torch.tensor(np.load(root / r["feature"])).float()   # [T, 4096]
        out = feat @ W.T + b                                   # [T, 2560]
        v = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ids = (tn @ v.T).argmax(dim=0)
        words = [tok.decode([int(i)]) for i in ids]
        # collapse CTC-style repeats so the readout is legible
        seen = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]
        print(f"\nTRUTH   {r['text'][:78]}")
        print(f"READOUT {''.join(seen)[:78]}")
        print(f"        {out.shape[0]} slots · norm {out.norm(dim=-1).mean():.1f} "
              f"(Gemma expects ~{tab.norm(dim=-1).mean():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
