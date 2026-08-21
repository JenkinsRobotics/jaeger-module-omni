"""Closed-form init for both deltas. No GPU, no training loop.

Adapted from the parallel agent's fit_alignment.py, which is very likely why
their adapter moved held-out CE 5.87 -> 1.14 in 40 steps while ours sat flat:
they started from a linear map that already roughly works, we started from
random.

The pairing trick is the whole idea. The same English word, embedded by two
different models' token tables, gives a paired sample. Thousands of words give
an over-determined system, and ridge regression solves it in closed form:

    W_fwd: Qwen 4096 -> Gemma 2560     initialises the INPUT delta
    W_rev: Gemma 2560 -> Qwen 4096     initialises SpeechOutDelta

Those are exactly our two adapters. MiniCPM's frozen projector emits Qwen-4096
space, so W_fwd is the bridge into Gemma; the talker consumes Qwen-4096, so
W_rev is the bridge out.

A word that tokenises to several subwords has its rows averaged — crude, but
the target is an initialisation, not a final answer.

    python training/fit_alignment.py --out training/runs/alignment.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QWEN_ID = "Qwen/Qwen2.5-0.5B"          # same tokenizer family as MiniCPM's backbone
GEMMA_ID = "google/gemma-4-E4B-it"
WORDS = "/usr/share/dict/words"


def word_list(n: int = 16000, min_len: int = 2, max_len: int = 9) -> list:
    import random
    p = Path(WORDS)
    if not p.exists():                       # fall back to the corpus vocabulary
        return []
    seen = [w.strip() for w in p.read_text().split("\n")]
    seen = [w for w in seen if min_len <= len(w) <= max_len and w.islower()
            and w.isalpha()]
    random.Random(0).shuffle(seen)
    return seen[:n]


def embed_words(model_id: str, words: list, prefix: str = " "):
    """Mean of the token-embedding rows a word maps to, per word."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    table = m.get_input_embeddings().weight.detach()
    del m

    rows, keep = [], []
    for i, w in enumerate(words):
        ids = tok(prefix + w, add_special_tokens=False).input_ids
        if not ids:
            continue
        rows.append(table[ids].mean(0).numpy())
        keep.append(i)
    return np.stack(rows), keep


def ridge(X, Y, lam: float = 1.0):
    """Solve W minimising ||XW - Y||^2 + lam||W||^2. X:[N,a] Y:[N,b] -> [a,b]."""
    import numpy as np
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=X.dtype)
    return np.linalg.solve(A, X.T @ Y)


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="training/runs/alignment.npz")
    ap.add_argument("--words", type=int, default=16000)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()

    ws = word_list(a.words)
    if not ws:
        raise SystemExit(f"no word list at {WORDS}")
    print(f"{len(ws)} candidate words", flush=True)

    Xq, keep_q = embed_words(QWEN_ID, ws)
    Xg, keep_g = embed_words(GEMMA_ID, [ws[i] for i in keep_q])
    Xq = Xq[keep_g]                                  # align the two keep-lists
    print(f"paired {Xq.shape[0]} words: qwen {Xq.shape[1]} <-> gemma {Xg.shape[1]}")

    # Split before fitting: a ridge map that only fits its own training words
    # is a memorisation, and would make a bad init look like a good one.
    n = Xq.shape[0]
    cut = int(0.9 * n)
    W_fwd = ridge(Xq[:cut], Xg[:cut], a.lam)
    W_rev = ridge(Xg[:cut], Xq[:cut], a.lam)

    def r2(X, Y, W):
        P = X @ W
        return 1.0 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()

    stats = {
        "fwd_train_r2": float(r2(Xq[:cut], Xg[:cut], W_fwd)),
        "fwd_heldout_r2": float(r2(Xq[cut:], Xg[cut:], W_fwd)),
        "rev_train_r2": float(r2(Xg[:cut], Xq[:cut], W_rev)),
        "rev_heldout_r2": float(r2(Xg[cut:], Xq[cut:], W_rev)),
        "n_pairs": int(n), "lam": a.lam,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, W_fwd=W_fwd, W_rev=W_rev, **{k: np.array(v)
                                                 for k, v in stats.items()})
    print(json.dumps(stats, indent=1))
    print(f"\nW_fwd {W_fwd.shape} (qwen->gemma, input delta init)")
    print(f"W_rev {W_rev.shape} (gemma->qwen, SpeechOutDelta init)")
    print(f"wrote {a.out}")
    return 0


def demo() -> None:
    """Ridge must recover a known linear map from noisy samples."""
    import numpy as np

    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 5))
    X = rng.normal(size=(4000, 12))
    Y = X @ W + rng.normal(scale=0.01, size=(4000, 5))
    W_hat = ridge(X, Y, lam=1e-3)
    err = np.abs(W - W_hat).max()
    assert err < 0.05, err
    # held-out prediction, not just training fit
    Xh = rng.normal(size=(500, 12))
    pred, true = Xh @ W_hat, Xh @ W
    r2 = 1 - ((true - pred) ** 2).sum() / ((true - true.mean(0)) ** 2).sum()
    assert r2 > 0.99, r2
    print(f"fit_alignment: ridge recovers a known map (max err {err:.4f}, "
          f"held-out r2 {r2:.4f})")


if __name__ == "__main__":
    import sys
    raise SystemExit(demo() if "--demo" in sys.argv else main())
