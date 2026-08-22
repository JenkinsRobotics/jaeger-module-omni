"""WER over a whole held-out corpus, not one clip.

Every adapter number in this project so far came from a single 27 s JFK
recording. n=1 cannot separate 13% from 21%, and it silently rewards
overfitting to one speaker, one recording chain, one era of audio.

This runs the cached validation set -- real held-out LibriSpeech, many
speakers -- through the adapter and the frozen backbone, and reports WER over
all of it. Cached features mean no whisper in the loop, so it is fast enough
to run after every training round.

    python research/wer_corpus.py --adapter <npz> [-n 64]
"""
from __future__ import annotations

import argparse, json, math, re, sys
from pathlib import Path

CACHE = Path.home() / "models/gemma-audio-adapter/cache"
SRC = (Path.home() / "GITHUB/VoiceLLM-Playground/Omni Model/gemma-backbone"
       / "train_audio_adapter.py")


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", t.lower())


def edits(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein on words."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def main() -> int:
    import importlib.util

    import mlx.core as mx
    import numpy as np
    from mlx_lm.generate import generate_step

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--words-per-second", type=float, default=2.8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("src", SRC)
    src = importlib.util.module_from_spec(spec); spec.loader.exec_module(src)
    model, tok = src._load_mlx_backbone(src.GEMMA_MLX, None)
    prefix_ids, suffix_ids, _ = src._render_parts(tok)

    z = np.load(a.adapter)
    W = mx.array(z["weight"].T.astype(np.float32))
    b = mx.array(z["bias"].astype(np.float32))

    root = CACHE / a.split.replace(".", "_")
    rows = [json.loads(l) for l in
            (root / "manifest.jsonl").read_text().splitlines() if l.strip()][:a.n]

    tot_ref, tot_edit, rows_out = 0, 0, []
    for i, r in enumerate(rows):
        feat = mx.array(np.load(root / r["feature"]).astype(np.float32))
        ids = mx.array(prefix_ids + [0] * feat.shape[0] + suffix_ids,
                       dtype=mx.int32)
        te = model.language_model.model.embed_tokens(ids[None, :])[0]
        proj = (feat @ W + b) / math.sqrt(2560)
        s = len(prefix_ids)
        emb = mx.concatenate([te[:s], proj.astype(te.dtype),
                              te[s + feat.shape[0]:]], axis=0)
        budget = max(8, int(r["seconds"] * a.words_per_second * 1.6))
        gen = []
        for tid, _ in generate_step(ids, model, input_embeddings=emb,
                                    max_tokens=budget):
            if tid in (1, 106):
                break
            gen.append(tid)
        hyp = words(tok.decode(gen, skip_special_tokens=True))
        ref = words(r["text"])
        e = edits(ref, hyp)
        tot_ref += len(ref); tot_edit += e
        rows_out.append({"id": r["id"], "wer": e / max(1, len(ref)),
                         "ref": len(ref), "hyp": len(hyp)})
        if i < 3:
            print(f"  ref: {' '.join(ref[:14])}")
            print(f"  hyp: {' '.join(hyp[:14])}\n")
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(rows)}  running WER "
                  f"{100*tot_edit/max(1,tot_ref):.2f}%", flush=True)

    wer = tot_edit / max(1, tot_ref)
    print(f"\nCORPUS WER  {wer*100:.2f}%   over {len(rows)} clips, "
          f"{tot_ref} reference words")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"adapter": a.adapter, "split": a.split, "n": len(rows),
             "wer": wer, "reference_words": tot_ref, "rows": rows_out}, indent=1))
        print(f"wrote {a.out}")
    return 0


def demo() -> None:
    assert edits(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edits(["a", "b"], ["a", "x"]) == 1
    assert edits(["a"], ["a", "b", "c"]) == 2
    assert words("Hello, THERE! it's") == ["hello", "there", "it's"]
    print("wer_corpus: edit distance and tokenisation OK")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
