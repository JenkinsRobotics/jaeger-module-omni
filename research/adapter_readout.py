"""What is the adapter actually handing Gemma? Read it back as words.

The adapter does not transcribe. It converts whisper's output into the FORMAT
Gemma expects at its input — the same space Gemma's own token embeddings live
in. So its output can be checked without running Gemma at all:

    known speech -> whisper -> adapter -> vectors
                                          |
                       nearest row in Gemma's embedding table
                                          |
                                        words

If the adapter is doing its job, saying "the capital of France is Paris"
produces vectors sitting near those words' embeddings. If it is emitting
something Gemma cannot read, the nearest words are noise — and that is visible
here in seconds instead of after an hour of training through a 4B backbone.

This also measures the two things that decide whether Gemma can read it at all:

    norm       Gemma receives embeddings scaled by sqrt(hidden) — 61.75, not
               the raw table's 1.20. Wrong scale is unreadable regardless of
               direction.
    cosine     to the transcript's own embeddings. Chance is ~0 in 2560-d.

    python research/adapter_readout.py --ckpt <adapter.pt>
    python research/adapter_readout.py --text-only   # what a PERFECT adapter
                                                     # would score
"""

from __future__ import annotations

import argparse, subprocess, sys, tempfile, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

SR = 16000
GEMMA = "google/gemma-4-E4B-it"
CLIPS = ["The capital of France is Paris.",
         "A spider has eight legs.",
         "Water freezes at zero degrees."]


def say(text: str, path: Path, voice: str = "Samantha"):
    import numpy as np
    subprocess.run(["say", "-v", voice, f"--data-format=LEI16@{SR}",
                    "-o", str(path), text], check=True, capture_output=True)
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def gemma_embeddings():
    """The table Gemma reads, at the scale it actually receives."""
    import torch
    from transformers import AutoTokenizer

    from fit_alignment import embedding_table          # reads the shard directly

    tok = AutoTokenizer.from_pretrained(GEMMA)
    table = embedding_table(GEMMA)                     # raw rows
    scale = table.shape[1] ** 0.5                      # Gemma multiplies by this
    return tok, table * scale, float(scale)


def readout(vectors, table, tok, k: int = 1):
    """Nearest table row per vector -> the word Gemma would see it as."""
    import torch
    v = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t = table / table.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    words = []
    for i in range(v.shape[0]):
        sim = t @ v[i]
        idx = int(torch.argmax(sim))
        words.append(tok.decode([idx]).strip() or "·")
    return words


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    ap.add_argument("--text-only", action="store_true",
                    help="upper bound: embed the transcript itself")
    a = ap.parse_args()

    tok, table, scale = gemma_embeddings()
    print(f"Gemma table {tuple(table.shape)}  embed_scale {scale:.2f}  "
          f"mean row norm {table.norm(dim=-1).mean():.2f}\n")

    if a.text_only:
        # What a PERFECT adapter scores. Without this the numbers below have
        # no ceiling to be read against.
        for text in CLIPS:
            ids = tok(text, add_special_tokens=False).input_ids
            vecs = table[ids]
            print(f"{text!r}")
            print(f"  readout   {' '.join(readout(vecs, table, tok))}")
            print(f"  norm      {vecs.norm(dim=-1).mean():.2f}")
        return 0

    if not a.ckpt:
        raise SystemExit("--ckpt <adapter.pt>, or --text-only for the ceiling")

    from transformers import WhisperFeatureExtractor, WhisperModel

    from jaeger_agent_omni.core.neural import DeltaAdapter, MiniCPMDeltaPath
    from load_minicpm import load_front_end

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    encoder, front = load_front_end()
    delta = DeltaAdapter(source_dim=front.cognition_dim,
                         cognition_dim=ck["cognition_dim"],
                         embed_norm=float(ck["embed_norm"]))
    proj = MiniCPMDeltaPath(front, delta)
    proj.load_state_dict(ck["projector"])
    proj.eval(); encoder.eval()

    fx = WhisperFeatureExtractor.from_pretrained(a.whisper)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.wav"
        for text in CLIPS:
            audio = say(text, p)
            feats = fx(audio, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                h = encoder(feats).last_hidden_state
                out = proj(h.float(), duration_s=len(audio) / SR)[0]
            ids = tok(text, add_special_tokens=False).input_ids
            target = table[ids]

            words = readout(out, table, tok)
            # sequence-level: does the adapter land where the transcript lives?
            cos = float(torch.nn.functional.cosine_similarity(
                out.mean(0, keepdim=True), target.mean(0, keepdim=True)))
            print(f"{text!r}")
            print(f"  adapter emits {out.shape[0]} vectors, "
                  f"norm {out.norm(dim=-1).mean():.2f} "
                  f"(Gemma expects ~{table.norm(dim=-1).mean():.2f})")
            print(f"  readout   {' '.join(words[:14])}")
            print(f"  cosine to transcript embedding: {cos:+.4f}  "
                  f"(chance ~0.00)")
    return 0


def demo() -> None:
    """The readout must recover words from exact embeddings."""
    import torch

    class Tok:
        def decode(self, ids): return f"w{ids[0]}"
    table = torch.randn(50, 16)
    picked = torch.stack([table[3], table[41], table[7]])
    got = readout(picked, table, Tok())
    assert got == ["w3", "w41", "w7"], got
    # scale must not matter to the readout — it is cosine
    assert readout(picked * 9.0, table, Tok()) == got
    print("adapter_readout: nearest-row decode OK, scale-invariant")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
