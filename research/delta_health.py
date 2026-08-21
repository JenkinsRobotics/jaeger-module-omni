"""Is the trained delta alive, or collapsed? Check before trusting any WER.

A collapsed projector emits nearly the same vector for every input. With
`embed_norm` pinning the magnitude it still produces correctly-scaled output,
so nothing downstream complains -- the backbone simply receives a constant
where the audio should be, and predicts from context alone. Loss falls,
gradients vanish, and free-running accuracy stays at zero.

That pattern was visible in this run: loss 4.24 -> 2.85 while gradient norm
fell 467 -> 12 and `agreement` never left 0.00.

    cross-clip cosine    1.00 = every clip maps to the same vector (dead)
                         0.91 = MiniCPM's trained projector (reference)
                         0.93 = our delta at init

    python research/delta_health.py --ckpt training/runs/delta_e4b_stage3_warmup.pt
"""
from __future__ import annotations

import argparse, itertools, subprocess, sys, tempfile, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

SR = 16000
CLIPS = ["The capital of France is Paris.", "A spider has eight legs.",
         "Water freezes at zero degrees.", "Jupiter is the largest planet.",
         "There are seven days in a week.", "Romeo and Juliet, by Shakespeare.",
         "Two plus two equals four.", "Autumn comes after summer."]


def say(text, path, voice="Samantha"):
    import numpy as np
    subprocess.run(["say", "-v", voice, f"--data-format=LEI16@{SR}",
                    "-o", str(path), text], check=True, capture_output=True)
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def main() -> int:
    import numpy as np
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="training/runs/delta_e4b_stage3_warmup.pt")
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    a = ap.parse_args()

    from transformers import WhisperFeatureExtractor, WhisperModel

    from jaeger_agent_omni.neural import DeltaAdapter, MiniCPMDeltaPath
    from load_minicpm import load_front_end

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cog = ck.get("cognition_dim")
    print(f"checkpoint step {ck.get('step')}  cognition_dim {cog}  "
          f"embed_norm {ck.get('embed_norm'):.2f}")
    if cog != 2560:
        raise SystemExit(f"cognition_dim {cog} is not E4B (2560)")

    encoder, front = load_front_end()
    delta = DeltaAdapter(source_dim=front.cognition_dim, cognition_dim=cog,
                         embed_norm=float(ck["embed_norm"]))
    proj = MiniCPMDeltaPath(front, delta)
    proj.load_state_dict(ck["projector"])
    proj.eval()
    encoder.eval()

    fx = WhisperFeatureExtractor.from_pretrained(a.whisper)
    embeds = []
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.wav"
        for text in CLIPS:
            audio = say(text, p)
            feats = fx(audio, sampling_rate=SR, return_tensors="pt").input_features
            with torch.no_grad():
                h = encoder(feats).last_hidden_state
                out = proj(h.float(), duration_s=len(audio) / SR)
            embeds.append(out[0].mean(0).numpy())     # mean over tokens

    E = np.stack(embeds)
    U = E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)
    cos = [float(U[i] @ U[j]) for i, j in itertools.combinations(range(len(U)), 2)]
    mean_cos = float(np.mean(cos))

    print(f"\ncross-clip cosine   {mean_cos:.4f}   "
          f"(min {min(cos):.4f}  max {max(cos):.4f})")
    print(f"  reference: MiniCPM trained 0.9052 | ours at init 0.9317 | dead 1.0000")
    print(f"per-token norm      {np.linalg.norm(E, axis=-1).mean():.2f} "
          f"(embed_norm {ck['embed_norm']:.2f})")

    verdict = ("COLLAPSED — every clip maps to the same vector; the backbone "
               "sees a constant" if mean_cos > 0.995 else
               "SUSPECT — barely discriminating" if mean_cos > 0.98 else
               "ALIVE — clips are distinguishable")
    print(f"\nverdict             {verdict}")
    return 0


def demo() -> None:
    import numpy as np
    ident = np.ones((4, 8))
    U = ident / np.linalg.norm(ident, axis=-1, keepdims=True)
    c = [float(U[i] @ U[j]) for i, j in itertools.combinations(range(4), 2)]
    assert abs(np.mean(c) - 1.0) < 1e-9, "identical inputs must score 1.0"
    rng = np.random.default_rng(0)
    R = rng.normal(size=(8, 64))
    U = R / np.linalg.norm(R, axis=-1, keepdims=True)
    c = [float(U[i] @ U[j]) for i, j in itertools.combinations(range(8), 2)]
    assert abs(np.mean(c)) < 0.3, np.mean(c)
    print("delta_health: collapse metric OK (identical=1.0, random~0)")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
