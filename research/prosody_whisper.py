"""Does WHISPER's encoder carry prosody, or is it transcription-shaped too?

The narrative probe (ask the model "how does the speaker sound?") returned the
same sentence for four very different takes, but that cannot distinguish

    the TOWER discarded prosody          from        the LLM ignored it

so this measures the embeddings directly, before the LLM sees anything.

Method: one sentence, four deliveries (rate, emotion, speaker). If the tower
encodes delivery, the four embeddings differ from each other. Compare that
spread against the spread between four DIFFERENT sentences, which is the
tower's known-good discrimination -- it separates words, we know that, it
scores 12.2% WER. The ratio is the answer:

    prosody spread ~= content spread   -> delivery IS encoded
    prosody spread << content spread   -> the tower is transcription-shaped

    python research/prosody_whisper.py
"""

from __future__ import annotations

import itertools, subprocess, sys, tempfile, wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
TEMPORAL = "--temporal" in sys.argv   # keep time structure, not just the mean
# same words, four deliveries: rate, emotion-by-rate, and a different speaker
PROSODY = [("neutral",      ["-v", "Samantha", "-r", "175"]),
           ("slow",         ["-v", "Samantha", "-r", "110"]),
           ("fast",         ["-v", "Samantha", "-r", "260"]),
           ("other_voice",  ["-v", "Alex",     "-r", "175"])]
SENTENCE = "I am doing perfectly fine today, thank you for asking me."
# four different sentences, matched for length, as the discrimination control
CONTENT = ["I am doing perfectly fine today, thank you for asking me.",
           "The brown dog ran quickly across the empty field this morning.",
           "Please close the heavy wooden door before you leave the room.",
           "It has been raining steadily since early yesterday afternoon here."]


def synth(text: str, flags: list, path: Path) -> np.ndarray:
    subprocess.run(["say", *flags, f"--data-format=LEI16@{SR}", "-o", str(path),
                    text], check=True, capture_output=True)
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def spread(vecs: list) -> float:
    """Mean pairwise cosine DISTANCE. Higher = more discriminated."""
    v = [x / (np.linalg.norm(x) + 1e-9) for x in vecs]
    return float(np.mean([1.0 - float(a @ b) for a, b in itertools.combinations(v, 2)]))


def main() -> int:
    import json

    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    wid = "openai/whisper-medium.en"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    fx = WhisperFeatureExtractor.from_pretrained(wid)
    enc = WhisperModel.from_pretrained(wid).encoder.to(dev).eval()

    def embed(audio: np.ndarray) -> np.ndarray:
        """Whisper ENCODER output, mean-pooled. The decoder never runs -- this
        is exactly the representation a delta adapter would bridge into Gemma.

        Whisper pads every input to 30 s / 1500 frames regardless of the clip's
        real length, so pool only over the REAL frames (50 per second).
        Averaging in 28 s of encoded silence would wash out precisely the
        differences this is trying to measure.
        """
        feats = fx(audio, sampling_rate=SR,
                   return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            h = enc(feats).last_hidden_state          # [1, 1500, 1024]
        real = max(1, min(h.shape[1], int(round(len(audio) / SR * 50))))
        x = h[0, :real].float()
        if TEMPORAL:
            # Prosody IS temporal -- pitch contour, rhythm, stress placement.
            # A mean over time averages exactly that away, so also carry the
            # per-dimension spread and the frame-to-frame delta magnitude,
            # which is where rate and emphasis live.
            d = (x[1:] - x[:-1]).abs().mean(0) if x.shape[0] > 1 else x.mean(0) * 0
            return torch.cat([x.mean(0), x.std(0), d]).cpu().numpy()
        return x.mean(0).cpu().numpy()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.wav"
        pros = [embed(synth(SENTENCE, f, p)) for _, f in PROSODY]
        cont = [embed(synth(s, ["-v", "Samantha", "-r", "175"], p)) for s in CONTENT]

    sp, sc = spread(pros), spread(cont)
    ratio = sp / sc if sc else float("nan")
    print(f"\nsame words, 4 deliveries   spread {sp:.4f}")
    print(f"4 different sentences      spread {sc:.4f}   (known-good control)")
    print(f"ratio                      {ratio:.3f}")
    verdict = ("delivery IS encoded" if ratio > 0.5 else
               "partially encoded" if ratio > 0.2 else
               "TRANSCRIPTION-SHAPED: delivery is largely discarded")
    print(f"verdict                    {verdict}")
    return 0


def demo() -> None:
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert abs(spread([a, a]) - 0.0) < 1e-6          # identical -> no spread
    assert abs(spread([a, b]) - 1.0) < 1e-6          # orthogonal -> full spread
    print("prosody_carry: spread metric OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())
