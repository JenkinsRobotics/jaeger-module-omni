"""Does the native Gemma audio tower carry prosody into the LLM at all?

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

    python research/prosody_carry.py
"""

from __future__ import annotations

import itertools, subprocess, sys, tempfile, wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
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

    import mlx.core as mx
    from mlx_vlm.utils import load

    from jaeger_agent_omni.core.runtime import GemmaAudioPrefiller

    model_path = "/Users/jonathanjenkins/models/gemma-4-e4b-it-mlx-full"
    m, proc = load(model_path)
    pre = GemmaAudioPrefiller(m, proc)
    # the id lives in the model config, not on the prefiller
    audio_id = json.load(open(Path(model_path) / "config.json"))["audio_token_id"]

    def embed(audio: np.ndarray) -> np.ndarray:
        """Tower output, mean-pooled over time. No LLM involved."""
        mel, valid = pre.mels(mx.array(audio))
        n = pre.n_tokens_for(mel, ~valid)
        ids = mx.array([[audio_id] * n])
        out = m.get_input_embeddings(input_ids=ids, audio_features=mel,
                                     input_features_mask=valid)
        # same unwrap as StreamingSession._embed_audio_chunk: newer mlx_vlm
        # returns an InputEmbeddingsFeatures object, older ones a tuple
        e = out[0] if isinstance(out, tuple) else getattr(out, "inputs_embeds", out)
        # bfloat16 has no numpy buffer format -- cast before crossing over
        return np.array(mx.mean(e, axis=1).astype(mx.float32)).reshape(-1)

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
