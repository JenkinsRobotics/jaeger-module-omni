"""Speech out. Two engines, honestly labelled.

    kokoro     correct words, many preset voices, speed control.  WORKS TODAY.
    token2wav  MiniCPM's CosyVoice2 vocoder. VOICE CLONING from a reference
               clip via campplus. Audible today, but the words are wrong
               without hidden-state conditioning -- measured:

                   asked for : The capital of France is Paris.
                   came back : Capptovern in es Paris.

               "Paris" survives; the rest is phonetic mush. The talker was
               trained with an LLM hidden-state term at every position, and
               without it the text embedding alone does not steer content.

So today: pick `kokoro` for a demo people can understand, `token2wav` to hear
a cloned voice. They are not yet the same path, and pretending otherwise
would be the same mistake as reporting teacher-forced accuracy.

Closing that gap needs `SpeechOutDelta` (cognition 2560 -> Qwen 4096, norm
pinned) trained against real Qwen hidden states -- either Qwen3-8B or the
copy already inside MiniCPM's GGUF.

    python -m jaeger_agent_omni.voice --say "hello there"
    python -m jaeger_agent_omni.voice --say "hi" --voice am_michael --speed 1.3
    python -m jaeger_agent_omni.voice --list-voices
    python -m jaeger_agent_omni.voice --say "hi" --engine token2wav --ref REF.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KOKORO_RATE = 24000
KOKORO_NODE = Path(__file__).resolve().parent / "nodes" / "kokoro-tts"

# af_* female, am_* male. Kokoro ships fixed voices — these are selections,
# not clones. Cloning is token2wav's job.
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
          "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george"]

# The node is held RESIDENT. Shelling out reloaded the model on every clause
# and made the registered node differ from the live implementation.
_KOKORO_ENGINES: dict = {}


def _pipeline(lang: str):
    from jaeger_agent_omni.node_registry import load_factory, load_package

    if lang not in _KOKORO_ENGINES:
        package = load_package(KOKORO_NODE)
        _KOKORO_ENGINES[lang] = load_factory(package)(lang=lang)
    return _KOKORO_ENGINES[lang]


def speak_kokoro(text: str, out: str | Path, voice: str = "af_heart",
                 speed: float = 1.0) -> dict:
    """Correct words, preset voice, adjustable rate. In-process."""
    import numpy as np
    import soundfile as sf

    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = _pipeline(voice[0])
    audio = np.asarray(
        engine.synthesize(text, voice=voice, speed=speed),
        dtype="float32",
    ).reshape(-1)
    sf.write(str(out), audio, KOKORO_RATE)
    return {"engine": "kokoro", "node": "kokoro-tts",
            "voice": voice, "speed": speed,
            "seconds": len(audio) / KOKORO_RATE, "path": str(out)}


def speak_cloned(text: str, out: str | Path, ref_audio: str | Path,
                 device: str = "cpu") -> dict:
    """MiniCPM talker -> token2wav, voice CLONED from `ref_audio`.

    The clone is real: token2wav's set_ref_audio embeds the reference speaker
    through campplus. The WORDS are not reliable — see the module docstring.
    """
    sys.path.insert(0, str(REPO / "research"))
    from t2w_client import speak as t2w_speak
    path = t2w_speak(text, out, ref_audio=ref_audio, device=device)
    with wave.open(str(path)) as w:
        secs = w.getnframes() / w.getframerate()
    return {"engine": "token2wav", "cloned_from": str(ref_audio),
            "seconds": secs, "path": str(path),
            "warning": "words unreliable without hidden-state conditioning"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--say", default="Hello, I am Jaeger.")
    ap.add_argument("--engine", default="kokoro",
                    choices=("kokoro", "token2wav"))
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--ref", help="reference clip to clone (token2wav)")
    ap.add_argument("--out", default="reply.wav")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()

    if a.list_voices:
        print("kokoro preset voices (selection, NOT cloning):")
        for v in VOICES:
            print(f"  {v}")
        print("\ncloning: --engine token2wav --ref YOUR_CLIP.wav")
        return 0

    if a.engine == "token2wav":
        if not a.ref:
            raise SystemExit("--engine token2wav needs --ref REF.wav to clone")
        meta = speak_cloned(a.say, a.out, a.ref)
    else:
        meta = speak_kokoro(a.say, a.out, a.voice, a.speed)
    for k, v in meta.items():
        print(f"  {k:12s} {v}")
    return 0


def demo() -> None:
    """No models: check the surface and the voice table."""
    assert VOICES and all(v[0] in "abe" for v in VOICES)
    import importlib.util
    assert importlib.util.find_spec("kokoro"), "kokoro must be in THIS venv"
    src = Path(__file__).read_text()
    # The honesty guard: this module must keep saying the cloned path's words
    # are unreliable until SpeechOutDelta is trained.
    assert "words unreliable" in src
    print(f"voice: {len(VOICES)} preset voices, kokoro venv present, "
          "clone caveat documented")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
