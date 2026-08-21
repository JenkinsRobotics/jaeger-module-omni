"""Synthesise a smoke-test corpus with the system TTS.

Real audio, real transcripts, zero download — enough to prove the training path
end to end before committing to a public corpus or a recording session.

    python training/make_corpus.py --out corpus/ --n 60

NOT training data for a real run. Synthetic speech is one narrow slice of
Whisper's output space; a projector trained only on it falls over on a
microphone. See the plan doc's "Data" section — this is a smoke test, tiers 1-4
are the real thing.

macOS only (uses `say`). On Linux, point --tts at espeak-ng or swap in a
downloaded corpus.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.capture import SAMPLE_RATE, CaptureWriter  # noqa: E402

# Short factual prompts: the answers are short and deterministic, so a tiny
# backbone produces stable targets and loss curves stay readable.
PROMPTS = [
    ("What is two plus two?", "Four."),
    ("What color is the sky?", "Blue."),
    ("How many days in a week?", "Seven."),
    ("What is the capital of France?", "Paris."),
    ("What is ten minus three?", "Seven."),
    ("How many legs does a spider have?", "Eight."),
    ("What is the largest planet?", "Jupiter."),
    ("What is five times five?", "Twenty five."),
    ("What season comes after summer?", "Autumn."),
    ("How many hours in a day?", "Twenty four."),
    ("What is the freezing point of water?", "Zero degrees celsius."),
    ("Who wrote Romeo and Juliet?", "Shakespeare."),
    ("What is the smallest prime number?", "Two."),
    ("How many continents are there?", "Seven."),
    ("What gas do plants absorb?", "Carbon dioxide."),
]

# Varying the voice is the only augmentation available for free here, and it
# is the difference between the projector learning speech and learning one
# speaker's timbre.
VOICES = ["Samantha", "Alex", "Daniel", "Karen", "Fred"]


def have_say() -> bool:
    return shutil.which("say") is not None


def synth(text: str, voice: str, path: Path) -> np.ndarray:
    subprocess.run(["say", "-v", voice, f"--data-format=LEI16@{SAMPLE_RATE}",
                    "-o", str(path), text], check=True, capture_output=True)
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        assert w.getframerate() == SAMPLE_RATE
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0


def build(out: Path, n: int, verbose: bool = True) -> int:
    writer = CaptureWriter(out)
    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "u.wav"
        for i in range(n):
            text, answer = PROMPTS[i % len(PROMPTS)]
            voice = VOICES[(i // len(PROMPTS)) % len(VOICES)]
            try:
                audio = synth(text, voice, scratch)
            except subprocess.CalledProcessError:
                if verbose:
                    print(f"  skip: voice {voice!r} unavailable")
                continue
            # confidence=1.0: the transcript is ground truth here, we wrote it.
            writer.write(audio, transcript=text, response=answer, confidence=1.0,
                         soul="mochi", session_id="synth",
                         meta={"voice": voice, "synthetic": True})
            written += 1
            if verbose and written % 10 == 0:
                print(f"  {written}/{n}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    if not have_say():
        print("`say` not found — macOS only. Use a downloaded corpus instead.")
        return 1

    out = Path(args.out)
    n = build(out, args.n)
    from jaeger_agent_omni.capture import CaptureReader
    r = CaptureReader(out)
    print(f"wrote {n} turns, {r.hours()*3600:.0f}s of audio to {out}")
    print("NOTE synthetic — smoke test only, not a real-run corpus")
    return 0


def demo():
    """Self-check. Skips gracefully where `say` does not exist."""
    if not have_say():
        print("make_corpus: `say` unavailable, skipped (macOS only)")
        return

    from jaeger_agent_omni.capture import CaptureReader

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "c"
        n = build(out, 4, verbose=False)
        assert n == 4, n

        turns = list(CaptureReader(out, min_confidence=0.8))
        assert len(turns) == 4, "synthetic turns must pass the confidence gate"
        assert all(t.confidence == 1.0 for t in turns), "ground truth, not scored"
        assert all(t.meta.get("synthetic") for t in turns), "must be labelled synthetic"
        assert all(0.3 < t.duration_s < 10 for t in turns), \
            [t.duration_s for t in turns]

        # Real audio, not silence — a corpus of zeros would train happily and
        # teach nothing.
        audio = CaptureReader(out).audio(turns[0])
        assert np.abs(audio).max() > 0.01, "synthesised audio is silent"

        # Distinct prompts produce distinct audio.
        a, b = (CaptureReader(out).audio(t) for t in turns[:2])
        assert turns[0].transcript != turns[1].transcript
        assert len(a) != len(b) or not np.array_equal(a, b)

    print(f"make_corpus: {n} synthetic turns, real audio, gate-passing OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
