"""Stage 1 — capture.

Log every legacy-stack turn as a training pair. Raw mic audio + the transcript
Whisper produced + the response Gemma produced. That is exactly the teacher
signal Stage 2 needs, and it costs nothing to collect today.

Format: one directory, a JSONL manifest, PCM16 wav per turn.

    turns.jsonl     one Turn per line
    audio/<id>.wav  16 kHz mono PCM16

    python -m jaeger_agent_omni.capture     # self-check, temp dir

ponytail: stdlib `wave` and `json`, no soundfile/pandas. The format is a
directory of files on purpose — it survives being copied to a training box.
"""

import json
import uuid
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

SAMPLE_RATE = 16000  # Whisper's rate. Resample upstream, not here.


@dataclass
class Turn:
    """One exchange. `transcript` and `response` are the teacher's, not targets.

    `confidence` is the ASR's own estimate (JaegerWhisperSTT's
    `TranscriptSegment.probability`, averaged). A wrong transcript produces a
    wrong teacher target and the projector learns to reproduce it faithfully —
    so this exists to be filtered on, not to be logged and ignored.

    Set `confidence=1.0` for human-labeled public corpora (LibriSpeech et al).
    Ground truth needs no gate, and using the same field for both means one
    filter expression works across every data tier.
    """

    turn_id: str
    audio_path: str
    transcript: str          # what Whisper heard  -> feeds the text teacher
    response: str            # what the LLM replied -> the answer to reproduce
    duration_s: float
    confidence: Optional[float] = None   # None = unscored, not "good"
    soul: Optional[str] = None
    session_id: Optional[str] = None
    meta: dict = field(default_factory=dict)


class CaptureWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "audio").mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "turns.jsonl"

    def write(self, audio: np.ndarray, transcript: str, response: str, **kw) -> Turn:
        """`audio` is float32 mono in [-1, 1] at SAMPLE_RATE."""
        if audio.ndim != 1:
            raise ValueError(f"expected mono, got shape {audio.shape}")

        turn_id = uuid.uuid4().hex[:12]
        rel = f"audio/{turn_id}.wav"
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")

        with wave.open(str(self.root / rel), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())

        turn = Turn(
            turn_id=turn_id,
            audio_path=rel,
            transcript=transcript,
            response=response,
            duration_s=len(audio) / SAMPLE_RATE,
            **kw,
        )
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        return turn


class CaptureReader:
    """Iterate captured turns.

    `min_confidence` gates the corpus. Turns with `confidence=None` are treated
    as failing any threshold — unscored is not the same as good, and defaulting
    the other way would quietly admit every unscored turn into training.
    """

    def __init__(self, root: str | Path, min_confidence: float | None = None):
        self.root = Path(root)
        self.min_confidence = min_confidence

    def __iter__(self) -> Iterator[Turn]:
        with open(self.root / "turns.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                turn = Turn(**json.loads(line))
                if self.min_confidence is not None:
                    if turn.confidence is None or turn.confidence < self.min_confidence:
                        continue
                yield turn

    def audio(self, turn: Turn) -> np.ndarray:
        with wave.open(str(self.root / turn.audio_path), "rb") as w:
            raw = w.readframes(w.getnframes())
            assert w.getframerate() == SAMPLE_RATE, w.getframerate()
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0

    def hours(self) -> float:
        return sum(t.duration_s for t in self) / 3600.0

    def stats(self) -> dict:
        """Corpus summary. `rejected` is what a threshold is costing you."""
        total = sum(1 for _ in CaptureReader(self.root))
        kept = sum(1 for _ in self)
        return {"total": total, "kept": kept, "rejected": total - kept,
                "hours": round(self.hours(), 4)}


def demo():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        w = CaptureWriter(tmp)
        audio = (np.sin(np.arange(SAMPLE_RATE * 2) * 0.05) * 0.3).astype(np.float32)
        written = w.write(
            audio,
            transcript="what is two plus two",
            response="Four.",
            confidence=0.94,
            soul="mochi",
            session_id="s1",
        )

        r = CaptureReader(tmp)
        turns = list(r)
        assert len(turns) == 1, turns
        got = turns[0]
        assert got.transcript == written.transcript
        assert got.response == "Four."
        assert got.soul == "mochi"
        assert abs(got.duration_s - 2.0) < 1e-6, got.duration_s

        back = r.audio(got)
        assert back.shape == audio.shape, (back.shape, audio.shape)
        # PCM16 round-trip: lossy, but only at quantisation scale.
        assert np.abs(back - audio).max() < 1e-4, np.abs(back - audio).max()
        assert abs(r.hours() - 2.0 / 3600.0) < 1e-9
        assert got.confidence == 0.94

        # Quality gate: a bad transcript is a bad teacher target.
        w.write(audio, transcript="wht s to pls to", response="Four.",
                confidence=0.31)
        w.write(audio, transcript="unscored", response="Four.")   # confidence=None

        assert len(list(CaptureReader(tmp))) == 3
        good = CaptureReader(tmp, min_confidence=0.8)
        assert [t.transcript for t in good] == ["what is two plus two"]

        # Unscored must NOT pass a threshold — silently admitting it would let
        # an unmeasured corpus into training looking like a measured one.
        assert all(t.confidence is not None for t in good)
        # A zero threshold still admits every *scored* turn — it only excludes
        # the unscored one. Confidence-gating and scored-ness are separate.
        assert len(list(CaptureReader(tmp, min_confidence=0.0))) == 2

        assert good.stats() == {"total": 3, "kept": 1, "rejected": 2,
                                "hours": round(2.0 / 3600.0, 4)}

    print("capture: round-trip OK (manifest, wav, float32 fidelity)")
    print("capture: confidence gate rejects bad and unscored transcripts")


if __name__ == "__main__":
    demo()
