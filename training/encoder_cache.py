"""Precompute frozen-encoder outputs once, reuse them every epoch.

The encoder is frozen, so its output for a given clip is deterministic and
identical on every pass. Recomputing it each step is the dominant cost of a
Stage 3 run and buys nothing.

It is worse than it looks: Whisper pads every input to 30 s / 3000 mel frames
regardless of the clip's real duration. A 3 s utterance costs the same encoder
forward as a 30 s one. Trimming in the projector saves *cognition tokens*, not
encoder compute — that saving only comes from not running the encoder again.

    python training/encoder_cache.py --corpus corpus/ --whisper <id>

Storage is the trade, and trimming decides how bad it is. Whisper emits 1500
frames for every clip, but a 5-second utterance only has 250 real ones — the
rest is encoded padding the projector discards anyway. Caching trimmed cuts
storage ~6x on conversational audio:

    untrimmed   [1500, 384] fp16 = 1.15 MB/clip   ~2.9 GB for 5 h
    trimmed     [ 250, 384] fp16 = 0.19 MB/clip   ~0.5 GB for 5 h

Cached states are therefore ALREADY TRIMMED. A consumer must not trim again —
pass `duration_s=None` to the projector for cached input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.core.capture import SAMPLE_RATE, CaptureReader  # noqa: E402

CACHE_DIRNAME = "encoder_cache"
FRAME_RATE = 50          # whisper encoder frames per second of audio


def cache_path(corpus: str | Path, whisper_id: str) -> Path:
    """One directory per encoder. Mixing encoders in one cache would serve a
    projector states from a model it was not trained against."""
    return Path(corpus) / CACHE_DIRNAME / whisper_id.replace("/", "__")


class EncoderCacheReader:
    """Loads precomputed states, falling back to None when absent.

    Returned states are already trimmed to the clip's real duration. Do not
    trim again — the projector must be called with `duration_s=None`.
    """

    def __init__(self, corpus, whisper_id: str):
        self.dir = cache_path(corpus, whisper_id)
        self.available = self.dir.is_dir()

    def get(self, turn_id: str) -> torch.Tensor | None:
        if not self.available:
            return None
        f = self.dir / f"{turn_id}.pt"
        return torch.load(f, weights_only=True) if f.exists() else None

    def __len__(self) -> int:
        return len(list(self.dir.glob("*.pt"))) if self.available else 0


def build(corpus, whisper_id: str, device: str = "cpu",
          dtype=torch.float16, verbose: bool = True) -> int:
    from transformers import WhisperModel, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(whisper_id)
    encoder = WhisperModel.from_pretrained(whisper_id, dtype=dtype)
    encoder = encoder.encoder.to(device).eval()

    out_dir = cache_path(corpus, whisper_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = CaptureReader(corpus)

    written = 0
    for turn in reader:
        target = out_dir / f"{turn.turn_id}.pt"
        if target.exists():
            continue
        audio = reader.audio(turn)
        feats = processor(audio, sampling_rate=SAMPLE_RATE,
                          return_tensors="pt").input_features.to(device, dtype)
        with torch.no_grad():
            states = encoder(feats).last_hidden_state[0]
        # Trim the encoded padding before storing. 50 frames per second of real
        # audio; everything past that is encoded silence the projector drops.
        n_valid = min(states.shape[0],
                      max(1, int(round(turn.duration_s * FRAME_RATE))))
        states = states[:n_valid].to(torch.float16).cpu()
        # Same write-then-rename discipline as checkpoints: a half-written
        # tensor file would be loaded as truth on the next run.
        tmp = target.with_suffix(".pt.tmp")
        torch.save(states, tmp)
        tmp.replace(target)
        written += 1
        if verbose and written % 25 == 0:
            print(f"  {written} cached  ({states.shape} {states.dtype})")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--whisper", default="openai/whisper-medium")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    n = build(args.corpus, args.whisper, device=device)
    d = cache_path(args.corpus, args.whisper)
    size = sum(f.stat().st_size for f in d.glob("*.pt")) / 1e6
    print(f"cached {n} new ({len(list(d.glob('*.pt')))} total, {size:.0f} MB) in {d}")
    return 0


def demo():
    import tempfile

    from jaeger_agent_omni.core.capture import CaptureWriter

    with tempfile.TemporaryDirectory() as tmp:
        w = CaptureWriter(tmp)
        tone = np.sin(np.arange(SAMPLE_RATE) * 0.02).astype(np.float32)
        turn = w.write(tone, transcript="hello", response="hi", confidence=0.9)

        # Separate directory per encoder: serving a projector states from a
        # different model than it was trained against is silent corruption.
        a = cache_path(tmp, "openai/whisper-tiny")
        b = cache_path(tmp, "openai/whisper-medium")
        assert a != b
        assert "/" not in a.name, a.name        # id must not create subdirs

        r = EncoderCacheReader(tmp, "openai/whisper-tiny")
        assert not r.available and r.get(turn.turn_id) is None and len(r) == 0

        # Simulate a built cache without loading a model. Stored states are
        # trimmed: 1 s of audio -> 50 frames, not 1500.
        a.mkdir(parents=True)
        fake = torch.randn(50, 384, dtype=torch.float16)
        torch.save(fake, a / f"{turn.turn_id}.pt")

        r2 = EncoderCacheReader(tmp, "openai/whisper-tiny")
        assert r2.available and len(r2) == 1
        got = r2.get(turn.turn_id)
        assert got is not None and got.shape == (50, 384)
        # Trimming is the point: the untrimmed form would be 30x this.
        assert got.shape[0] == int(turn.duration_s * FRAME_RATE)
        assert torch.equal(got, fake), "cache must round-trip exactly"

        # A turn with no cached entry returns None rather than raising, so a
        # partially-built cache degrades to recomputation instead of crashing.
        assert r2.get("deadbeefcafe") is None

        # The other encoder's reader must not see it.
        assert EncoderCacheReader(tmp, "openai/whisper-medium").get(turn.turn_id) is None

    print("encoder_cache: per-encoder isolation, round-trip, graceful miss OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
