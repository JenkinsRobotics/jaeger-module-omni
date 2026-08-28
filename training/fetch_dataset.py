"""Pull public speech corpora into the capture format.

Every corpus stores its audio and text differently. Rather than special-case
each one, a corpus declares **how to map its schema onto ours** and the
pipeline is shared:

    Corpus(key=..., hf_id=..., text_field="normalized_text",
           text_transform=strip_markers)

Adding a corpus is a few lines of mapping, not new code. Same idea as the
modality projectors: an adapter at the boundary, one shared path behind it.

    python training/fetch_dataset.py --list
    python training/fetch_dataset.py --dataset librispeech --hours 5 --out corpus/
    python training/fetch_dataset.py --mix librispeech,tedlium --hours 6 --out corpus/

Human transcripts land with `confidence=1.0` — ground truth, no gate needed. No
ASR error enters the teacher path, which is the whole reason Tier 1 outranks
raw volume.

ponytail: `datasets` v5 wants torchcodec to decode audio, dragging in ffmpeg.
We take raw bytes (`Audio(decode=False)`) and decode with soundfile, already a
dependency.
"""

from __future__ import annotations

import argparse
import io
import re
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.capture import SAMPLE_RATE, CaptureWriter  # noqa: E402

MIN_CLIP_S = 0.5
MAX_CLIP_S = 30.0        # whisper's context window


# -- text normalisation -------------------------------------------------------
# Casing carries no loudness. Delivery is a PerformanceSpec field
# (normal/whisper/shout) precisely so text does not have to encode it, and
# corpora disagree wildly about case and punctuation anyway: LibriSpeech is
# UPPERCASE with no punctuation, TED-LIUM and AMI are natural. Mixing them raw
# teaches the model the inconsistency.
#
# Casing and punctuation restoration is its own task (inverse text
# normalisation) with its own training data. Not this one's job.

_MARKERS = re.compile(r"[<\[\(](?:unk|noise|laughter|applause|silence|music|"
                      r"vocalized-noise|breath|cough|sil)[>\]\)]", re.I)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop corpus markers, collapse whitespace."""
    text = _MARKERS.sub(" ", text or "")
    return _WS.sub(" ", text).strip().lower()


def strip_ted_markers(text: str) -> str:
    """TED-LIUM uses {breath} style markers and ' ' spacing quirks."""
    return re.sub(r"\{[^}]*\}", " ", text or "").replace(" '", "'")


def drop_ignore(text: str) -> str:
    """AMI/CHiME mark unusable segments; empty means the caller skips it."""
    return "" if (text or "").strip().lower() in ("ignore_time_segment_in_scoring",
                                                  "ignore_time_segment") else text


# -- corpus catalog -----------------------------------------------------------
@dataclass(frozen=True)
class Corpus:
    key: str
    hf_id: str
    config: str | None
    split: str
    text_field: str
    tier: int
    licence: str
    notes: str
    audio_field: str = "audio"
    text_transform: Callable[[str], str] | None = None
    verified: bool = False        # schema confirmed against the live hub
    needs_auth: bool = False

    def extract(self, sample: dict) -> tuple:
        """Map this corpus's schema onto (audio_dict, text). The adapter."""
        text = sample.get(self.text_field, "")
        if self.text_transform:
            text = self.text_transform(text)
        return sample.get(self.audio_field), normalize_text(text)


CORPORA = {c.key: c for c in [
    Corpus("librispeech", "openslr/librispeech_asr", "clean", "train.100",
           "text", 1, "CC BY 4.0",
           "~100 h read speech, human transcripts. The standard warmup set.",
           verified=True),
    Corpus("librispeech_dev", "openslr/librispeech_asr", "clean", "validation",
           "text", 1, "CC BY 4.0",
           "~5.4 h, 38 speakers. Fetches in minutes; good first real run.",
           verified=True),
    Corpus("librispeech_other", "openslr/librispeech_asr", "other", "train.500",
           "text", 1, "CC BY 4.0",
           "~500 h, harder acoustics than 'clean'. Diversity, not just volume.",
           verified=True),
    Corpus("librispeech_tiny", "hf-internal-testing/librispeech_asr_dummy",
           "clean", "validation", "text", 1, "CC BY 4.0",
           "73 clips. Proves the path end to end in seconds.", verified=True),
    Corpus("tedlium", "LIUM/tedlium", "release3", "train",
           "text", 1, "CC BY-NC-ND 3.0",
           "~450 h TED talks. Spontaneous, varied speakers. CHECK LICENCE: NC.",
           text_transform=strip_ted_markers),
    Corpus("voxpopuli", "facebook/voxpopuli", "en", "train",
           "normalized_text", 1, "CC0",
           "~500 h parliamentary. Accented English, public domain."),
    Corpus("fleurs", "google/fleurs", "en_us", "train",
           "transcription", 1, "CC BY 4.0",
           "~10 h read, many languages available. Small and clean.",
           verified=True),      # 2596 turns fetched live 2026-08-20
    Corpus("ami", "edinburghcstr/ami", "ihm", "train",
           "text", 2, "CC BY 4.0",
           "Real meeting rooms, distance mics, overlap. Tier 2 far-field.",
           text_transform=drop_ignore),
    Corpus("peoples_speech", "MLCommons/peoples_speech", "clean", "train",
           "text", 1, "CC BY-SA 4.0",
           "~30k h. Volume and acoustic variety; large download."),
    Corpus("common_voice", "mozilla-foundation/common_voice_17_0", "en", "train",
           "sentence", 1, "CC0",
           "Accent and speaker diversity. Needs a HF token + terms accepted.",
           needs_auth=True),
]}


def to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(data) * SAMPLE_RATE / sr))
        data = np.interp(np.linspace(0, len(data) - 1, n),
                         np.arange(len(data)), data)
    return data.astype(np.float32)


def fetch(corpus: Corpus, out, *, hours=None, limit=None, skip_hours=0.0,
          verbose=True) -> tuple:
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(corpus.hf_id, corpus.config, split=corpus.split,
                      streaming=True)
    ds = ds.cast_column(corpus.audio_field, Audio(decode=False))

    writer = CaptureWriter(out)
    stopping = {"flag": False}

    def stop(_s, _f):
        stopping["flag"] = True
        if verbose:
            print("\nfetch: interrupted — keeping what was written")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    written, seconds, skipped = 0, 0.0, 0
    offset_seconds = 0.0
    budget = hours * 3600 if hours else None
    offset_budget = skip_hours * 3600

    for sample in ds:
        if stopping["flag"]:
            break
        a, text = corpus.extract(sample)
        if not text or a is None:
            skipped += 1
            continue

        src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
        try:
            data, sr = sf.read(src, dtype="float32")
        except Exception:                              # noqa: BLE001
            skipped += 1
            continue

        audio = to_mono_16k(data, sr)
        dur = len(audio) / SAMPLE_RATE
        if not MIN_CLIP_S <= dur <= MAX_CLIP_S:
            skipped += 1
            continue
        if offset_seconds < offset_budget:
            offset_seconds += dur
            continue

        writer.write(audio, transcript=text, response="",
                     confidence=1.0,          # human transcript: ground truth
                     session_id=corpus.key,
                     meta={"source": corpus.hf_id, "tier": corpus.tier,
                           "speaker": str(sample.get("speaker_id", "")),
                           "license": corpus.licence})
        written += 1
        seconds += dur

        if verbose and written % 200 == 0:
            print(f"  {corpus.key}: {written} turns  {seconds/60:.1f} min")
        if (budget and seconds >= budget) or (limit and written >= limit):
            break

    return written, seconds, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dataset", default=None, choices=list(CORPORA))
    ap.add_argument("--mix", default=None,
                    help="comma-separated keys; the hours budget is split evenly")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--skip-hours", type=float, default=0.0,
                    help="skip this much usable source audio before writing")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        print(f"{'key':18s} {'tier':4s} {'ok':3s} {'licence':14s} notes")
        for c in sorted(CORPORA.values(), key=lambda c: (c.tier, c.key)):
            mark = "yes" if c.verified else ("AUTH" if c.needs_auth else "?")
            print(f"{c.key:18s} {c.tier:<4d} {mark:3s} {c.licence:14s} {c.notes}")
        print("\nok: schema verified against the live hub. '?' means the mapping "
              "is declared but unconfirmed — try it, it will fail loudly.")
        return 0

    if not args.out or not (args.dataset or args.mix):
        ap.error("need --out and one of --dataset / --mix")

    keys = args.mix.split(",") if args.mix else [args.dataset]
    unknown = [k for k in keys if k not in CORPORA]
    if unknown:
        ap.error(f"unknown corpora: {unknown}")

    # Mixing beats volume from one source: a projector trained on one corpus
    # learns that corpus's microphones and speaking style.
    per = (args.hours / len(keys)) if args.hours else None
    total_n, total_s, failures = 0, 0.0, []

    for key in keys:
        c = CORPORA[key]
        print(f"\nfetching {c.hf_id} [{c.config}/{c.split}]  tier {c.tier}  {c.licence}")
        if c.needs_auth:
            print("  needs a HF token and accepted terms; will fail otherwise")
        try:
            n, secs, skipped = fetch(
                c, args.out, hours=per, limit=args.limit,
                skip_hours=args.skip_hours)
        except Exception as exc:                       # noqa: BLE001
            # One bad corpus must not lose the others already written.
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            failures.append(key)
            continue
        print(f"  {n} turns, {secs/60:.1f} min ({skipped} skipped)")
        total_n += n
        total_s += secs

    print(f"\ntotal {total_n} turns, {total_s/60:.1f} min ({total_s/3600:.2f} h) "
          f"in {args.out}")
    if failures:
        print(f"failed: {', '.join(failures)}")
    return 1 if failures and total_n == 0 else 0


def demo():
    # Normalisation is uniform across corpora, or the model learns their
    # disagreement about case and punctuation instead of the speech.
    assert normalize_text("MISTER QUILTER IS THE APOSTLE") == \
        "mister quilter is the apostle"
    assert normalize_text("Hello   there") == "hello there"
    assert normalize_text("") == "" and normalize_text(None) == ""

    # Corpus markers are noise, not words.
    assert normalize_text("hello [laughter] there") == "hello there"
    assert normalize_text("um <unk> yes") == "um yes"
    assert "applause" not in normalize_text("well [APPLAUSE] then")

    # Per-corpus transforms only remove; normalize_text does the tidying.
    assert "{breath}" not in strip_ted_markers("so {breath} then")
    assert normalize_text(strip_ted_markers("so {breath} then")) == "so then"
    assert drop_ignore("ignore_time_segment_in_scoring") == ""
    assert drop_ignore("real words") == "real words"

    # The adapter maps each schema onto ours.
    ted = CORPORA["tedlium"]
    a, t = ted.extract({"audio": {"bytes": b""}, "text": "SO {breath} THEN"})
    assert t == "so then", t
    assert a == {"bytes": b""}

    ami = CORPORA["ami"]
    _, t = ami.extract({"audio": {}, "text": "ignore_time_segment_in_scoring"})
    assert t == "", "unusable AMI segments must map to empty and be skipped"

    vox = CORPORA["voxpopuli"]
    _, t = vox.extract({"audio": {}, "normalized_text": "THE HOUSE",
                        "text": "should not be used"})
    assert t == "the house", "must read the declared field, not guess"

    # Missing field maps to empty rather than crashing a 5-hour fetch.
    _, t = CORPORA["librispeech"].extract({"audio": {}})
    assert t == ""

    # Audio conversion.
    stereo = np.stack([np.ones(1000), np.ones(1000) * 3], axis=1)
    mono = to_mono_16k(stereo, SAMPLE_RATE)
    assert mono.ndim == 1 and np.allclose(mono, 2.0), "stereo must average"
    assert len(to_mono_16k(np.zeros(8000, np.float32), 8000)) == 16000
    assert len(to_mono_16k(np.zeros(48000, np.float32), 48000)) == 16000

    # Catalog integrity.
    for c in CORPORA.values():
        assert c.text_field and c.licence and c.hf_id and c.tier in (1, 2, 3)
        assert c.key in CORPORA
    assert sum(c.verified for c in CORPORA.values()) >= 4

    print(f"fetch_dataset: {len(CORPORA)} corpora "
          f"({sum(c.verified for c in CORPORA.values())} verified), "
          f"adapters + normalisation OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
