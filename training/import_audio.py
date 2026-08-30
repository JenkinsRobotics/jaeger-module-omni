"""Turn long-form audio into a training corpus.

Any wav — a recorded session, a podcast, a speech — becomes capture-format
turns: segmented on silence, transcribed with timestamps, confidence-scored.

    python training/import_audio.py --audio speech.wav --out corpus/ \
        --whisper openai/whisper-medium

Segments come out as (audio, transcript) pairs, which is what Stage 3 warmup
trains on. There is no response field to fill: for a monologue there is no
reply, and warmup does not need one. Stage 4 needs responses, so it needs
conversational sources — see `data.py` tiers.

Why segment at all rather than feed the whole file: Whisper's context is 30 s,
and a projector trained on 18-minute inputs would never see the shape it meets
at inference, which is one utterance at a time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.core.capture import SAMPLE_RATE, CaptureWriter  # noqa: E402

MIN_SEGMENT_S = 1.0
MAX_SEGMENT_S = 28.0      # under whisper's 30 s window, with headroom


# Absolute floor. A purely relative threshold cannot tell a quiet recording
# from silence: normalise an all-zero file and every frame reads as loud as the
# loudest, so the whole thing scores as speech. Anything below this is silence
# no matter what the rest of the file looks like.
ABSOLUTE_SILENCE_RMS = 1e-4


def split_on_silence(audio: np.ndarray, *, top_db: float = 32.0,
                     min_silence_s: float = 0.35,
                     min_segment_s: float = MIN_SEGMENT_S,
                     max_segment_s: float = MAX_SEGMENT_S) -> list:
    """Segment on energy. Returns [(start_sample, end_sample), ...].

    ponytail: RMS over frames + a dB threshold, not webrtcvad. This is offline
    batch work on clean-ish recordings; the live path has a real VAD already.
    """
    frame = int(0.02 * SAMPLE_RATE)                       # 20 ms
    n_frames = max(1, len(audio) // frame)
    rms = np.array([
        np.sqrt(np.mean(audio[i*frame:(i+1)*frame] ** 2) + 1e-12)
        for i in range(n_frames)])
    if rms.max() < ABSOLUTE_SILENCE_RMS:
        return []
    db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = (db > -top_db) & (rms > ABSOLUTE_SILENCE_RMS)

    min_silence_frames = max(1, int(min_silence_s * SAMPLE_RATE / frame))
    segments, start, silence = [], None, 0

    for i, is_voice in enumerate(voiced):
        if is_voice:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_silence_frames:
                segments.append((start * frame, (i - silence + 1) * frame))
                start = None
    if start is not None:
        segments.append((start * frame, len(audio)))

    # Enforce bounds: drop the too-short, split the too-long at their midpoint
    # rather than discarding minutes of usable speech.
    out = []
    for s, e in segments:
        dur = (e - s) / SAMPLE_RATE
        if dur < min_segment_s:
            continue
        while (e - s) / SAMPLE_RATE > max_segment_s:
            cut = s + int(max_segment_s * SAMPLE_RATE)
            out.append((s, cut))
            s = cut
        if (e - s) / SAMPLE_RATE >= min_segment_s:
            out.append((s, e))
    return out


def mean_logprob_to_confidence(logprob: float | None) -> float:
    """Whisper's avg token logprob -> a 0..1 score for the corpus gate.

    Not calibrated, and not pretending to be. It is monotonic in the model's
    own certainty, which is all a threshold needs.
    """
    if logprob is None:
        return 0.5
    return float(np.clip(np.exp(logprob), 0.0, 1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper", default="openai/whisper-medium")
    ap.add_argument("--language", default="en")
    ap.add_argument("--soul", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N segments (smoke testing)")
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from jaeger_agent_omni.app import read_wav, transcribe_kwargs

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    audio = read_wav(args.audio)
    segments = split_on_silence(audio)

    # Report coverage over ALL segments before --limit truncates, or the
    # percentage silently describes the sample rather than the file.
    voiced_s = sum(e - s for s, e in segments) / SAMPLE_RATE
    total_s = len(audio) / SAMPLE_RATE
    print(f"{total_s/60:.1f} min -> {len(segments)} segments "
          f"({voiced_s/60:.1f} min speech, {100*voiced_s/total_s:.0f}% voiced)")

    if args.limit and args.limit < len(segments):
        segments = segments[: args.limit]
        kept_s = sum(e - s for s, e in segments) / SAMPLE_RATE
        print(f"  --limit {args.limit}: importing {kept_s/60:.1f} min of that")

    processor = WhisperProcessor.from_pretrained(args.whisper)
    model = WhisperForConditionalGeneration.from_pretrained(args.whisper).to(device).eval()

    writer = CaptureWriter(args.out)
    kept = 0
    for i, (s, e) in enumerate(segments):
        clip = audio[s:e]
        feats = processor(clip, sampling_rate=SAMPLE_RATE,
                          return_tensors="pt").input_features.to(device, model.dtype)
        with torch.no_grad():
            out = model.generate(feats, output_scores=True,
                                 return_dict_in_generate=True,
                                 **transcribe_kwargs(model, args.language))
        text = processor.batch_decode(out.sequences, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
        if not text:
            continue

        scores = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True)
        conf = mean_logprob_to_confidence(scores[0][scores[0].isfinite()].mean().item())

        writer.write(clip, transcript=text, response="", confidence=conf,
                     soul=args.soul, session_id=Path(args.audio).stem,
                     meta={"source": Path(args.audio).name,
                           "start_s": round(s / SAMPLE_RATE, 2)})
        kept += 1
        if kept % 10 == 0:
            print(f"  {kept}/{len(segments)}  conf {conf:.2f}  {text[:52]}")

    from jaeger_agent_omni.core.capture import CaptureReader
    r = CaptureReader(args.out)
    print(f"wrote {kept} turns, {r.hours()*60:.1f} min to {args.out}")
    print(f"      {len(list(CaptureReader(args.out, min_confidence=0.8)))} pass the 0.8 gate")
    return 0


def demo():
    """Segmentation only — the transcribe path needs weights."""
    sr = SAMPLE_RATE
    rng = np.random.default_rng(0)

    def speech(sec):
        return (rng.standard_normal(int(sr * sec)) * 0.3).astype(np.float32)

    def quiet(sec):
        return np.zeros(int(sr * sec), dtype=np.float32)

    # 2s speech, 0.6s gap, 3s speech, 0.6s gap, 0.3s blip (too short to keep)
    audio = np.concatenate([speech(2), quiet(0.6), speech(3), quiet(0.6), speech(0.3)])
    segs = split_on_silence(audio)
    assert len(segs) == 2, [(s/sr, e/sr) for s, e in segs]
    assert all((e - s) / sr >= MIN_SEGMENT_S for s, e in segs)
    # Segments must be ordered and non-overlapping, or turns interleave.
    assert all(segs[i][1] <= segs[i+1][0] for i in range(len(segs)-1))

    # Over-length audio is split, not dropped: a 70 s monologue is three usable
    # turns, and discarding it would throw away the most fluent speech.
    long_audio = speech(70)
    segs = split_on_silence(long_audio)
    assert len(segs) >= 3, len(segs)
    assert all((e - s) / sr <= MAX_SEGMENT_S + 1e-6 for s, e in segs)
    assert sum(e - s for s, e in segs) / sr > 65, "lost audio while splitting"

    # Silence alone yields nothing rather than one giant empty turn. A purely
    # relative threshold gets this wrong: normalising an all-zero file makes
    # every frame read as loud as the loudest.
    assert split_on_silence(quiet(5)) == []
    assert split_on_silence(np.zeros(sr, np.float32) + 1e-6) == []

    # Speech preceded and followed by silence must not absorb the silence.
    padded = np.concatenate([quiet(1.5), speech(2), quiet(1.5)])
    segs = split_on_silence(padded)
    assert len(segs) == 1, segs
    start_s, end_s = segs[0][0] / sr, segs[0][1] / sr
    assert 1.3 < start_s < 1.7 and 3.3 < end_s < 3.9, (start_s, end_s)

    # Confidence mapping is monotonic and bounded.
    assert mean_logprob_to_confidence(None) == 0.5
    assert 0.0 <= mean_logprob_to_confidence(-5.0) < \
        mean_logprob_to_confidence(-0.1) <= 1.0

    print(f"import_audio: segmentation {len(split_on_silence(audio))} segs, "
          f"bounds + splitting + confidence OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
