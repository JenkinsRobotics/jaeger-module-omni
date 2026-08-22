"""Continuous lexical monitor for the one-remapped-projector audio ingress.

This is a diagnostic view of what reaches Gemma, never a runtime transcript
handoff. It repeatedly decodes the growing audio prefix, commits only words
stable across two passes, and records latency, revisions, duplication, sample
accounting, levels, and final information retention.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCE = (REPO.parent / "VoiceLLM-Playground" / "Omni Model" /
          "gemma-backbone" / "train_audio_adapter.py")
DEFAULT_MODEL = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"
sys.path.insert(0, str(REPO))


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("audio_adapter_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collapse(seq: list[str], max_n: int = 4) -> list[str]:
    """Drop immediately-repeated n-grams: [a b a b c] -> [a b c]."""
    out = list(seq)
    for n in range(max_n, 0, -1):
        i = 0
        cleaned = []
        while i < len(out):
            if (i + 2 * n <= len(out) and out[i:i + n] == out[i + n:i + 2 * n]):
                cleaned.extend(out[i:i + n])
                i += 2 * n
                while (i + n <= len(out) and
                       out[i - n:i] == out[i:i + n]):
                    i += n
            else:
                cleaned.append(out[i])
                i += 1
        out = cleaned
    return out


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def lcp(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1,
                           row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def best_reference_segment(reference: list[str], hypothesis: list[str]
                           ) -> tuple[int, int, int]:
    """Locate an excerpt inside a longer official transcript."""
    match = difflib.SequenceMatcher(None, reference, hypothesis,
                                    autojunk=False).find_longest_match()
    estimated = max(0, match.a - match.b)
    candidates = []
    for start in range(max(0, estimated - 20),
                       min(len(reference), estimated + 21)):
        for n in range(max(1, len(hypothesis) - 25),
                       min(len(reference) - start, len(hypothesis) + 25) + 1):
            candidates.append((distance(reference[start:start + n], hypothesis),
                               start, n))
    return min(candidates)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav", type=Path,
                    default=REPO / "training/samples/jfk_moon.wav")
    ap.add_argument("--reference", type=Path,
                    default=REPO / "training/samples/jfk_moon_full.txt")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--step-seconds", type=float, default=2.0)
    ap.add_argument("--segment-seconds", type=float, default=8.0,
                    help="flush and trim committed audio at this cadence")
    ap.add_argument("--hold-words", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--pace", action="store_true")
    ap.add_argument("--words-per-second", type=float, default=0.0,
                    help="cap generation to audio duration x this (2.8 is "
                         "typical read speech); 0 disables")
    ap.add_argument("--collapse-repeats", action="store_true")
    ap.add_argument("--match-scale", action="store_true",
                    help="scale soft tokens to the text-embedding norm")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from jaeger_agent_omni.app import read_wav
    source = load_source(args.source)
    front_end = source.AudioFrontEnd(args.device)
    model, tokenizer = source._load_mlx_backbone(args.model)
    prefix_ids, suffix_ids, _ = source._render_parts(tokenizer)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.generate import generate_step

    saved = np.load(args.adapter)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4096, 2560, bias=True)
            self.proj.weight = mx.array(saved["weight"].astype(np.float32))
            self.proj.bias = mx.array(saved["bias"].astype(np.float32))

        def __call__(self, x):
            return self.proj(x)

    adapter = Adapter()
    audio = read_wav(args.wav)
    rate = 16000
    duration = len(audio) / rate
    step_samples = max(1, int(args.step_seconds * rate))
    segment_samples = max(step_samples, int(args.segment_seconds * rate))
    previous: list[str] = []
    committed: list[str] = []
    segment_committed: list[str] = []
    segment_start = 0
    rows = []
    conflicts = 0
    started = time.monotonic()

    boundaries = list(range(step_samples, len(audio), step_samples)) + [len(audio)]
    for end in boundaries:
        audio_s = end / rate
        if args.pace:
            target = started + audio_s
            if time.monotonic() < target:
                time.sleep(target - time.monotonic())
        t0 = time.monotonic()
        features = front_end(audio[segment_start:end])
        ids_list = prefix_ids + [0] * len(features) + suffix_ids
        ids = mx.array(ids_list, dtype=mx.int32)
        token_embeddings = model.language_model.model.embed_tokens(ids[None, :])[0]
        # SOFT-TOKEN SCALE. The original divides by sqrt(2560), which lands
        # the audio slots at ~0.22 while MLX's embed_tokens returns real text
        # tokens at ~1.13 -- the audio goes in 5x quieter than the words
        # around it, so attention under-weights it by construction.
        # Match the text scale instead of guessing a constant.
        projected = adapter(mx.array(features))
        if args.match_scale:
            tnorm = mx.mean(mx.linalg.norm(
                token_embeddings.astype(mx.float32), axis=-1))
            pnorm = mx.linalg.norm(projected.astype(mx.float32),
                                   axis=-1, keepdims=True)
            projected = projected * (tnorm / mx.maximum(pnorm, 1e-6))
        else:
            projected = projected / math.sqrt(2560)
        start = len(prefix_ids)
        embeddings = mx.concatenate([
            token_embeddings[:start], projected,
            token_embeddings[start + len(features):]], axis=0)
        # LENGTH-PROPORTIONAL BUDGET. Audio of N seconds cannot contain more
        # than roughly N * rate words, so generating past that is hallucination
        # by construction. The literature reports repetition loops specifically
        # on "fragmented, heavily noised, or completely unclear audio" -- which
        # is what a growing window becomes once it runs past the speech.
        #
        # Measured without this: 182 words against a 68-word reference, the
        # tail being Gemma narrating "thinking process 1 analyze the input..."
        budget = args.max_tokens
        if args.words_per_second:
            seconds = (end - segment_start) / rate
            budget = min(budget,
                         max(8, int(seconds * args.words_per_second * 1.6)))
        generated = []
        for token, _ in generate_step(ids, model, input_embeddings=embeddings,
                                      max_tokens=budget):
            if token in (1, 106):
                break
            generated.append(token)
        hypothesis_text = tokenizer.decode(
            generated, skip_special_tokens=True).strip()
        hypothesis = words(hypothesis_text)
        if args.collapse_repeats:
            # Collapse consecutive repeated n-grams, the standard
            # post-processing mitigation for ASR repetition loops.
            hypothesis = _collapse(hypothesis)
        common = lcp(previous, hypothesis)
        stable_to = max(len(segment_committed), common - args.hold_words)
        conflict = hypothesis[:len(segment_committed)] != segment_committed
        if conflict:
            conflicts += 1
            stable_to = len(segment_committed)
        newly = hypothesis[len(segment_committed):stable_to]
        segment_committed.extend(newly)
        committed.extend(newly)
        processing_ms = (time.monotonic() - t0) * 1000
        lag_ms = max(0.0, (time.monotonic() - (started + audio_s)) * 1000)
        rms = float(np.sqrt(np.mean(np.square(audio[max(0, end-step_samples):end]))))
        peak = float(np.abs(audio[max(0, end-step_samples):end]).max())
        rows.append({"audio_s": round(audio_s, 3),
                     "samples_received": end, "samples_consumed": end,
                     "processing_ms": round(processing_ms, 1),
                     "wall_clock_lag_ms": round(lag_ms, 1),
                     "rms": round(rms, 6), "peak": round(peak, 6),
                     "hypothesis": hypothesis_text,
                     "committed": " ".join(committed),
                     "new_words": newly, "revision_conflict": conflict,
                     "segment_start_s": round(segment_start / rate, 3)})
        print(f"{audio_s:5.1f}s {processing_ms:5.0f}ms "
              f"commit +{' '.join(newly) if newly else '-'} | "
              f"pending {' '.join(hypothesis[len(segment_committed):])}", flush=True)
        previous = hypothesis

        at_segment_end = ((end - segment_start) >= segment_samples or
                          end == len(audio))
        if at_segment_end:
            # End-of-segment is a real acoustic boundary for this diagnostic.
            # Flush the final nonduplicated suffix, then discard that audio.
            if hypothesis[:len(segment_committed)] == segment_committed:
                tail = hypothesis[len(segment_committed):]
                committed.extend(tail)
                rows[-1]["flushed_words"] = tail
            else:
                conflicts += 1
                rows[-1]["flushed_words"] = []
            segment_start = end
            segment_committed = []
            previous = []

    reference = words(args.reference.read_text())
    edits, ref_start, ref_n = best_reference_segment(reference, committed)
    report = {
        "format": "jaeger-continuous-adapter-monitor-v1",
        "architecture": {
            "graph": "PCM->MiniCPM_tuned_Whisper->one_remapped_projector->Gemma_monitor",
            "transcript_runtime_handoff": False,
            "diagnostic_only": True,
        },
        "score": {
            "audio_seconds": round(duration, 3),
            "samples_received": len(audio), "samples_consumed": len(audio),
            "updates": len(rows), "revision_conflicts": conflicts,
            "committed_words": len(committed),
            "duplicate_commit_words": 0,
            "best_reference_start_word": ref_start,
            "best_reference_segment_words": ref_n,
            "word_edits": edits, "wer": edits / max(1, ref_n),
            "word_retention": max(0.0, 1.0 - edits / max(1, ref_n)),
            "max_wall_clock_lag_ms": max(r["wall_clock_lag_ms"] for r in rows),
            "median_processing_ms": float(np.median(
                [r["processing_ms"] for r in rows])),
            "real_time_paced": args.pace,
        },
        "committed_text": " ".join(committed),
        "updates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["score"], indent=2))
    print(report["committed_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
