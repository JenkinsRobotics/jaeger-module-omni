"""Continuous lexical monitor for the one-remapped-projector audio ingress.

This is a diagnostic view of what reaches Gemma, never a runtime transcript
handoff. It repeatedly decodes the growing audio prefix, commits only words
stable across two passes, and records latency, revisions, duplication, sample
accounting, levels, and final information retention.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCE = (REPO.parent / "VoiceLLM-Playground" / "Omni Model" /
          "gemma-backbone" / "train_audio_adapter.py")
DEFAULT_MODEL = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("audio_adapter_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def best_reference_prefix(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    """Official text is longer than the clip; find its best leading cut."""
    lo = max(1, len(hypothesis) - 30)
    hi = min(len(reference), len(hypothesis) + 30)
    choices = [(distance(reference[:n], hypothesis), n)
               for n in range(lo, hi + 1)]
    return min(choices)


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
    ap.add_argument("--hold-words", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--pace", action="store_true")
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
    previous: list[str] = []
    committed: list[str] = []
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
        features = front_end(audio[:end])
        ids_list = prefix_ids + [0] * len(features) + suffix_ids
        ids = mx.array(ids_list, dtype=mx.int32)
        token_embeddings = model.language_model.model.embed_tokens(ids[None, :])[0]
        projected = adapter(mx.array(features)) / math.sqrt(2560)
        start = len(prefix_ids)
        embeddings = mx.concatenate([
            token_embeddings[:start], projected,
            token_embeddings[start + len(features):]], axis=0)
        generated = []
        for token, _ in generate_step(ids, model, input_embeddings=embeddings,
                                      max_tokens=args.max_tokens):
            if token in (1, 106):
                break
            generated.append(token)
        hypothesis_text = tokenizer.decode(
            generated, skip_special_tokens=True).strip()
        hypothesis = words(hypothesis_text)
        common = lcp(previous, hypothesis)
        stable_to = max(len(committed), common - args.hold_words)
        if hypothesis[:len(committed)] != committed:
            conflicts += 1
            stable_to = len(committed)
        newly = hypothesis[len(committed):stable_to]
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
                     "new_words": newly, "revision_conflict":
                     hypothesis[:len(committed)] != committed})
        print(f"{audio_s:5.1f}s {processing_ms:5.0f}ms "
              f"commit +{' '.join(newly) if newly else '-'} | "
              f"pending {' '.join(hypothesis[len(committed):])}", flush=True)
        previous = hypothesis

    # End-of-stream flush: append only the uncommitted suffix of the final
    # hypothesis. Since committed is always a prefix, this cannot duplicate.
    if previous[:len(committed)] == committed:
        committed.extend(previous[len(committed):])
    else:
        conflicts += 1
    reference = words(args.reference.read_text())
    edits, ref_n = best_reference_prefix(reference, committed)
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
            "best_reference_prefix_words": ref_n,
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
