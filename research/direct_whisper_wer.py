#!/usr/bin/env python3
"""Run Whisper's own decoder on the exact adapter evaluation manifests.

This is the lexical source control for the end-to-end adapter: identical
waveforms, order, references, normalization, and corpus-level WER, but no
adapter or Gemma generation step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SAMPLE_RATE = 16_000


def read_manifest(spec: str) -> tuple[str, Path, list[dict]]:
    try:
        name, raw_path = spec.split("=", 1)
    except ValueError as exc:
        raise ValueError("dataset must be NAME=MANIFEST.jsonl") from exc
    path = Path(raw_path).expanduser().resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        audio = Path(row["audio"]).expanduser()
        if not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        row["audio"] = str(audio)
        row["target"] = row.get("target", row.get("text", ""))
    return name, path, rows


def read_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        count = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, count),
                          np.arange(len(audio)), audio)
    return np.asarray(audio, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", required=True,
                        help="repeatable NAME=MANIFEST.jsonl")
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument("--device", default="mps",
                        choices=("mps", "cuda", "cpu"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from training.evaluate import normalize, wer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model = model.to(args.device).eval()
    results = {}

    for spec in args.dataset:
        name, manifest, rows = read_manifest(spec)
        if args.limit:
            rows = rows[:args.limit]
        edits_total = 0
        normalized_edits_total = 0
        words_total = 0
        normalized_words_total = 0
        elapsed = []
        outputs = []
        for index, row in enumerate(rows, 1):
            audio = read_audio(row["audio"])
            inputs = processor(audio, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt")
            features = inputs.input_features.to(
                device=args.device, dtype=model.dtype)
            started = time.monotonic()
            with torch.inference_mode():
                token_ids = model.generate(
                    features, language=args.language, task="transcribe",
                    num_beams=args.beam_size)
            hypothesis = processor.batch_decode(
                token_ids, skip_special_tokens=True)[0]
            elapsed_ms = (time.monotonic() - started) * 1000
            reference_words = normalize(row["target"])
            hypothesis_words = normalize(hypothesis)
            edits = int(round(wer(reference_words, hypothesis_words)
                              * len(reference_words)))
            normalized_reference = processor.tokenizer.normalize(
                row["target"]).split()
            normalized_hypothesis = processor.tokenizer.normalize(
                hypothesis).split()
            normalized_edits = int(round(
                wer(normalized_reference, normalized_hypothesis)
                * len(normalized_reference)))
            edits_total += edits
            words_total += len(reference_words)
            normalized_edits_total += normalized_edits
            normalized_words_total += len(normalized_reference)
            elapsed.append(elapsed_ms)
            outputs.append({
                "id": row.get("id"), "reference": row["target"],
                "hypothesis": hypothesis,
                "wer": edits / max(1, len(reference_words)),
                "normalized_wer": normalized_edits /
                                  max(1, len(normalized_reference)),
                "decoder_ms": elapsed_ms,
            })
            if index <= 3:
                print(f"[{name}] ref: {row['target']}")
                print(f"[{name}] hyp: {hypothesis}\n")
            if index % 16 == 0:
                print(f"[{name}] {index}/{len(rows)} running WER "
                      f"{100*edits_total/max(1, words_total):.2f}%", flush=True)
        score = edits_total / max(1, words_total)
        normalized_score = normalized_edits_total / max(
            1, normalized_words_total)
        results[name] = {
            "manifest": str(manifest), "n": len(rows), "wer": score,
            "edits": edits_total, "reference_words": words_total,
            "normalized_wer": normalized_score,
            "normalized_edits": normalized_edits_total,
            "normalized_reference_words": normalized_words_total,
            "median_decoder_ms": statistics.median(elapsed), "rows": outputs,
        }
        print(f"[{name}] CORPUS WER {100*score:.2f}% over {len(rows)} clips, "
              f"{words_total} reference words; standard-normalized "
              f"{100*normalized_score:.2f}%", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model,
        "evaluation_mode": "whisper_decoder_free_running",
        "teacher_forced": False,
        "language": args.language,
        "beam_size": args.beam_size,
        "datasets": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
