#!/usr/bin/env python3
"""Trace one waveform through the three independently scored audio layers.

This probe prevents a strong stock Whisper decoder from hiding a weak
encoder-to-Gemma connector.  Every case is evaluated as:

    waveform -> stock Whisper decoder -> transcript
             -> stock Whisper encoder -> Jaeger adapter -> Gemma transcript
             -> stock Whisper encoder -> Jaeger adapter -> Gemma answer

The first two paths receive the same waveform and use pooled WER against the
human transcript.  The answer path is graded against ``cases.json``.  Whisper
and Gemma remain frozen; this script performs no teacher forcing or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jaeger_agent_omni.app import read_wav  # noqa: E402
from jaeger_agent_omni.model_nodes import (  # noqa: E402
    resolve_audio_node,
    validate_node_checkpoint,
)
from jaeger_agent_omni.runtime.stock_whisper import (  # noqa: E402
    DEFAULT_GEMMA,
    StockWhisperGemmaSession,
    validate_checkpoint,
)
from research.grading import _digits, graded  # noqa: E402
from training.evaluate import normalize, wer  # noqa: E402
from training.stock_whisper_gemma import (  # noqa: E402
    ANSWER_SYSTEM,
    TRANSCRIPT_SYSTEM,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_edits(reference: str, hypothesis: str) -> tuple[int, int, float]:
    # Whisper commonly emits ``2`` where the human reference says ``two``.
    # That is identical lexical content and must not be charged to transfer.
    def canonical(text: str) -> list[str]:
        # Whisper sometimes renders spoken operators typographically. Preserve
        # their lexical meaning before the punctuation normalizer removes them.
        text = text.replace("−", " minus ").replace("×", " times ")
        return _digits(" ".join(normalize(text))).split()

    reference_words = canonical(reference)
    hypothesis_words = canonical(hypothesis)
    count = len(reference_words)
    edits = int(round(wer(reference_words, hypothesis_words) * count))
    return edits, count, edits / max(1, count)


def decode_whisper(cases: list[dict], cases_dir: Path, model_id: str,
                   device_name: str) -> tuple[dict[str, str], dict[str, float]]:
    """Run the complete stock encoder+decoder before loading MLX Gemma."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype).to(device).eval().requires_grad_(False)
    transcripts: dict[str, str] = {}
    times: dict[str, float] = {}
    for case in cases:
        audio = read_wav(cases_dir / f"{case['key']}.wav")
        batch = processor(
            audio, sampling_rate=16_000, return_tensors="pt",
            return_attention_mask=True)
        features = batch.input_features.to(device, dtype=dtype)
        attention_mask = batch.attention_mask.to(device)
        started = time.monotonic()
        with torch.inference_mode():
            ids = model.generate(
                features, attention_mask=attention_mask,
                language="en", task="transcribe")
        if device.type == "mps":
            torch.mps.synchronize()
        times[case["key"]] = (time.monotonic() - started) * 1000
        transcripts[case["key"]] = processor.batch_decode(
            ids, skip_special_tokens=True)[0].strip()
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return transcripts, times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=REPO / "research/cases12")
    parser.add_argument("--model", type=Path, default=DEFAULT_GEMMA)
    parser.add_argument("--whisper", default=None,
                        help="defaults to the encoder recorded in the adapter")
    parser.add_argument("--decoder-device", default="auto",
                        choices=("auto", "mps", "cpu"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested_adapter = args.adapter.expanduser().resolve()
    adapter, node = resolve_audio_node(requested_adapter)
    cases_dir = args.cases.expanduser().resolve()
    values = validate_checkpoint(adapter)
    if node is not None:
        validate_node_checkpoint(node, values)
    whisper_id = args.whisper or str(values["encoder"])
    if whisper_id != str(values["encoder"]):
        raise ValueError(
            f"adapter uses {str(values['encoder'])!r}, not {whisper_id!r}")
    cases = json.loads((cases_dir / "cases.json").read_text())
    if not cases:
        raise ValueError("cases file is empty")

    decoder_source = str(node.encoder_path) if node else whisper_id
    print(f"decoder control: {whisper_id} ({decoder_source})", flush=True)
    decoder_text, decoder_ms = decode_whisper(
        cases, cases_dir, decoder_source, args.decoder_device)

    print(f"adapter path: {adapter}", flush=True)
    session = StockWhisperGemmaSession(
        adapter, args.model, whisper_id, TRANSCRIPT_SYSTEM)
    rows = []
    for case in cases:
        key = case["key"]
        reference = case.get("transcript") or case.get("text") or ""
        if not reference:
            raise ValueError(f"{key}: missing human transcript")
        audio = read_wav(cases_dir / f"{key}.wav")
        session.reset()
        session.open_turn()
        session.feed_audio(audio)
        encode_started = time.monotonic()
        session.end_of_turn()
        encode_ms = (time.monotonic() - encode_started) * 1000

        transcript_budget = max(24, len(normalize(reference)) * 2 + 8)
        session.system = TRANSCRIPT_SYSTEM
        started = time.monotonic()
        gemma_transcript = session.respond(
            max_tokens=transcript_budget, eos_ids={1, 106})
        transcript_ms = (time.monotonic() - started) * 1000

        session.system = ANSWER_SYSTEM
        started = time.monotonic()
        gemma_answer = session.respond(max_tokens=48, eos_ids={1, 106})
        answer_ms = (time.monotonic() - started) * 1000

        decoder_edits, words, decoder_wer = word_edits(
            reference, decoder_text[key])
        adapter_edits, adapter_words, adapter_wer = word_edits(
            reference, gemma_transcript)
        if adapter_words != words:
            raise AssertionError("reference word counts differ between paths")
        answer_ok = graded(gemma_answer, case["answers"])
        row = {
            "case": key,
            "reference": reference,
            "reference_words": words,
            "whisper_decoder_transcript": decoder_text[key],
            "whisper_decoder_edits": decoder_edits,
            "whisper_decoder_wer": decoder_wer,
            "whisper_decoder_ms": decoder_ms[key],
            "gemma_transcript": gemma_transcript,
            "gemma_transcript_edits": adapter_edits,
            "gemma_transcript_wer": adapter_wer,
            "encoder_ms": encode_ms,
            "gemma_transcript_ms": transcript_ms,
            "gemma_answer": gemma_answer,
            "gemma_answer_ok": answer_ok,
            "gemma_answer_ms": answer_ms,
        }
        rows.append(row)
        print(
            f"{key} decoder-WER={100*decoder_wer:5.1f}% "
            f"Gemma-WER={100*adapter_wer:5.1f}% "
            f"answer={'OK' if answer_ok else 'MISS'}\n"
            f"  raw: {decoder_text[key]}\n"
            f"  via: {gemma_transcript}\n"
            f"  ans: {gemma_answer}", flush=True)

    total_words = sum(row["reference_words"] for row in rows)
    decoder_edits = sum(row["whisper_decoder_edits"] for row in rows)
    adapter_edits = sum(row["gemma_transcript_edits"] for row in rows)
    correct = sum(bool(row["gemma_answer_ok"]) for row in rows)
    summary = {
        "reference_words": total_words,
        "whisper_decoder_edits": decoder_edits,
        "whisper_decoder_wer": decoder_edits / max(1, total_words),
        "gemma_transcript_edits": adapter_edits,
        "gemma_transcript_wer": adapter_edits / max(1, total_words),
        "transfer_gap_wer_points":
            100 * (adapter_edits - decoder_edits) / max(1, total_words),
        "gemma_answers_correct": correct,
        "n": len(rows),
        "median_whisper_decoder_ms": statistics.median(
            row["whisper_decoder_ms"] for row in rows),
        "median_encoder_ms": statistics.median(
            row["encoder_ms"] for row in rows),
        "median_gemma_transcript_ms": statistics.median(
            row["gemma_transcript_ms"] for row in rows),
        "median_gemma_answer_ms": statistics.median(
            row["gemma_answer_ms"] for row in rows),
    }
    report = {
        "probe": "raw_decoder_to_gemma_transcript_to_gemma_answer",
        "adapter": str(adapter),
        "node": node.node_id if node else None,
        "adapter_sha256": sha256(adapter),
        "encoder": whisper_id,
        "cases": str(cases_dir),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def selfcheck() -> None:
    assert word_edits("What is two plus two?", "what is two plus two") == (
        0, 5, 0.0)
    edits, words, score = word_edits("one two three four", "one two free four")
    assert (edits, words, score) == (1, 4, 0.25)
    print("transfer_probe: pooled layer boundaries and WER accounting OK")


if __name__ == "__main__":
    raise SystemExit(selfcheck() if "--selfcheck" in sys.argv else main())
