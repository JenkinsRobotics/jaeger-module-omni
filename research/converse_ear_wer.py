#!/usr/bin/env python3
"""Matched WER for the imported Gemma Omni V2 ear on the locked eval manifest.

Twin of `training/stock_whisper_gemma.py wer`: identical row selection
(same manifest, same order, same -n), identical scoring (normalize + wer
from training/evaluate.py), identical output shape, so
training/compare_candidate.py and human eyes can compare the two ears
row-for-row. The only difference is the pipeline under test: this feeds the
RAW waveform to the V2 node's own production transcribe() path instead of
cached encoder features + the session's free-running budget — each ear runs
its own production decoding policy on identical audio.

    .venv/bin/python research/converse_ear_wer.py -n 64 \
        --output research/candidate_evals/converse-93h-<date>-devclean64.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

DEFAULT_MANIFEST = (Path.home() / "models/jaeger-training-data"
                    / "stock-whisper-medium-eval-v1/devclean/manifest.jsonl")
DEFAULT_NODE = (REPO / "jaeger_agent_omni/nodes"
                / "stock-whisper-medium-stream-bargein-93h")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    from evaluate import normalize, wer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", type=Path, default=DEFAULT_NODE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("-n", "--examples", type=int, default=64)
    ap.add_argument(
        "--instructed", action="store_true",
        help="add the transcription instruction, the keypoint's recorded "
             "protocol; without it the ear runs exactly as converse wires it")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import soundfile as sf

    spec = importlib.util.spec_from_file_location(
        "converse_ear_node", args.node / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ear = module.build()
    if args.instructed:
        # Same prompt override as the playground's eval/matched_eval.py.
        import mlx_audio_streaming as mas
        instruction = ("\n\nTranscribe the audio exactly. Reply with the "
                       "transcript text only — no preamble, no corrections, "
                       "no commentary.")
        rendered = ear.tokenizer.apply_chat_template(
            [{"role": "user", "content": mas.AUDIO_MARKER + instruction}],
            tokenize=False, add_generation_prompt=True)
        prefix, suffix = rendered.split(mas.AUDIO_MARKER, 1)
        ear.prefix_ids = ear.tokenizer.encode(prefix, add_special_tokens=False)
        ear.suffix_ids = ear.tokenizer.encode(suffix, add_special_tokens=False)

    rows = []
    for line in args.manifest.expanduser().read_text().splitlines():
        row = json.loads(line)
        if str(row.get("text") or "").strip():
            rows.append(row)
    rows = rows[:args.examples]
    if not rows:
        raise RuntimeError("WER gate selected no held-out rows")

    total_edits = total_words = 0
    outputs = []
    latencies = []
    for index, row in enumerate(rows, 1):
        waveform, rate = sf.read(row["audio"], dtype="float32")
        if rate != 16000:
            raise ValueError(f"{row['id']}: expected 16 kHz, got {rate}")
        started = time.monotonic()
        hypothesis, _stats = ear.transcribe(np.asarray(waveform, np.float32))
        latencies.append(time.monotonic() - started)
        reference_words = normalize(row["text"])
        hypothesis_words = normalize(hypothesis)
        edits = int(round(
            wer(reference_words, hypothesis_words) * len(reference_words)))
        total_edits += edits
        total_words += len(reference_words)
        outputs.append({
            "id": row["id"], "reference": row["text"],
            "hypothesis": hypothesis,
            "wer": edits / max(1, len(reference_words)),
        })
        if index <= 3:
            print(f"ref: {row['text']}")
            print(f"hyp: {hypothesis}\n")
        if index % 16 == 0:
            print(f"{index}/{len(rows)} running WER "
                  f"{100 * total_edits / max(1, total_words):.2f}%", flush=True)

    score = total_edits / max(1, total_words)
    print(f"CORPUS WER {score * 100:.2f}% over {len(rows)} held-out clips, "
          f"{total_words} reference words; "
          f"median transcribe {sorted(latencies)[len(latencies) // 2]:.2f}s")
    adapter = args.node / "weights/adapter.npz"
    args.output.write_text(json.dumps({
        "adapter": str(adapter), "split": "devclean",
        "adapter_sha256": sha256(adapter),
        "encoder": "openai/whisper-medium (node-local snapshot)",
        "evaluation_mode": ("node_transcribe_instructed" if args.instructed
                            else "node_production_transcribe"),
        "teacher_forced": False,
        "input_handoff": "raw_waveform_no_transcript",
        "id_file": str(args.manifest),
        "source_prefix": None,
        "n": len(rows), "wer": score, "edits": total_edits,
        "reference_words": total_words,
        "median_transcribe_s": sorted(latencies)[len(latencies) // 2],
        "rows": outputs,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
