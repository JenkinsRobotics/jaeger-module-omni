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
                / "gemmaomni-v2-stream-bargein-93h")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    from evaluate import normalize, wer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    route = ap.add_mutually_exclusive_group()
    route.add_argument("--node", type=Path)
    route.add_argument("--tier", choices=["half", "quasi", "full"],
                       help="evaluate the actual converse tier's selected ear")
    ap.add_argument("--heard-reply", action="store_true",
                    help="score HEARD from the production one-generation reply "
                         "prompt; fresh history and no speaker context per clip")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("-n", "--examples", type=int, default=64)
    ap.add_argument(
        "--instructed", action="store_true",
        help="add the transcription instruction, the keypoint's recorded "
             "protocol; without it the ear runs exactly as converse wires it")
    ap.add_argument(
        "--adapter", type=Path, default=None,
        help="cross-mount a different streaming_causal_v1 adapter npz at the "
             "converse seam (encoder and Gemma stay the node's own). Metadata "
             "keys are filtered; only dotted parameter keys are loaded.")
    ap.add_argument(
        "--system", default=None,
        help="wrap the audio turn in this system prompt (e.g. the adapter's "
             "training-time TRANSCRIPT_SYSTEM) instead of a bare audio turn")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.heard_reply and (args.instructed or args.system):
        ap.error("--heard-reply uses the production reply prompt")
    from jaeger_agent_omni import converse
    if args.tier:
        args.node = REPO / "jaeger_agent_omni/nodes" / converse.semantic_node_id(args.tier)
    else:
        args.node = args.node or DEFAULT_NODE

    import soundfile as sf

    if args.tier:
        ear = converse.load_trained_semantic(args.tier)
    else:
        spec = importlib.util.spec_from_file_location(
            "converse_ear_node", args.node / "runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ear = getattr(module, "build_converse", module.build)()
    if args.adapter:
        with np.load(args.adapter, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arch = bytes(arrays.get("arch", b"")).decode(errors="replace")
        if "streaming_causal" not in arch and arch != "context_causal_v1":
            raise SystemExit(f"--adapter is {arch or 'unknown'}, not "
                             "streaming_causal_v1; the seam is causal-only "
                             "(bidirectional adapters need streaming training)")
        params = {k: v for k, v in arrays.items() if "." in k}
        if arch == "context_causal_v1":
            from gemmaomni_adapter import build_context_adapter
            ear.adapter = build_context_adapter(
                int(arrays["hidden"][0]), arrays=params, causal=True)
        else:
            from streaming_adapter import build_streaming_adapter
            ear.adapter = build_streaming_adapter(
                int(arrays["hidden"][0]), arrays=params, causal=True)
    if args.instructed and args.system:
        raise SystemExit("--instructed and --system are alternative prompts")
    if args.instructed or args.system:
        import mlx_audio_streaming as mas
        if args.system:
            # The adapter's own training-time system prompt, jaeger-style.
            messages = [{"role": "system", "content": args.system},
                        {"role": "user", "content": mas.AUDIO_MARKER}]
        else:
            # Same prompt override as the playground's eval/matched_eval.py.
            instruction = ("\n\nTranscribe the audio exactly. Reply with the "
                           "transcript text only — no preamble, no "
                           "corrections, no commentary.")
            messages = [{"role": "user",
                         "content": mas.AUDIO_MARKER + instruction}]
        rendered = ear.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
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
        audio = np.asarray(waveform, np.float32)
        if args.heard_reply:
            raw, _stats = ear.respond(audio, system=converse.NN_SYSTEM,
                                      history=[], max_tokens=140)
            hypothesis, _reply = converse.parse_heard_reply(raw)
        else:
            hypothesis, _stats = ear.transcribe(audio)
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
            "metrics": _stats,
            **({"raw_reply": raw} if args.heard_reply else {}),
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
    adapter = args.adapter or (args.node / "weights/adapter.npz")
    mode = ("node_transcribe_instructed" if args.instructed
            else "node_production_transcribe")
    if args.adapter:
        mode = "cross_mounted_streaming_adapter"
    if args.heard_reply:
        mode = "converse_heard_reply_fresh_history_no_perception_context"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "command": [sys.executable, *sys.argv],
        "node": args.node.name, "tier": args.tier,
        "manifest_sha256": sha256(args.manifest.expanduser()),
        "scorer_sha256": sha256(REPO / "training/evaluate.py"),
        "system": converse.NN_SYSTEM if args.heard_reply else args.system,
        "adapter": str(adapter), "split": args.manifest.parent.name,
        "adapter_sha256": sha256(adapter),
        "encoder": "openai/whisper-medium (node-local snapshot)",
        "evaluation_mode": mode,
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
