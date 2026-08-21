"""Benchmark the single MiniCPM-remapped audio adapter against spoken cases.

Runtime graph: PCM -> MiniCPM-tuned Whisper encoder -> one remapped projector
-> Gemma. No Whisper decoder, transcript, tokenizer round-trip, or text hint is
present. Generated text is a diagnostic output and may later condition Talker
through the tensor-only speech bridge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LEGACY_IMPL = (REPO.parent / "VoiceLLM-Playground" / "Omni Model" /
               "gemma-backbone" / "train_audio_adapter.py")
DEFAULT_MODEL = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research"))

from grading import graded
from jaeger_agent_omni.app import read_wav


def load_front_end_source(path: Path):
    spec = importlib.util.spec_from_file_location("remapped_audio_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--cases", type=Path, default=REPO / "research/cases12")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--front-end-source", type=Path, default=LEGACY_IMPL)
    args = ap.parse_args()

    source = load_front_end_source(args.front_end_source)
    front_end = source.AudioFrontEnd(args.device)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.generate import generate_step

    model, tokenizer = load(str(args.model), lazy=False)
    model.freeze(); model.eval()
    marker = source.AUDIO_MARKER
    rendered = tokenizer.apply_chat_template([
        {"role": "system", "content": (
            "You are a voice assistant. Answer the spoken question directly "
            "in one short sentence.")},
        {"role": "user", "content": marker},
    ], tokenize=False, add_generation_prompt=True)
    if marker not in rendered:
        raise RuntimeError("audio marker disappeared from Gemma chat template")
    prefix, suffix = rendered.split(marker, 1)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    saved = np.load(args.adapter)
    if saved["weight"].shape != (2560, 4096):
        raise ValueError(f"wrong adapter shape {saved['weight'].shape}")

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4096, 2560, bias=True)
            self.proj.weight = mx.array(saved["weight"].astype(np.float32))
            self.proj.bias = mx.array(saved["bias"].astype(np.float32))

        def __call__(self, x):
            return self.proj(x)

    adapter = Adapter()
    cases = json.loads((args.cases / "cases.json").read_text())[:args.n]
    rows = []
    for case in cases:
        wav = args.cases / f"{case['key']}.wav"
        t0 = time.monotonic()
        cached = front_end(read_wav(wav))
        encode_ms = (time.monotonic() - t0) * 1000
        ids_list = prefix_ids + [0] * len(cached) + suffix_ids
        ids = mx.array(ids_list, dtype=mx.int32)
        token_embeddings = model.language_model.model.embed_tokens(ids[None, :])[0]
        audio = adapter(mx.array(cached)) / math.sqrt(2560)
        start = len(prefix_ids)
        embeddings = mx.concatenate([
            token_embeddings[:start], audio,
            token_embeddings[start + len(cached):]], axis=0)
        t0 = time.monotonic()
        generated = []
        for token, _ in generate_step(ids, model, input_embeddings=embeddings,
                                      max_tokens=args.max_tokens):
            if token in (1, 106):
                break
            generated.append(token)
        mx.synchronize()
        generate_ms = (time.monotonic() - t0) * 1000
        reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
        ok = graded(reply, case["answers"])
        rows.append({"case": case["key"], "answers": case["answers"],
                     "ok": ok, "reply": reply,
                     "input_pcm": str(wav.resolve()),
                     "audio_tokens": int(cached.shape[0]),
                     "encode_project_ms": round(encode_ms, 1),
                     "generate_ms": round(generate_ms, 1),
                     "input_samples_received": len(read_wav(wav)),
                     "transcript_handoff": False})
        print(f"{case['key']} {'OK' if ok else 'MISS'} "
              f"{encode_ms:.0f}+{generate_ms:.0f} ms {reply!r}", flush=True)

    correct = sum(row["ok"] for row in rows)
    report = {
        "format": "jaeger-single-remapped-audio-input-v1",
        "architecture": {
            "graph": "PCM->MiniCPM_tuned_Whisper->one_remapped_projector->Gemma",
            "projector": "ReLU(1024->4096 frozen)->AvgPool5->4096->2560 trained",
            "transcript_or_decoded_text_handoff": False,
            "adapter": str(args.adapter.resolve()),
        },
        "score": {"correct": correct, "n": len(rows),
                  "semantic_accuracy": correct / max(1, len(rows))},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["score"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
