"""Measure speech information retained by the single remapped adapter.

This uses Gemma's greedy lexical readout only as a measurement instrument. It
does not ask Gemma to answer or reason, and it does not feed the decoded text
back into runtime. The headline metric is held-out word error rate (WER).
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
DEFAULT_CACHE = Path.home() / "models/gemma-audio-adapter/cache"
DEFAULT_MODEL = Path.home() / "models/gemma-4-E4B-it-4bit-mlx"


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("audio_adapter_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1,
                           row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=SOURCE)
    args = ap.parse_args()

    source = load_source(args.source)
    rows = source._manifest_examples(args.cache_dir, args.split)[:args.n]
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
    total_edits = 0
    total_words = 0
    exact = 0
    results = []
    started = time.monotonic()
    for i, row in enumerate(rows):
        cached = np.load(row["feature"]).astype(np.float32)
        ids_list = prefix_ids + [0] * len(cached) + suffix_ids
        ids = mx.array(ids_list, dtype=mx.int32)
        token_embeddings = model.language_model.model.embed_tokens(ids[None, :])[0]
        audio = adapter(mx.array(cached)) / math.sqrt(2560)
        start = len(prefix_ids)
        embeddings = mx.concatenate([
            token_embeddings[:start], audio,
            token_embeddings[start + len(cached):]], axis=0)
        generated = []
        for token, _ in generate_step(ids, model, input_embeddings=embeddings,
                                      max_tokens=args.max_tokens):
            if token in (1, 106):
                break
            generated.append(token)
        hypothesis = tokenizer.decode(generated, skip_special_tokens=True).strip()
        ref, hyp = words(row["text"]), words(hypothesis)
        edits = distance(ref, hyp)
        total_edits += edits
        total_words += len(ref)
        exact += int(ref == hyp)
        results.append({"id": row["id"], "reference": row["text"],
                        "hypothesis": hypothesis, "edits": edits,
                        "reference_words": len(ref)})
        if i < 5 or (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} edits={edits} {hypothesis!r}", flush=True)
    report = {
        "format": "jaeger-remapped-adapter-information-v1",
        "architecture": {
            "graph": "cached_MiniCPM_Whisper->one_remapped_projector->Gemma_lexical_readout",
            "transcript_runtime_handoff": False,
            "adapter": str(args.adapter.resolve()),
        },
        "score": {
            "utterances": len(results), "reference_words": total_words,
            "word_edits": total_edits,
            "wer": total_edits / max(1, total_words),
            "word_retention": max(0.0, 1.0 - total_edits / max(1, total_words)),
            "exact_utterances": exact,
            "exact_ratio": exact / max(1, len(results)),
            "elapsed_s": round(time.monotonic() - started, 1),
        },
        "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["score"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
