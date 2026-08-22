"""Cache local spoken Q/A at the single-projector pre-linear2 boundary.

The split is by question, not by waveform, so held-out evaluation cannot be
passed by hearing the same question in another synthetic voice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCE = (REPO.parent / "VoiceLLM-Playground" / "Omni Model" /
          "gemma-backbone" / "train_audio_adapter.py")
DEFAULT_CACHE = Path.home() / "models/gemma-audio-adapter/cache"
HOLDOUT_QUESTIONS = {
    "what is the capital of france?",
    "what is the largest planet?",
    "who wrote romeo and juliet?",
}

sys.path.insert(0, str(REPO))
from jaeger_agent_omni.app import read_wav


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("audio_adapter_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path,
                    default=REPO / "training/corpus_mixed")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--device", choices=("mps", "cpu"), default="mps")
    ap.add_argument("--front-end-source", type=Path, default=SOURCE)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (args.corpus / "turns.jsonl").read_text().splitlines() if line]
    rows = [row for row in rows if row.get("response")]
    source = load_source(args.front_end_source)
    front_end = source.AudioFrontEnd(args.device)
    counts = {"jaeger_qa_train": 0, "jaeger_qa_validation": 0}
    outputs = {}
    for split in counts:
        root = args.cache_dir / split
        (root / "features").mkdir(parents=True, exist_ok=True)
        outputs[split] = (root, (root / "manifest.jsonl").open("w"))
    try:
        for i, row in enumerate(rows):
            question = row["transcript"].strip()
            split = ("jaeger_qa_validation" if question.lower() in HOLDOUT_QUESTIONS
                     else "jaeger_qa_train")
            root, manifest = outputs[split]
            wav = args.corpus / row["audio_path"]
            features = front_end(read_wav(wav))
            name = f"{row['turn_id']}.npy"
            np.save(root / "features" / name, features.astype(np.float16))
            manifest.write(json.dumps({
                "id": row["turn_id"],
                "text": row["response"].strip(),
                "question": question,
                "seconds": row["duration_s"],
                "frames": int(features.shape[0]),
                "feature": f"features/{name}",
                "source_pcm": str(wav.resolve()),
            }) + "\n")
            counts[split] += 1
            if (i + 1) % 50 == 0:
                print(f"cached {i + 1}/{len(rows)}", flush=True)
    finally:
        for _, manifest in outputs.values():
            manifest.close()
    print(json.dumps({"counts": counts,
                      "held_out_questions": sorted(HOLDOUT_QUESTIONS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
