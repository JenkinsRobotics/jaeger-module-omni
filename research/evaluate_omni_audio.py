"""Independent ASR evaluator for benchmark_omni_audio.py artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import graded  # noqa: E402


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def edit_distance(a: list[str], b: list[str]) -> int:
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
    ap.add_argument("results", type=Path)
    ap.add_argument("--model", default="base.en")
    args = ap.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    report = json.loads(args.results.read_text())
    for row in report["rows"]:
        segments, _ = model.transcribe(row["output_pcm"], beam_size=5,
                                       vad_filter=False)
        heard = " ".join(s.text.strip() for s in segments).strip()
        ref = words(row["gemma_reply_log_only"])
        hyp = words(heard)
        wer = edit_distance(ref, hyp) / max(len(ref), 1)
        row["metric_transcript"] = heard
        row["semantic_ok"] = graded(heard, row["answers"])
        row["output_wer"] = round(wer, 4)
        row["output_word_accuracy"] = round(max(0.0, 1.0 - wer), 4)
        print(f"{row['case']} {'OK' if row['semantic_ok'] else 'MISS'} "
              f"word-acc {row['output_word_accuracy']:.0%}  {heard!r}")
    n = len(report["rows"])
    semantic = sum(r["semantic_ok"] for r in report["rows"])
    mean_acc = sum(r["output_word_accuracy"] for r in report["rows"]) / max(n, 1)
    report["score"] = {
        "semantic_correct": semantic, "n": n,
        "semantic_ratio_to_legacy_12_of_12": semantic / max(n, 1),
        "mean_output_word_accuracy": round(mean_acc, 4),
        "median_first_audio_ms": round(statistics.median(
            r["first_audio_ms"] for r in report["rows"]), 1),
        "streaming_output": all(r["stream_chunks"] > 1 for r in report["rows"]),
        "strict_80_percent_pass": (semantic / max(n, 1) >= 0.8 and
                                    mean_acc >= 0.8 and
                                    all(r["stream_chunks"] > 1
                                        for r in report["rows"]))}
    args.results.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["score"], indent=2))
    return 0 if report["score"]["strict_80_percent_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
