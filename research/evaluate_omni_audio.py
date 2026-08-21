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
    number_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three",
        "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
        "12": "twelve", "13": "thirteen", "14": "fourteen",
        "15": "fifteen", "16": "sixteen", "17": "seventeen",
        "18": "eighteen", "19": "nineteen", "20": "twenty",
        "21": "twenty one", "22": "twenty two", "23": "twenty three",
        "24": "twenty four", "25": "twenty five"}
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    expanded = []
    for token in tokens:
        expanded.extend(number_words.get(token, token).split())
    return expanded


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
    semantic_gate = semantic / max(n, 1) >= 0.8
    intelligibility_gate = mean_acc >= 0.8
    streaming_gate = all(r.get("talker_streaming", False) and
                         r["stream_chunks"] > 1 for r in report["rows"])
    lossless_gate = all(r.get("input_samples_received") ==
                        r.get("input_samples_consumed")
                        for r in report["rows"])
    wall_clock_gate = all(r.get("wall_clock_paced_input", False)
                          for r in report["rows"])
    barge_in_gate = all(r.get("barge_in_tested", False)
                        for r in report["rows"])
    legacy_median = report.get("legacy_median_first_audio_ms")
    measured_median = statistics.median(r["first_audio_ms"]
                                        for r in report["rows"])
    latency_gate = (legacy_median is not None and
                    measured_median <= 1.25 * legacy_median)
    provenance_gate = bool(report.get("provenance_complete", False))
    core_gate = semantic_gate and intelligibility_gate and streaming_gate
    report["score"] = {
        "semantic_correct": semantic, "n": n,
        "semantic_ratio_to_legacy_12_of_12": semantic / max(n, 1),
        "mean_output_word_accuracy": round(mean_acc, 4),
        "median_first_audio_ms": round(measured_median, 1),
        "streaming_output": streaming_gate,
        "lossless_input": lossless_gate,
        "wall_clock_paced_input": wall_clock_gate,
        "output_barge_in_tested": barge_in_gate,
        "legacy_relative_latency_pass": latency_gate,
        "provenance_complete": provenance_gate,
        "core_audio_quality_streaming_pass": core_gate,
        "strict_80_percent_pass": (core_gate and lossless_gate and
                                    wall_clock_gate and barge_in_gate and
                                    latency_gate and provenance_gate)}
    args.results.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["score"], indent=2))
    return 0 if report["score"]["strict_80_percent_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
