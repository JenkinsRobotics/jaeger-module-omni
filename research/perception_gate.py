#!/usr/bin/env python3
"""Held-out event metrics for the causal audio-perception heads.

Thresholds are selected once on speaker-disjoint synthetic and meeting-disjoint
AMI validation folds, then applied unchanged to a second speaker fold and a
second set of meetings. Frame accuracy is intentionally absent: a model
predicting "nothing" would exceed 99% on rare speaker-change labels while being
useless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.context_distill import refuse_gpu_contention
from training.perception_model import HEADS, build, load_arrays


SHARED = Path.home() / "models/jaeger-training-data/gemmaomni-cache-v1"
DEFAULT_ADAPTER = (
    REPO / "jaeger_agent_omni/nodes/jaegeromni-41m-causal-balanced/"
    "weights/adapter.npz")


def report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def stable_fold(value: str, folds: int = 6) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % folds


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def resolve_feature(row: dict, feature_root: Path) -> str | None:
    value = row.get("feature")
    if not value:
        return None
    direct = Path(value).expanduser()
    if direct.is_file():
        return str(direct)
    relocated = feature_root / "features" / direct.name
    return str(relocated) if relocated.is_file() else None


def synthetic_rows(label_split: str = "bargein_train",
                   feature_split: str = "bargein_feat") -> list[dict]:
    labels = {str(row["id"]): row for row in read_jsonl(
        SHARED / label_split / "manifest.jsonl")}
    features = {str(row["id"]): row for row in read_jsonl(
        SHARED / feature_split / "manifest.jsonl")}
    rows = []
    for item_id in sorted(labels.keys() & features.keys()):
        row = dict(labels[item_id])
        feature = resolve_feature(
            features[item_id], SHARED / feature_split)
        if feature:
            row["feature"] = feature
            speaker_folds = {stable_fold(str(value))
                             for value in row.get("speakers", ())}
            row["speaker_folds"] = sorted(speaker_folds)
            # Both voices must belong to the same held-out fold. Cross-fold
            # pairs are discarded rather than leaking one voice into train.
            row["speaker_fold"] = (next(iter(speaker_folds))
                                   if len(speaker_folds) == 1 else None)
            rows.append(row)
    return rows


def ami_rows(limit: int, folds: set[int] | None = None) -> list[dict]:
    root = SHARED / "ami_overlap_feat"
    rows = []
    for row in read_jsonl(root / "manifest.jsonl"):
        feature = resolve_feature(row, root)
        if feature and all(name in row for name in HEADS):
            row = dict(row)
            row["feature"] = feature
            row["meeting_fold"] = stable_fold(str(row.get("meeting", "")))
            rows.append(row)
    if folds is not None:
        rows = [row for row in rows if row["meeting_fold"] in folds]
    rows.sort(key=lambda row: hashlib.sha256(
        str(row["id"]).encode()).hexdigest())
    return rows[:limit] if limit else rows


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def binary_counts(logits: np.ndarray, labels: np.ndarray,
                  threshold: float) -> dict:
    predicted = sigmoid(logits) >= threshold
    truth = labels >= 0.5
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def choose_threshold(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    candidates = np.linspace(0.05, 0.95, 37)
    scored = [(float(value), binary_counts(logits, labels, float(value)))
              for value in candidates]
    return max(scored, key=lambda item: (item[1]["f1"],
                                         item[1]["precision"]))


def run_rows(adapter, rows: list[dict]) -> dict:
    import mlx.core as mx

    values = {name: {"logits": [], "labels": []} for name in HEADS}
    seconds = 0.0
    speaker_logits, speaker_labels = [], []
    for index, row in enumerate(rows, 1):
        feature = np.load(row["feature"]).astype(np.float32)
        _, logits = adapter.perceive(mx.array(feature))
        for name in HEADS:
            predicted = np.asarray(logits[name], dtype=np.float32)
            truth = np.asarray(row[name], dtype=np.float32)
            n = min(len(predicted), len(truth))
            values[name]["logits"].append(predicted[:n])
            values[name]["labels"].append(truth[:n])
        if row.get("speaker_index") is not None:
            predicted = np.asarray(logits["speaker_id"], dtype=np.float32)
            truth = np.asarray(row["speaker_index"], dtype=np.int64)
            n = min(len(predicted), len(truth))
            speaker_logits.append(predicted[:n])
            speaker_labels.append(truth[:n])
        seconds += float(row.get("seconds", 0.0))
        if index == 1 or index % 100 == 0:
            print(f"  scored {index}/{len(rows)}", flush=True)
    result = {"clips": len(rows), "seconds": seconds, "heads": {}}
    for name, blocks in values.items():
        result["heads"][name] = {
            "logits": np.concatenate(blocks["logits"]),
            "labels": np.concatenate(blocks["labels"]),
        }
    if speaker_logits:
        result["speaker_logits"] = np.concatenate(speaker_logits)
        result["speaker_labels"] = np.concatenate(speaker_labels)
    return result


def speaker_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    predicted = logits.argmax(axis=-1)
    classes = sorted(set(int(value) for value in labels))
    per_class = {}
    f1s = []
    for cls in classes:
        tp = int(np.sum((predicted == cls) & (labels == cls)))
        fp = int(np.sum((predicted == cls) & (labels != cls)))
        fn = int(np.sum((predicted != cls) & (labels == cls)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[str(cls)] = {"precision": precision, "recall": recall,
                               "f1": f1, "support": int(np.sum(labels == cls))}
        f1s.append(f1)
    return {"macro_f1": float(np.mean(f1s)), "classes": per_class}


def summarize(raw: dict, thresholds: dict[str, float]) -> dict:
    report = {"clips": raw["clips"], "seconds": raw["seconds"], "heads": {}}
    for name in HEADS:
        blocks = raw["heads"][name]
        counts = binary_counts(
            blocks["logits"], blocks["labels"], thresholds[name])
        counts["threshold"] = thresholds[name]
        counts["false_positives_per_minute"] = (
            counts["fp"] / max(1e-9, raw["seconds"] / 60.0))
        report["heads"][name] = counts
    if "speaker_logits" in raw:
        report["speaker_id"] = speaker_metrics(
            raw["speaker_logits"], raw["speaker_labels"])
    return report


def evaluate(adapter_path: Path, ami_limit: int) -> dict:
    arrays = load_arrays(adapter_path)
    adapter, _ = build(arrays)
    synthetic = synthetic_rows()
    validation = [row for row in synthetic if row["speaker_fold"] == 1]
    test = [row for row in synthetic if row["speaker_fold"] == 0]
    if len(validation) < 25 or len(test) < 25:
        raise RuntimeError(
            f"speaker-disjoint folds too small: val={len(validation)} test={len(test)}")
    print(f"synthetic speaker-disjoint validation={len(validation)} "
          f"test={len(test)}", flush=True)
    raw_validation = run_rows(adapter, validation)
    ami_validation_rows = ami_rows(ami_limit, {1})
    print(f"real AMI meeting-disjoint validation={len(ami_validation_rows)}",
          flush=True)
    raw_ami_validation = run_rows(adapter, ami_validation_rows)
    thresholds = {}
    validation_summary = {"clips": raw_validation["clips"], "heads": {}}
    for name in HEADS:
        blocks = raw_validation["heads"][name]
        ami_blocks = raw_ami_validation["heads"][name]
        threshold, counts = choose_threshold(
            np.concatenate((blocks["logits"], ami_blocks["logits"])),
            np.concatenate((blocks["labels"], ami_blocks["labels"])))
        thresholds[name] = threshold
        validation_summary["heads"][name] = {
            **counts, "threshold": threshold}
    raw_test = run_rows(adapter, test)
    meetings = ami_rows(ami_limit, {0})
    print(f"real AMI meeting-disjoint test={len(meetings)}", flush=True)
    raw_ami = run_rows(adapter, meetings)
    return {
        "kind": "jaeger_audio_perception_gate",
        "evaluation": "free_running_causal_heads",
        "teacher_forced": False,
        "adapter": report_path(adapter_path),
        "threshold_selection": (
            "synthetic speaker-disjoint plus AMI meeting-disjoint validation"),
        "thresholds": thresholds,
        "synthetic_validation": validation_summary,
        "synthetic_test": summarize(raw_test, thresholds),
        "ami_real": summarize(raw_ami, thresholds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--ami-limit", type=int, default=256)
    parser.add_argument("--out", type=Path,
                        default=Path("research/results/perception-gate.json"))
    args = parser.parse_args()
    refuse_gpu_contention()
    report = evaluate(args.adapter, args.ami_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for corpus in ("synthetic_test", "ami_real"):
        heads = report[corpus]["heads"]
        print(corpus, " · ".join(
            f"{name} F1={heads[name]['f1']:.3f}" for name in HEADS))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
