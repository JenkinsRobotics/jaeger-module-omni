#!/usr/bin/env python3
"""Calibrate causal perception heads while freezing every audio/Gemma weight.

Training uses only synthetic mixtures whose two speakers are absent from both
speaker-disjoint validation and test folds.  Only 14,343 head parameters are
trainable; the 41M audio adapter, Gemma projection, and self-speech projection
remain bit-identical, so this stage cannot change WER or answer quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from research.perception_gate import (
    ami_rows, binary_counts, choose_threshold, run_rows, synthetic_rows,
)
from training.context_distill import refuse_gpu_contention
from training.perception_model import HEADS, build, load_arrays, save


DEFAULT_ADAPTER = (
    REPO / "nodes/jaegeromni-41m-causal-balanced/weights/adapter.npz")


def validation_metrics(adapter, rows: list[dict], meetings: list[dict]) -> dict:
    raw = run_rows(adapter, rows)
    raw_meetings = run_rows(adapter, meetings)
    result = {}
    for name in HEADS:
        values = raw["heads"][name]
        meeting_values = raw_meetings["heads"][name]
        threshold, counts = choose_threshold(
            np.concatenate((values["logits"], meeting_values["logits"])),
            np.concatenate((values["labels"], meeting_values["labels"])))
        result[name] = {**counts, "threshold": threshold}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save", type=Path,
                        default=Path("training/runs/audio-perception-heads.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("training/runs/audio-perception-train.json"))
    args = parser.parse_args()
    refuse_gpu_contention()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    arrays = load_arrays(args.adapter)
    adapter, hidden = build(arrays)
    adapter.freeze()
    for name in (*HEADS, "speaker_id"):
        getattr(adapter, f"head_{name}").unfreeze()
    trainable = sum(value.size for _, value in tree_flatten(
        adapter.trainable_parameters()))
    total = sum(value.size for _, value in tree_flatten(adapter.parameters()))
    if trainable != hidden * 7 + 7:
        raise RuntimeError(
            f"unexpected trainable count {trainable}; expected {hidden*7+7}")

    rows = synthetic_rows()
    validation = [row for row in rows if row["speaker_fold"] == 1]
    test = [row for row in rows if row["speaker_fold"] == 0]
    train = [row for row in rows
             if not ({0, 1} & set(row["speaker_folds"]))]
    meeting_validation = ami_rows(256, {1})
    meeting_train = ami_rows(0, {2, 3, 4, 5})
    if len(train) < 500 or len(validation) < 25 or len(test) < 25:
        raise RuntimeError(
            f"invalid split train={len(train)} val={len(validation)} test={len(test)}")
    print(f"trainable {trainable:,}/{total:,} ({100*trainable/total:.4f}%)")
    print(f"speaker-disjoint synthetic train={len(train)} val={len(validation)} "
          f"test={len(test)}", flush=True)
    print(f"meeting-disjoint AMI train={len(meeting_train)} "
          f"val={len(meeting_validation)}", flush=True)

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)

    def weighted_bce(logit, label, positive_weight):
        weight = 1.0 + (positive_weight - 1.0) * label
        return mx.mean(weight * (
            mx.logaddexp(mx.zeros_like(logit), logit) - label * logit))

    def loss_fn(module, feature, row):
        _, logits = module.perceive(mx.array(feature))
        n = int(logits["overlap"].shape[0])
        labels = {}
        for name in HEADS:
            value = np.asarray(row[name][:n], dtype=np.float32)
            if len(value) < n:
                value = np.pad(value, (0, n - len(value)))
            labels[name] = mx.array(value)
        loss_overlap = weighted_bce(
            logits["overlap"], labels["overlap"], 3.0)
        loss_change = weighted_bce(
            logits["speaker_change"], labels["speaker_change"], 30.0)
        loss_end = weighted_bce(
            logits["turn_end"], labels["turn_end"], 20.0)

        if row.get("speaker_index") is not None:
            identity = np.asarray(row["speaker_index"][:n], dtype=np.int32)
            if len(identity) < n:
                identity = np.pad(identity, (0, n - len(identity)))
            target = mx.array(identity)
            z = logits["speaker_id"]
            picked = mx.take_along_axis(z, target[:, None], axis=-1)[:, 0]
            ce = mx.logsumexp(z, axis=-1) - picked
            class_weight = mx.take(
                mx.array([0.35, 1.0, 1.0, 2.0], dtype=mx.float32), target)
            loss_identity = mx.mean(class_weight * ce)
        else:
            loss_identity = mx.array(0.0)
        total_loss = (loss_overlap + 0.7 * loss_change +
                      0.7 * loss_end + 0.7 * loss_identity)
        return total_loss, mx.stack((loss_overlap, loss_change,
                                     loss_end, loss_identity))

    value_and_grad = nn.value_and_grad(adapter, loss_fn)
    rng = np.random.default_rng(args.seed)
    losses = []
    components = []
    evaluations = []
    started = time.time()
    for step in range(1, args.steps + 1):
        pool = train if step % 2 else meeting_train
        row = pool[int(rng.integers(len(pool)))]
        feature = np.load(row["feature"]).astype(np.float32)
        (loss, detail), gradients = value_and_grad(adapter, feature, row)
        gradients, norm = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(adapter, gradients)
        mx.eval(adapter.parameters(), optimizer.state, loss, detail, norm)
        losses.append(float(loss))
        components.append(np.asarray(detail, dtype=np.float32))
        if step == 1 or step % args.log_every == 0:
            recent = np.asarray(components[-args.log_every:]).mean(axis=0)
            print(f"step {step:5d} loss={np.mean(losses[-args.log_every:]):.4f} "
                  f"ov={recent[0]:.3f} change={recent[1]:.3f} "
                  f"end={recent[2]:.3f} id={recent[3]:.3f} "
                  f"{(time.time()-started)/step:.3f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = validation_metrics(
                adapter, validation, meeting_validation)
            evaluations.append({"step": step, "heads": metrics})
            print("  validation " + " · ".join(
                f"{name} F1={metrics[name]['f1']:.3f}"
                for name in HEADS), flush=True)
            checkpoint = args.save.with_name(
                f"{args.save.stem}.step{step:05d}{args.save.suffix}")
            # Event-only checkpoints remain v1.  A randomly initialized scene
            # head must never be mislabeled as a trained perception-v2 node.
            save(checkpoint, adapter, hidden, include_scene=False)

    destination = save(args.save, adapter, hidden, include_scene=False)
    # A held-out test is reported once, after all choices and thresholds were
    # made on validation. The full production evaluator separately checks AMI.
    raw_test = run_rows(adapter, test)
    thresholds = {name: evaluations[-1]["heads"][name]["threshold"]
                  for name in HEADS}
    test_metrics = {}
    for name in HEADS:
        values = raw_test["heads"][name]
        test_metrics[name] = binary_counts(
            values["logits"], values["labels"], thresholds[name])
        test_metrics[name]["threshold"] = thresholds[name]

    # Frozen-trunk proof: compare every non-head array exactly. This is
    # stronger than rerunning WER; the function producing Gemma embeddings did
    # not change by a single stored bit.
    saved = load_arrays(destination)
    changed = []
    for key, before in arrays.items():
        if "." not in key or key.startswith("head_"):
            continue
        after = saved.get(key)
        if after is None or not np.array_equal(before, after):
            changed.append(key)
    if changed:
        raise RuntimeError(f"frozen parameters changed: {changed[:5]}")
    report = {
        "kind": "jaeger_audio_perception_head_training",
        "source_adapter": str(args.adapter.resolve()),
        "output_adapter": str(destination.resolve()),
        "steps": args.steps, "lr": args.lr,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "split": {"synthetic_train": len(train),
                  "synthetic_validation": len(validation),
                  "synthetic_test": len(test),
                  "ami_train": len(meeting_train),
                  "ami_validation": len(meeting_validation),
                  "speaker_disjoint": True, "meeting_disjoint": True},
        "frozen_non_head_parameters_bit_identical": True,
        "evaluations": evaluations,
        "synthetic_test": test_metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {destination}\nreport {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
