#!/usr/bin/env python3
"""Train the four-way speech/music/noise/silence head on a frozen adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.context_distill import refuse_gpu_contention
from training.perception_model import SCENE_CLASSES, build, load_arrays, save


CACHE = Path.home() / "models/jaeger-training-data/stock-whisper-medium-scenes-v1"
DEFAULT_ADAPTER = Path("training/runs/audio-perception-mixed-cont1500.step01500.npz")


def rows(split: str) -> list[dict]:
    root = CACHE / f"scenes_{split}"
    result = []
    for line in (root / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        feature = Path(row["feature"])
        if not feature.is_absolute():
            feature = root / feature
        if feature.is_file():
            row["feature"] = str(feature)
            result.append(row)
    return result


def metrics(adapter, items: list[dict]) -> dict:
    import mlx.core as mx

    truth, predicted = [], []
    for row in items:
        feature = np.load(row["feature"]).astype(np.float32)
        _, logits = adapter.perceive(mx.array(feature))
        score = np.asarray(logits["scene"], dtype=np.float32).mean(axis=0)
        truth.append(int(row["scene_index"]))
        predicted.append(int(score.argmax()))
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    confusion = np.zeros((len(SCENE_CLASSES), len(SCENE_CLASSES)), dtype=int)
    for actual, guess in zip(truth, predicted):
        confusion[actual, guess] += 1
    per_class = {}
    f1s = []
    for index, name in enumerate(SCENE_CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": f1, "support": int(confusion[index].sum())}
    return {
        "clips": len(items), "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(np.mean(f1s)), "classes": per_class,
        "confusion_actual_rows_predicted_columns": confusion.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save", type=Path,
                        default=Path("training/runs/audio-perception-v2.npz"))
    parser.add_argument("--report", type=Path,
                        default=Path("training/runs/audio-perception-v2-scene.json"))
    args = parser.parse_args()
    refuse_gpu_contention()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    arrays = load_arrays(args.adapter)
    adapter, hidden = build(arrays)
    adapter.freeze()
    adapter.head_scene.unfreeze()
    trainable = sum(value.size for _, value in tree_flatten(
        adapter.trainable_parameters()))
    expected = hidden * len(SCENE_CLASSES) + len(SCENE_CLASSES)
    if trainable != expected:
        raise RuntimeError(f"trainable {trainable}, expected {expected}")
    train, validation, test = rows("train"), rows("validation"), rows("test")
    print(f"scene classes {SCENE_CLASSES}")
    print(f"train={len(train)} validation={len(validation)} test={len(test)}")
    print(f"trainable {trainable:,} parameters; audio/event trunk frozen")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)

    def loss_fn(module, feature, target):
        _, logits = module.perceive(mx.array(feature))
        z = logits["scene"]
        y = mx.full((z.shape[0],), int(target), dtype=mx.int32)
        picked = mx.take_along_axis(z, y[:, None], axis=-1)[:, 0]
        return mx.mean(mx.logsumexp(z, axis=-1) - picked)

    value_and_grad = nn.value_and_grad(adapter, loss_fn)
    rng = np.random.default_rng(0)
    losses = []
    evaluations = []
    started = time.time()
    for step in range(1, args.steps + 1):
        row = train[int(rng.integers(len(train)))]
        feature = np.load(row["feature"]).astype(np.float32)
        loss, gradients = value_and_grad(
            adapter, feature, int(row["scene_index"]))
        optimizer.update(adapter, gradients)
        mx.eval(adapter.parameters(), optimizer.state, loss)
        losses.append(float(loss))
        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} loss={np.mean(losses[-100:]):.4f} "
                  f"{(time.time()-started)/step:.3f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = metrics(adapter, validation)
            evaluations.append({"step": step, **result})
            print(f"  validation accuracy={result['accuracy']:.3f} "
                  f"macro-F1={result['macro_f1']:.3f}", flush=True)

    destination = save(args.save, adapter, hidden)
    test_result = metrics(adapter, test)
    saved = load_arrays(destination)
    changed = []
    for key, before in arrays.items():
        if "." not in key or key.startswith("head_scene."):
            continue
        after = saved.get(key)
        if after is None or not np.array_equal(before, after):
            changed.append(key)
    if changed:
        raise RuntimeError(f"frozen parameters changed: {changed[:5]}")
    report = {
        "kind": "jaeger_audio_scene_head_training",
        "classes": list(SCENE_CLASSES), "steps": args.steps, "lr": args.lr,
        "source_adapter": str(args.adapter.resolve()),
        "output_adapter": str(destination.resolve()),
        "trainable_parameters": trainable,
        "source_file_disjoint": True,
        "frozen_non_scene_parameters_bit_identical": True,
        "validation": evaluations, "test": test_result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"test accuracy={test_result['accuracy']:.3f} "
          f"macro-F1={test_result['macro_f1']:.3f}")
    print(f"saved {destination}\nreport {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
